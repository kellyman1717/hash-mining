//! PFFT Miner Bot — Pow Free Fair Token (Ethereum Mainnet)
//!
//! Rust + OpenCL GPU port of the original `pfft_miner.py`.
//!
//! Contract: 0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB
//!
//! Mining algorithm (matches the on-chain contract):
//!   challenge = currentPowChallenge(miner)                 // bytes32, from contract
//!   target    = (2^256 - 1) >> (currentPowHexZeros() * 4)  // upper bound
//!   valid     = keccak256(challenge || nonce_be_u256) <= target
//!
//! Submission: freeMint(uint256 powNonce) — costs gas only, no ETH transferred.
//!
//! Configuration (via env var or `.env`):
//!   PRIVATE_KEY            (required) 0x-prefixed hex private key
//!   ETH_RPC                Ethereum RPC URL (default: publicnode)
//!   GPU                    "1" to enable GPU backend (default: enabled if built)
//!   GPU_BATCH              GPU work-item batch size per dispatch (default 2^22)
//!   MINER_THREADS          CPU worker thread count for fallback (default: num cpus)
//!   PRIORITY_GWEI          EIP-1559 priority tip (default 2)
//!   MAX_FEE_GWEI           EIP-1559 max fee ceiling (default 100)
//!   GAS_LIMIT_OVERRIDE     Override estimated gas limit (default: alloy estimate)
//!   PAUSE_BETWEEN_ROUNDS   Seconds to sleep between rounds (default 5)

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use alloy::network::EthereumWallet;
use alloy::primitives::{address, keccak256, Address, B256, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use eyre::{eyre, Result};
use rand::Rng;

#[cfg(feature = "gpu")]
use hash_miner_rs::gpu;

const PFFT_CONTRACT_ADDRESS: Address = address!("EFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB");
const DEFAULT_RPC_URL: &str = "https://ethereum-rpc.publicnode.com";
const STATS_INTERVAL: Duration = Duration::from_secs(2);
const WALLET_MINT_CAP_PFFT: u128 = 10_000; // 10,000 PFFT per-wallet cap (whole tokens)

sol! {
    #[sol(rpc)]
    contract PfftToken {
        function currentPowHexZeros() external view returns (uint256);
        function totalMinted() external view returns (uint256);
        function MAX_SUPPLY() external view returns (uint256);
        function calculateActualMint(uint256 requested) external view returns (uint256);
        function currentPowChallenge(address user) external view returns (bytes32);
        function isValidPow(address user, uint256 powNonce) external view returns (bool);
        function freeMint(uint256 powNonce) external;
        function mintedByAddress(address user) external view returns (uint256);
        function balanceOf(address account) external view returns (uint256);
    }
}

struct Status {
    hex_zeros: U256,
    total_minted: U256,
    max_supply: U256,
    next_mint: U256,
    wallet_minted: U256,
    wallet_bal: U256,
    target: U256,
}

/// Difficulty `target` such that any keccak256 hash `<= target` is a valid PoW.
///
/// Mirrors the Python reference: `target = (2**256 - 1) >> (hex_zeros * 4)`.
fn target_from_hex_zeros(hex_zeros: U256) -> U256 {
    // hex_zeros is at most 64 (the whole hash) in practice; cap defensively.
    let hz: u32 = hex_zeros.to::<u128>().min(64) as u32;
    let shift = hz.saturating_mul(4);
    if shift == 0 {
        U256::MAX
    } else if shift >= 256 {
        U256::ZERO
    } else {
        U256::MAX >> shift as usize
    }
}

/// Convert the inclusive `target` into the exclusive `difficulty` that the GPU
/// kernel and CPU worker both compare against (they test `hash < difficulty`).
fn target_to_difficulty(target: U256) -> U256 {
    // Valid hash iff hash <= target  iff  hash < target + 1.
    // Saturate at U256::MAX so we never overflow on a degenerate hex_zeros=0.
    target.checked_add(U256::from(1u64)).unwrap_or(U256::MAX)
}

#[inline]
fn check_proof(challenge: &B256, nonce: U256, target: U256) -> bool {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge.as_slice());
    buf[32..].copy_from_slice(&nonce.to_be_bytes::<32>());
    let hash = keccak256(buf);
    U256::from_be_bytes::<32>(hash.0) <= target
}

/// CPU worker pool fallback (used when GPU is unavailable or disabled).
fn run_cpu_workers(
    challenge: B256,
    target: U256,
    start_nonce: U256,
    stop_flag: Arc<AtomicBool>,
    attempts_counter: Arc<AtomicU64>,
    num_threads: usize,
) -> Option<U256> {
    let solution_slot: Mutex<Option<U256>> = Mutex::new(None);
    let stride = U256::from(num_threads);

    std::thread::scope(|s| {
        for tid in 0..num_threads {
            let stop_flag = &stop_flag;
            let attempts_counter = &attempts_counter;
            let solution_slot = &solution_slot;
            s.spawn(move || {
                let mut nonce = start_nonce + U256::from(tid);
                let mut local_attempts: u64 = 0;
                loop {
                    if check_proof(&challenge, nonce, target) {
                        let mut slot = solution_slot.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(nonce);
                        }
                        stop_flag.store(true, Ordering::Relaxed);
                        attempts_counter.fetch_add(local_attempts, Ordering::Relaxed);
                        return;
                    }
                    nonce += stride;
                    local_attempts += 1;

                    if local_attempts & 0x3FFF == 0 {
                        attempts_counter.fetch_add(local_attempts, Ordering::Relaxed);
                        local_attempts = 0;
                        if stop_flag.load(Ordering::Relaxed) {
                            return;
                        }
                    }
                }
            });
        }
    });

    solution_slot.into_inner().ok().flatten()
}

fn hex_short(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(16);
    for b in bytes.iter().take(8) {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn format_eth(wei: U256) -> String {
    let one_eth = U256::from(10u64).pow(U256::from(18u64));
    let whole = wei / one_eth;
    let frac = wei % one_eth;
    // Show up to 6 decimal places.
    let scale = U256::from(10u64).pow(U256::from(12u64));
    let micro_eth = frac / scale; // 0..1_000_000
    format!("{}.{:06}", whole.to::<u128>(), micro_eth.to::<u64>())
}

/// Macro-style helper: pulls the seven on-chain values that make up [`Status`].
///
/// Inlined as a macro instead of a function because the concrete contract
/// instance type from `sol!` carries provider/transport generics that are
/// awkward to thread through a function signature in alloy 0.8.
macro_rules! fetch_status {
    ($contract:expr, $wallet_addr:expr) => {{
        let hex_zeros = $contract.currentPowHexZeros().call().await?._0;
        let total_minted = $contract.totalMinted().call().await?._0;
        let max_supply = $contract.MAX_SUPPLY().call().await?._0;
        // Reference: contract.calculateActualMint(w3.to_wei(1000, 'ether'))
        let probe = U256::from(1000u64) * U256::from(10u64).pow(U256::from(18u64));
        let next_mint = $contract.calculateActualMint(probe).call().await?._0;
        let wallet_minted = $contract.mintedByAddress($wallet_addr).call().await?._0;
        let wallet_bal = $contract.balanceOf($wallet_addr).call().await?._0;
        let target = target_from_hex_zeros(hex_zeros);
        Status {
            hex_zeros,
            total_minted,
            max_supply,
            next_mint,
            wallet_minted,
            wallet_bal,
            target,
        }
    }};
}

fn print_status_header(s: &Status) {
    let one_eth = U256::from(10u64).pow(U256::from(18u64));
    let progress_bp =
        (s.total_minted * U256::from(10_000u64) / s.max_supply).to::<u128>() as f64 / 100.0;
    let hex_zeros = s.hex_zeros.to::<u128>() as u32;
    let difficulty_bits = hex_zeros * 4;

    println!("\nContract:");
    println!(
        "   Minted: {} / {} PFFT ({:.1}%)",
        (s.total_minted / one_eth).to::<u128>(),
        (s.max_supply / one_eth).to::<u128>(),
        progress_bp
    );
    println!(
        "   Next mint: ~{} PFFT",
        (s.next_mint / one_eth).to::<u128>()
    );
    println!(
        "   Difficulty: {} hex zeros ({}-bit)",
        hex_zeros, difficulty_bits
    );
    println!(
        "   Wallet minted: {} / {} PFFT",
        (s.wallet_minted / one_eth).to::<u128>(),
        WALLET_MINT_CAP_PFFT
    );
    println!(
        "   Wallet balance: {} PFFT",
        (s.wallet_bal / one_eth).to::<u128>()
    );
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Load .env if present (ignore if missing — env vars still work).
    let _ = dotenvy::dotenv();

    println!("============================================================");
    println!("  PFFT Miner Bot - Pow Free Fair Token (Rust + GPU)");
    println!("  Contract: {PFFT_CONTRACT_ADDRESS}");
    println!("============================================================\n");

    let raw_key = match std::env::var("PRIVATE_KEY") {
        Ok(v) => v,
        Err(_) => {
            println!("No PRIVATE_KEY env var found.");
            rpassword::prompt_password("Private Key: ")?
        }
    };
    let key_trimmed = raw_key.trim().trim_start_matches("0x");
    if key_trimmed.len() != 64 {
        return Err(eyre!(
            "Invalid private key length (expected 64 hex chars, got {})",
            key_trimmed.len()
        ));
    }
    let signer: PrivateKeySigner = key_trimmed.parse()?;
    let miner_address = signer.address();
    let wallet = EthereumWallet::from(signer);

    let rpc_url_str = std::env::var("ETH_RPC")
        .or_else(|_| std::env::var("RPC_URL"))
        .unwrap_or_else(|_| DEFAULT_RPC_URL.to_string());
    let provider = ProviderBuilder::new()
        .with_recommended_fillers()
        .wallet(wallet)
        .on_http(rpc_url_str.parse()?);

    let contract = PfftToken::new(PFFT_CONTRACT_ADDRESS, provider.clone());

    let num_threads = std::env::var("MINER_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(num_cpus::get);

    println!("PFFT Miner initialized");
    println!("   Miner Address: {miner_address}");
    println!("   RPC URL:       {rpc_url_str}");
    println!("   CPU threads:   {num_threads} (fallback)");

    // Sanity check: connection / current block.
    match provider.get_block_number().await {
        Ok(n) => println!("   Connected. Latest block: {n}"),
        Err(e) => return Err(eyre!("Cannot connect to RPC: {e}")),
    }

    // ETH balance check.
    let eth_bal = provider.get_balance(miner_address).await?;
    println!("   ETH balance:   {} ETH", format_eth(eth_bal));
    let min_gas = U256::from(50_000_000_000_000u64); // 0.00005 ETH
    if eth_bal < min_gas {
        eprintln!("   Warning: Low ETH balance. Need >= 0.00005 ETH for gas.");
    }

    // --- GPU backend ---
    // Default behaviour matches the existing HASH miner: GPU is opt-in via GPU=1.
    // Override with GPU=auto to silently fall back to CPU if GPU init fails.
    let gpu_mode = std::env::var("GPU").unwrap_or_else(|_| "1".to_string());
    let gpu_enabled = matches!(gpu_mode.as_str(), "1" | "true" | "yes" | "auto");
    let gpu_required = matches!(gpu_mode.as_str(), "1" | "true" | "yes");

    #[cfg(feature = "gpu")]
    let gpu_miner: Option<Arc<gpu::GpuMiner>> = if gpu_enabled {
        let batch = std::env::var("GPU_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        match gpu::GpuMiner::new(batch) {
            Ok(g) => {
                println!("   GPU device:    {}", g.device_name());
                println!("   GPU batch:     {} nonces/dispatch", g.batch_size());
                match g.self_test() {
                    Ok(()) => println!("   GPU self-test: passed"),
                    Err(e) => return Err(eyre!("GPU self-test FAILED - aborting: {e}")),
                }
                Some(Arc::new(g))
            }
            Err(e) => {
                if gpu_required {
                    return Err(eyre!(
                        "GPU init failed (set GPU=auto to allow CPU fallback): {e}"
                    ));
                }
                eprintln!("   GPU init failed, falling back to CPU: {e}");
                None
            }
        }
    } else {
        println!("   GPU disabled (GPU={gpu_mode}), using CPU");
        None
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_miner: Option<()> = {
        if gpu_enabled {
            eprintln!(
                "   Warning: GPU requested but binary built without the `gpu` feature. Using CPU."
            );
        }
        None
    };

    // Initial contract status read.
    let s: Status = fetch_status!(contract, miner_address);
    print_status_header(&s);

    let pause_between_rounds: u64 = std::env::var("PAUSE_BETWEEN_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    // How long to wait for a tx receipt before giving up and mining the next
    // round. Stops the miner from hanging forever on a tx that never lands
    // (RPC drop, mempool eviction, etc.). The original tx may still confirm
    // later — we just stop blocking on it.
    let confirmation_timeout_secs: u64 = std::env::var("CONFIRMATION_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(90);

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                println!("\nShutdown signal received, stopping after current attempt...");
                shutdown.store(true, Ordering::Relaxed);
            }
        });
    }

    let one_eth = U256::from(10u64).pow(U256::from(18u64));
    let wallet_cap_wei = U256::from(WALLET_MINT_CAP_PFFT) * one_eth;
    let session_start = Instant::now();
    let mut session_attempts: u64 = 0;
    let mut round_num: u64 = 0;
    let mut total_minted_count: u64 = 0;
    let mut total_pfft_earned_wei = U256::ZERO;

    while !shutdown.load(Ordering::Relaxed) {
        round_num += 1;
        println!("\n------------------------------------------------------------");
        println!("  Round #{round_num}");
        println!("------------------------------------------------------------");

        let s: Status = match async {
            Ok::<Status, eyre::Report>(fetch_status!(contract, miner_address))
        }
        .await
        {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Status error: {e}, retrying in 15s...");
                tokio::time::sleep(Duration::from_secs(15)).await;
                continue;
            }
        };

        let hex_zeros = s.hex_zeros.to::<u128>() as u32;
        let difficulty_bits = hex_zeros * 4;
        let progress =
            (s.total_minted * U256::from(10_000u64) / s.max_supply).to::<u128>() as f64 / 100.0;
        println!(
            "  Supply: {} ({:.1}%) | Next: ~{} PFFT | Diff: {}-bit",
            (s.total_minted / one_eth).to::<u128>(),
            progress,
            (s.next_mint / one_eth).to::<u128>(),
            difficulty_bits
        );

        if s.total_minted >= s.max_supply {
            println!("  Max supply reached!");
            break;
        }
        if s.wallet_minted >= wallet_cap_wei {
            println!("  Wallet cap ({} PFFT) reached!", WALLET_MINT_CAP_PFFT);
            break;
        }

        let challenge = match contract.currentPowChallenge(miner_address).call().await {
            Ok(c) => c._0,
            Err(e) => {
                eprintln!("Challenge fetch error: {e}, retrying in 5s...");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        let backend = if gpu_miner.is_some() { "GPU" } else { "CPU" };
        println!(
            "  Mining ({difficulty_bits}-bit) on {backend}, challenge=0x{}...",
            hex_short(challenge.as_slice())
        );

        let start_nonce_u64: u64 = rand::thread_rng().gen();
        let start_nonce = U256::from(start_nonce_u64);
        let target = s.target;
        let difficulty_for_kernel = target_to_difficulty(target);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let attempts_counter = Arc::new(AtomicU64::new(0));

        // --- Watchdog: live hash rate display ---
        let watchdog = {
            let stop_flag = Arc::clone(&stop_flag);
            let attempts_counter = Arc::clone(&attempts_counter);
            let shutdown = Arc::clone(&shutdown);
            let round_start = Instant::now();
            tokio::spawn(async move {
                let mut last_print = Instant::now();
                let mut last_attempts: u64 = 0;
                loop {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    if shutdown.load(Ordering::Relaxed) {
                        stop_flag.store(true, Ordering::Relaxed);
                        break;
                    }
                    if last_print.elapsed() >= STATS_INTERVAL {
                        let total = attempts_counter.load(Ordering::Relaxed);
                        let delta = total.saturating_sub(last_attempts);
                        let secs = last_print.elapsed().as_secs_f64().max(0.001);
                        let rate = delta as f64 / secs;
                        let elapsed = round_start.elapsed().as_secs_f64();
                        eprint!(
                            "\r  {:>10.2} H/s | round {:>6.0}s | attempts {:>14}",
                            rate, elapsed, total
                        );
                        last_attempts = total;
                        last_print = Instant::now();
                    }
                }
            })
        };

        let solution: Option<U256> = {
            let stop_flag = Arc::clone(&stop_flag);
            let attempts_counter = Arc::clone(&attempts_counter);

            #[cfg(feature = "gpu")]
            {
                if let Some(g) = gpu_miner.as_ref().cloned() {
                    let res = tokio::task::spawn_blocking(move || {
                        g.mine(
                            challenge,
                            difficulty_for_kernel,
                            start_nonce_u64,
                            stop_flag,
                            attempts_counter,
                        )
                    })
                    .await?;
                    match res {
                        Ok(Some(n)) => Some(U256::from(n)),
                        Ok(None) => None,
                        Err(e) => {
                            eprintln!("\n  GPU mining error: {e}");
                            None
                        }
                    }
                } else {
                    tokio::task::spawn_blocking(move || {
                        run_cpu_workers(
                            challenge,
                            target,
                            start_nonce,
                            stop_flag,
                            attempts_counter,
                            num_threads,
                        )
                    })
                    .await?
                }
            }

            #[cfg(not(feature = "gpu"))]
            {
                let _ = &gpu_miner;
                let _ = difficulty_for_kernel;
                tokio::task::spawn_blocking(move || {
                    run_cpu_workers(
                        challenge,
                        target,
                        start_nonce,
                        stop_flag,
                        attempts_counter,
                        num_threads,
                    )
                })
                .await?
            }
        };

        stop_flag.store(true, Ordering::Relaxed);
        let _ = watchdog.await;
        let round_attempts = attempts_counter.load(Ordering::Relaxed);
        session_attempts += round_attempts;
        eprintln!();

        let Some(nonce) = solution else {
            println!("  No solution this round, retrying...");
            continue;
        };

        println!("  FOUND nonce: {nonce}");

        // Verify on-chain before broadcasting to avoid wasted gas if the
        // challenge changed underneath us (e.g. supply ticked over).
        match contract.isValidPow(miner_address, nonce).call().await {
            Ok(v) if !v._0 => {
                eprintln!("  Nonce invalid on-chain (supply changed?), re-mining...");
                continue;
            }
            Ok(_) => {}
            Err(e) => eprintln!("  Verify error: {e}, submitting anyway..."),
        }

        // --- Submit freeMint(nonce) ---
        let priority_gwei: f64 = std::env::var("PRIORITY_GWEI")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0);
        let max_fee_gwei: f64 = std::env::var("MAX_FEE_GWEI")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100.0);
        let priority_wei = (priority_gwei * 1e9) as u128;
        let max_fee_wei = (max_fee_gwei * 1e9) as u128;

        let mut tx = contract
            .freeMint(nonce)
            .max_priority_fee_per_gas(priority_wei)
            .max_fee_per_gas(max_fee_wei);
        if let Some(g) = std::env::var("GAS_LIMIT_OVERRIDE")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            tx = tx.gas(g);
        }
        println!("  Gas: priority={priority_gwei} gwei, maxFee={max_fee_gwei} gwei (ceiling)");

        match tx.send().await {
            Ok(pending) => {
                let tx_hash = *pending.tx_hash();
                println!("  TX:    https://etherscan.io/tx/{tx_hash}");
                println!("  Waiting for confirmation (timeout {confirmation_timeout_secs}s)...");
                let wait_fut = pending.with_required_confirmations(1).get_receipt();
                match tokio::time::timeout(Duration::from_secs(confirmation_timeout_secs), wait_fut)
                    .await
                {
                    Ok(Ok(receipt)) => {
                        if receipt.status() {
                            println!(
                                "  MINT OK | Block {} | Gas {}",
                                receipt.block_number.unwrap_or_default(),
                                receipt.gas_used
                            );
                            total_minted_count += 1;
                            total_pfft_earned_wei += s.next_mint;
                            println!(
                                "  +~{} PFFT (estimated) | Session mints: {}",
                                (s.next_mint / one_eth).to::<u128>(),
                                total_minted_count
                            );

                            // Optional: live PFFT balance.
                            if let Ok(bal) = contract.balanceOf(miner_address).call().await {
                                println!(
                                    "  PFFT balance: {} PFFT",
                                    (bal._0 / one_eth).to::<u128>()
                                );
                            }
                        } else {
                            println!("  REVERTED | Gas {}", receipt.gas_used);
                        }
                    }
                    Ok(Err(e)) => eprintln!("  Receipt error: {e}"),
                    Err(_) => {
                        eprintln!(
                            "  TX still pending after {confirmation_timeout_secs}s — moving on."
                        );
                        eprintln!(
                            "  Tx hash {tx_hash} may still confirm later (check the link above)."
                        );
                        eprintln!("  Tip: try a different ETH_RPC (e.g. https://rpc.mevblocker.io/fast, https://rpc.flashbots.net/fast, https://eth.llamarpc.com).");
                    }
                }
            }
            Err(e) => eprintln!("  TX error: {e}"),
        }

        // Session summary.
        let elapsed_min = session_start.elapsed().as_secs_f64() / 60.0;
        println!(
            "\n  Session: {} mints | ~{} PFFT | {:.1} min",
            total_minted_count,
            (total_pfft_earned_wei / one_eth).to::<u128>(),
            elapsed_min
        );

        if !shutdown.load(Ordering::Relaxed) && pause_between_rounds > 0 {
            println!("  Cooldown {pause_between_rounds}s...");
            tokio::time::sleep(Duration::from_secs(pause_between_rounds)).await;
        }
    }

    let elapsed = session_start.elapsed().as_secs_f64().max(0.001);
    let rate = session_attempts as f64 / elapsed;
    println!("\n============================================================");
    println!("  Session Summary");
    println!("  Mints:        {total_minted_count}");
    println!(
        "  PFFT earned:  ~{}",
        (total_pfft_earned_wei / one_eth).to::<u128>()
    );
    println!("  Attempts:     {session_attempts}");
    println!("  Hash rate:    {rate:.2} H/s (session avg)");
    println!("  Runtime:      {:.1} min", elapsed / 60.0);
    println!("============================================================");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_for_seven_hex_zeros_matches_python() {
        // Python reference: (2**256 - 1) >> 28
        let target = target_from_hex_zeros(U256::from(7u64));
        let expected = U256::MAX >> 28usize;
        assert_eq!(target, expected);
    }

    #[test]
    fn target_zero_hex_zeros_is_u256_max() {
        let target = target_from_hex_zeros(U256::ZERO);
        assert_eq!(target, U256::MAX);
    }

    #[test]
    fn difficulty_for_kernel_is_target_plus_one() {
        let target = target_from_hex_zeros(U256::from(7u64));
        let diff = target_to_difficulty(target);
        // hash <= target  iff  hash < diff
        assert_eq!(diff, target + U256::from(1u64));
    }

    #[test]
    fn proof_check_matches_python_formula() {
        // Use a low-difficulty (high target) to find any small nonce reliably.
        let challenge = B256::ZERO;
        // hex_zeros = 1 -> target has 4 leading zero bits, ~1 in 16 hashes wins.
        let target = target_from_hex_zeros(U256::from(1u64));
        let mut found = None;
        for n in 0u64..1000 {
            if check_proof(&challenge, U256::from(n), target) {
                found = Some(n);
                break;
            }
        }
        assert!(found.is_some(), "expected at least one win in 1000 tries");

        // Cross-check exclusive-difficulty semantics: same nonce, < diff, must hold.
        let diff = target_to_difficulty(target);
        let nonce = U256::from(found.unwrap());
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(challenge.as_slice());
        buf[32..].copy_from_slice(&nonce.to_be_bytes::<32>());
        let h = U256::from_be_bytes::<32>(keccak256(buf).0);
        assert!(h <= target);
        assert!(h < diff);
    }
}
