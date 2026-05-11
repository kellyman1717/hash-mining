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
mod gpu;

const HASH_CONTRACT_ADDRESS: Address = address!("AC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc");
const DEFAULT_RPC_URL: &str = "https://eth.llamarpc.com";
const EPOCH_BLOCKS: u64 = 100; // matches contract constant
const EPOCH_POLL_INTERVAL: Duration = Duration::from_secs(15);
const STATS_INTERVAL: Duration = Duration::from_secs(2);
const ERA_MINTS: u64 = 100_000;

sol! {
    #[sol(rpc)]
    contract HashToken {
        // Public storage getters
        function currentDifficulty() external view returns (uint256);
        function totalMints() external view returns (uint256);
        function totalMiningMinted() external view returns (uint256);
        function genesisComplete() external view returns (bool);

        // Helpers
        function getChallenge(address miner) external view returns (bytes32);
        function epochBlocksLeft() external view returns (uint256);
        function currentReward() external view returns (uint256);
        function miningState() external view returns (
            uint256 era,
            uint256 reward,
            uint256 difficulty,
            uint256 minted,
            uint256 remaining,
            uint256 epoch,
            uint256 epochBlocksLeft
        );

        // Mining entry-point: NO challenge arg, contract recomputes it.
        function mine(uint256 nonce) external;

        // ERC20
        function totalSupply() external view returns (uint256);
    }
}

struct Solution {
    nonce: U256,
    epoch: u64,
}

#[inline]
fn check_proof(challenge: &B256, nonce: U256, difficulty: U256) -> bool {
    // Contract: keccak256(abi.encode(bytes32 challenge, uint256 nonce))
    // For bytes32 + uint256, abi.encode produces exactly 64 packed bytes
    // (each field is already 32 bytes, no padding needed).
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge.as_slice());
    buf[32..].copy_from_slice(&nonce.to_be_bytes::<32>());
    let hash = keccak256(buf);
    U256::from_be_bytes::<32>(hash.0) < difficulty
}

/// Run N CPU workers in a scoped thread pool. Each thread iterates nonces with
/// stride = num_threads so no two threads ever hash the same nonce.
fn run_workers(
    challenge: B256,
    difficulty: U256,
    epoch: u64,
    start_nonce: U256,
    stop_flag: Arc<AtomicBool>,
    attempts_counter: Arc<AtomicU64>,
    num_threads: usize,
) -> Option<Solution> {
    let solution_slot: Mutex<Option<Solution>> = Mutex::new(None);
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
                    if check_proof(&challenge, nonce, difficulty) {
                        let mut slot = solution_slot.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(Solution { nonce, epoch });
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

fn reward_for_total_mints(total_mints: U256) -> u128 {
    let era = total_mints / U256::from(ERA_MINTS);
    if era < U256::from(64u64) {
        100u128 >> era.to::<u128>().min(63)
    } else {
        0
    }
}

fn hex_short(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(16);
    for b in bytes.iter().take(8) {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Load .env if present (ignore if missing — env vars still work).
    let _ = dotenvy::dotenv();

    println!("🔐 HASH Token CPU Miner (Rust)");
    println!("================================\n");

    let raw_key = match std::env::var("PRIVATE_KEY") {
        Ok(v) => v,
        Err(_) => {
            println!("⚠️  No PRIVATE_KEY env var found.");
            rpassword::prompt_password("Private Key: ")?
        }
    };
    let key_trimmed = raw_key.trim().trim_start_matches("0x");
    if key_trimmed.len() != 64 {
        return Err(eyre!("Invalid private key length (expected 64 hex chars)"));
    }
    let signer: PrivateKeySigner = key_trimmed.parse()?;
    let miner_address = signer.address();
    let wallet = EthereumWallet::from(signer);

    let rpc_url_str = std::env::var("RPC_URL").unwrap_or_else(|_| DEFAULT_RPC_URL.to_string());
    let provider = ProviderBuilder::new()
        .with_recommended_fillers()
        .wallet(wallet)
        .on_http(rpc_url_str.parse()?);

    let contract = HashToken::new(HASH_CONTRACT_ADDRESS, provider.clone());

    let num_threads = std::env::var("MINER_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(num_cpus::get);

    println!("🔨 HASH Miner initialized");
    println!("📍 Miner Address: {}", miner_address);
    println!("⛽ RPC URL: {}", rpc_url_str);
    println!("🧵 Worker threads: {}", num_threads);

    // --- Initial info via miningState() (one RPC call instead of four) ---
    match contract.miningState().call().await {
        Ok(s) => {
            println!("\n📋 Mining State:");
            println!("   Era: {}", s.era);
            println!("   Reward: {} (raw, 1e18)", s.reward);
            println!("   Difficulty: {}", s.difficulty);
            println!("   Mining minted: {} / {}", s.minted, s.minted + s.remaining);
            println!("   Current epoch: {}", s.epoch);
            println!("   Blocks left in epoch: {}", s.epochBlocksLeft);
        }
        Err(e) => eprintln!("⚠️  miningState() failed: {e}"),
    }

    // Refuse to mine if genesis isn't complete — saves gas on guaranteed reverts.
    match contract.genesisComplete().call().await {
        Ok(g) if !g._0 => {
            return Err(eyre!(
                "Genesis is not complete yet — mining is closed. Check https://hash256.org/mine"
            ));
        }
        Ok(_) => println!("✅ Genesis complete — mining is open"),
        Err(e) => eprintln!("⚠️  Could not verify genesisComplete: {e}"),
    }

    // --- Optional GPU backend ---
    let gpu_enabled = std::env::var("GPU").ok().as_deref() == Some("1");
    #[cfg(feature = "gpu")]
    let gpu_miner: Option<Arc<gpu::GpuMiner>> = if gpu_enabled {
        let batch = std::env::var("GPU_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        match gpu::GpuMiner::new(batch) {
            Ok(g) => {
                println!("🎮 GPU device: {}", g.device_name());
                println!("🎮 GPU batch size: {} nonces/dispatch", g.batch_size());
                match g.self_test() {
                    Ok(()) => println!("✅ GPU self-test passed"),
                    Err(e) => return Err(eyre!("GPU self-test FAILED — aborting: {e}")),
                }
                Some(Arc::new(g))
            }
            Err(e) => {
                eprintln!("⚠️  GPU init failed, falling back to CPU: {e}");
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_miner: Option<()> = {
        if gpu_enabled {
            eprintln!("⚠️  GPU=1 set but binary built without the `gpu` feature. Using CPU.");
        }
        None
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                println!("\n🛑 Ctrl-C received, stopping after current attempt...");
                shutdown.store(true, Ordering::Relaxed);
            }
        });
    }

    let session_start = Instant::now();
    let mut session_attempts: u64 = 0;
    let mut success_count: u64 = 0;

    while !shutdown.load(Ordering::Relaxed) {
        // --- Determine current epoch from block number ---
        let block_num = match provider.get_block_number().await {
            Ok(n) => n,
            Err(e) => {
                eprintln!("❌ RPC error fetching block number: {e}");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        let epoch: u64 = block_num / EPOCH_BLOCKS;

        // --- Fetch challenge from contract (avoids any encoding mistakes) ---
        let challenge = match contract.getChallenge(miner_address).call().await {
            Ok(v) => v._0,
            Err(e) => {
                eprintln!("❌ RPC error fetching challenge: {e}");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        let difficulty = match contract.currentDifficulty().call().await {
            Ok(v) => v._0,
            Err(e) => {
                eprintln!("❌ RPC error fetching difficulty: {e}");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        let backend = if gpu_miner.is_some() { "GPU" } else { "CPU" };
        println!("\n📊 Round start:");
        println!("   Block: {}  Epoch: {}", block_num, epoch);
        println!("   Difficulty: {}", difficulty);
        println!("   Challenge: 0x{}...", hex_short(challenge.as_slice()));
        println!("⛏️  Mining epoch {} on {} ({} threads)...", epoch, backend, num_threads);

        // u64 random start works for both backends; CPU widens via U256::from.
        let start_nonce_u64: u64 = rand::thread_rng().gen();
        let start_nonce = U256::from(start_nonce_u64);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let attempts_counter = Arc::new(AtomicU64::new(0));

        // --- Watchdog: stats display + epoch poll via block number ---
        let watchdog = {
            let stop_flag = Arc::clone(&stop_flag);
            let attempts_counter = Arc::clone(&attempts_counter);
            let shutdown = Arc::clone(&shutdown);
            let provider = provider.clone();
            let target_epoch = epoch;
            let round_start = Instant::now();
            tokio::spawn(async move {
                let mut last_print = Instant::now();
                let mut last_attempts: u64 = 0;
                let mut last_poll = Instant::now();
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
                            "\r⚡ {:>10.2} H/s | round {:>6.0}s | attempts {:>14}",
                            rate, elapsed, total
                        );
                        last_attempts = total;
                        last_print = Instant::now();
                    }

                    if last_poll.elapsed() >= EPOCH_POLL_INTERVAL {
                        last_poll = Instant::now();
                        match provider.get_block_number().await {
                            Ok(bn) => {
                                let cur_epoch = bn / EPOCH_BLOCKS;
                                if cur_epoch != target_epoch {
                                    eprintln!(
                                        "\n🔄 Epoch changed {} -> {} (block {}), restarting round",
                                        target_epoch, cur_epoch, bn
                                    );
                                    stop_flag.store(true, Ordering::Relaxed);
                                    break;
                                }
                            }
                            Err(e) => eprintln!("\n⚠️  block poll failed: {e}"),
                        }
                    }
                }
            })
        };

        // --- Mining (GPU if available, else CPU thread pool) ---
        let mining_result: Option<Solution> = {
            let stop_flag = Arc::clone(&stop_flag);
            let attempts_counter = Arc::clone(&attempts_counter);

            #[cfg(feature = "gpu")]
            {
                if let Some(g) = gpu_miner.as_ref().cloned() {
                    let res = tokio::task::spawn_blocking(move || {
                        g.mine(challenge, difficulty, start_nonce_u64, stop_flag, attempts_counter)
                    })
                    .await?;
                    match res {
                        Ok(Some(nonce_u64)) => Some(Solution {
                            nonce: U256::from(nonce_u64),
                            epoch,
                        }),
                        Ok(None) => None,
                        Err(e) => {
                            eprintln!("❌ GPU mining error: {e}");
                            None
                        }
                    }
                } else {
                    tokio::task::spawn_blocking(move || {
                        run_workers(
                            challenge, difficulty, epoch, start_nonce,
                            stop_flag, attempts_counter, num_threads,
                        )
                    })
                    .await?
                }
            }

            #[cfg(not(feature = "gpu"))]
            {
                let _ = &gpu_miner; // silence unused
                tokio::task::spawn_blocking(move || {
                    run_workers(
                        challenge, difficulty, epoch, start_nonce,
                        stop_flag, attempts_counter, num_threads,
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

        let Some(sol) = mining_result else {
            continue;
        };

        println!("🎉 FOUND VALID NONCE: {} (epoch {})", sol.nonce, sol.epoch);

        // --- Gas tuning (EIP-1559) ---
        // PRIORITY_GWEI: tip you actually pay miners (default 5 gwei — aggressive).
        // MAX_FEE_GWEI:  absolute ceiling; you only pay this if base_fee spikes.
        //                Effective gas = min(MAX_FEE, base_fee + PRIORITY).
        // GAS_LIMIT_OVERRIDE: optional fixed gas limit.
        let priority_gwei: f64 = std::env::var("PRIORITY_GWEI")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5.0);
        let max_fee_gwei: f64 = std::env::var("MAX_FEE_GWEI")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100.0);
        let priority_wei = (priority_gwei * 1e9) as u128;
        let max_fee_wei = (max_fee_gwei * 1e9) as u128;

        let mut tx = contract
            .mine(sol.nonce)
            .max_priority_fee_per_gas(priority_wei)
            .max_fee_per_gas(max_fee_wei);
        if let Some(g) = std::env::var("GAS_LIMIT_OVERRIDE")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            tx = tx.gas(g);
        }
        println!(
            "💸 Gas: priority={priority_gwei} gwei, maxFee={max_fee_gwei} gwei (ceiling)"
        );

        match tx.send().await {
            Ok(pending) => {
                let tx_hash = *pending.tx_hash();
                println!("📋 Transaction submitted: {tx_hash}");
                println!("⏳ Waiting for confirmation...");
                match pending.with_required_confirmations(1).get_receipt().await {
                    Ok(receipt) => {
                        if receipt.status() {
                            println!(
                                "✅ SUCCESS! Confirmed in block {}",
                                receipt.block_number.unwrap_or_default()
                            );
                            println!("🔗 https://etherscan.io/tx/{tx_hash}");
                            success_count += 1;
                            if let Ok(total_mints) = contract.totalMints().call().await {
                                println!(
                                    "🏆 Mined ~{} HASH tokens",
                                    reward_for_total_mints(total_mints._0)
                                );
                                println!(
                                    "📈 Total successful mints this session: {success_count}"
                                );
                            }
                        } else {
                            println!("❌ Transaction reverted (status=0)");
                        }
                    }
                    Err(e) => eprintln!("❌ Receipt error: {e}"),
                }
            }
            Err(e) => eprintln!("❌ Submission failed: {e}"),
        }
    }

    let elapsed = session_start.elapsed().as_secs_f64().max(0.001);
    let rate = session_attempts as f64 / elapsed;
    println!("\n📊 Final Statistics:");
    println!("   Total Attempts: {session_attempts}");
    println!("   Successful Mints: {success_count}");
    println!("   Average Hash Rate: {rate:.2} H/s");
    println!("   Mining Duration: {elapsed:.2} seconds");
    Ok(())
}
