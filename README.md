# HASH + PFFT Token Miners (Rust + GPU)

Workspace berisi **dua binary mining** untuk dua token PoW di Ethereum mainnet,
satu codebase, satu OpenCL keccak256 kernel.

| Binary | Token | Contract |
|---|---|---|
| `hash-miner-rs` | HASH Token | `0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc` |
| `pfft-miner-rs` | PFFT (Pow Free Fair Token) | `0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB` |

Kedua-duanya pakai:
- **GPU (OpenCL)** sebagai default backend — keccak256 kernel di
  `src/keccak_kernel.cl` di-share antara dua binary
- **CPU thread pool** sebagai fallback
- [`alloy`](https://github.com/alloy-rs/alloy) untuk RPC + signing + EIP-1559 tx
- Satu file `.env` untuk konfigurasi

## Kenapa Rust + GPU?

| Hal | Python (`pfft_miner.py`) / JS (`miner.js`) | Rust + GPU (di sini) |
|---|---|---|
| Hashing | single-thread Python keccak | OpenCL kernel, ribuan work-item paralel |
| Per-attempt cost | overhead Python loop | murni GPU, RPC dipoll seperlunya |
| Hash rate (laptop) | ~175 k H/s (Python) | ratusan juta H/s di GPU diskrit |
| Hash rate (CPU fallback) | — | ratusan ribu – jutaan H/s |
| Binary | butuh runtime + deps | satu binary statis |

## Requirements

- Rust toolchain >= 1.85 — install via [rustup.rs](https://rustup.rs)
- **OpenCL runtime** untuk GPU mode:
  - Linux: `sudo apt install ocl-icd-opencl-dev opencl-headers` + vendor driver
    (NVIDIA `nvidia-opencl-icd`, AMD ROCm, Intel `intel-opencl-icd`)
  - Windows: NVIDIA / AMD / Intel driver biasanya sudah ship `OpenCL.dll`.
    Untuk MSVC linker, sebuah `vendor/OpenCL.lib` lokal dipakai (lihat `build.rs`).
- Wallet Ethereum dengan ETH untuk gas (sangat sedikit per mint)
- RPC endpoint (default publik — ganti ke Alchemy/Infura kalau bisa)

## Build

```bash
cargo build --release
```

Hasil:
- `target/release/hash-miner-rs` — HASH miner
- `target/release/pfft-miner-rs` — PFFT miner

Untuk build tanpa OpenCL (CPU-only):

```bash
cargo build --release --no-default-features
```

## Konfigurasi

Copy `.env.example` ke `.env`, isi `PRIVATE_KEY`, atur RPC/GPU sesuai mesin.
Semua nilai juga bisa di-pass via env var di shell.

| Variable | Default | Dipakai oleh | Keterangan |
|---|---|---|---|
| `PRIVATE_KEY` | *(prompt)* | semua | 0x + 64 hex |
| `RPC_URL` | `https://eth.llamarpc.com` | hash-miner-rs | RPC URL |
| `ETH_RPC` | `https://ethereum-rpc.publicnode.com` | pfft-miner-rs | RPC URL (fallback `RPC_URL`) |
| `MINER_THREADS` | num CPU | semua | thread CPU fallback |
| `GPU` | `1` (pfft), opsional (hash) | semua | `1`=wajib, `auto`=fallback CPU, `0`=CPU only |
| `GPU_BATCH` | `4194304` (2^22) | semua | nonce per dispatch |
| `PRIORITY_GWEI` | `2` (pfft) / `5` (hash) | semua | tip EIP-1559 |
| `MAX_FEE_GWEI` | `100` | semua | ceiling EIP-1559 |
| `GAS_LIMIT_OVERRIDE` | — | semua | kalau di-set, override estimate |
| `PAUSE_BETWEEN_ROUNDS` | `5` | pfft-miner-rs | jeda antar mint (detik) |

## Run

### PFFT miner

```bash
# Wajib GPU (abort kalau init gagal)
PRIVATE_KEY=0xabc... GPU=1 ./target/release/pfft-miner-rs

# GPU opsional, fall back ke CPU silently
PRIVATE_KEY=0xabc... GPU=auto ./target/release/pfft-miner-rs

# CPU saja
PRIVATE_KEY=0xabc... GPU=0 ./target/release/pfft-miner-rs

# Prompt interaktif (private key tidak echo)
./target/release/pfft-miner-rs
```

### HASH miner

```bash
# Default = GPU off (untuk backwards-compat dengan versi sebelum 0.2)
PRIVATE_KEY=0xabc... ./target/release/hash-miner-rs

# Aktifkan GPU
PRIVATE_KEY=0xabc... GPU=1 ./target/release/hash-miner-rs
```

## Cara kerja mining

### PFFT

1. Baca `currentPowChallenge(miner)` (bytes32) dari kontrak
2. Baca `currentPowHexZeros()` → tentukan target =
   `(2^256 - 1) >> (hex_zeros * 4)`
3. GPU dispatch: tiap work-item compute `keccak256(challenge || nonce_be_u256)`,
   nonce = `nonce_base + global_id`
4. Solusi valid kalau `hash <= target`
5. Verify dengan `isValidPow(miner, nonce)` (view), lalu submit `freeMint(nonce)`
6. Loop sampai wallet cap (10,000 PFFT) atau max supply (21M PFFT) tercapai

### HASH

1. Baca `getChallenge(miner)` dari kontrak (epoch = blockNumber / 100)
2. Baca `currentDifficulty()`
3. GPU dispatch: tiap work-item compute `keccak256(challenge || nonce_be_u256)`
4. Solusi valid kalau `hash < difficulty`
5. Submit `mine(nonce)` — gagal kalau epoch sudah pindah
6. Watchdog polls block tiap 15 detik, restart round saat epoch berubah

Kernel OpenCL-nya **sama persis** untuk dua-duanya — hanya formula
target→difficulty yang beda di host side (PFFT pakai `<=`, HASH pakai `<`).

## Run as systemd service

```bash
# Build dulu:
cargo build --release

# Setup di host target:
sudo mkdir -p /root/.hermes/workspace/pfft-miner
sudo cp target/release/pfft-miner-rs /root/.hermes/workspace/pfft-miner/
sudo cp .env /root/.hermes/workspace/pfft-miner/   # isi PRIVATE_KEY dll
sudo cp pfft-miner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pfft-miner

# Logs
sudo journalctl -u pfft-miner -f
```

## Self-test GPU

Setiap startup di GPU mode, kernel di-self-test melawan CPU reference
keccak256 pada nonce yang diketahui. Kalau hasilnya beda, binary abort —
gak ada kemungkinan kernel bug nyampe ke produksi.

## Stop

`Ctrl+C` — signal worker buat berhenti setelah attempt sekarang, lalu print
ringkasan akhir.

## Security

- Jangan commit `.env` atau private key.
- RPC publik bisa rate-limit / di-MITM. Pakai punyamu kalau bisa.
- Mining butuh ETH untuk gas. Kalau gas-mu kurang, tx revert.

## License

MIT — risiko ditanggung sendiri.
