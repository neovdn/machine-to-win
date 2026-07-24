# machine-to-win 🏆
**Sistem Analisis Trading XAUUSD — Berbasis Rule (Non-AI)**

Sistem otomatis untuk menganalisis peluang entry trading pada instrumen XAUUSD timeframe M5,
menggunakan framework analisis berbasis aturan (EMA, RSI, S/R, pola candle).

---

## Arsitektur Sistem

```
[MT5 Desktop] → [data_fetcher] → [indicators] → [rule_engine] → [Web UI]
     ↑               ↑               ↑               ↑              ↑
  Data OHLC      Koneksi MT5    EMA, RSI, S/R   Logika if-else  Tampilan hasil
```

## Status Step

| Step | Komponen | Status |
|------|----------|--------|
| 1 | Project setup & koneksi MT5 | ✅ Done |
| 2 | Indicator engine (EMA, RSI, S/R) | 🔲 Planned |
| 3 | Rule engine (logika entry) | 🔲 Planned |
| 4 | Decision output (SL, TP, RRR) | 🔲 Planned |
| 5 | Web UI | 🔲 Planned |

---

## Cara Setup

### 1. Prasyarat
- Python 3.10.6+
- MetaTrader 5 desktop sudah terinstall dan **sudah login** ke akun broker
- Pip (package manager Python)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi

File `.env` sudah ada dengan nilai default. Jika nama simbol di broker kamu berbeda
(misalnya `XAUUSDm` atau `GOLD`), edit file `.env`:

```
MT5_SYMBOL=XAUUSD      # Ganti sesuai nama di MT5 Market Watch
MT5_TIMEFRAME=M5
MT5_CANDLE_COUNT=500
```

**Cara cek nama simbol yang benar:** Buka MT5 → View → Market Watch → klik kanan simbol gold → Properties → lihat nama simbolnya.

### 4. Jalankan Test Koneksi Data

Pastikan MT5 desktop **sudah terbuka dan login** sebelum menjalankan script ini.

```bash
python scripts/fetch_candles.py
```

Output yang diharapkan:
```
✅ Koneksi ke MT5 berhasil
📊 Menarik 500 candle XAUUSD M5...
✅ Data berhasil ditarik: 500 candle

=== 10 CANDLE TERBARU ===
[tabel data candle OHLC]

=== STATISTIK DATA ===
Rentang waktu  : 2024-xx-xx xx:xx  →  2024-xx-xx xx:xx
Jumlah candle  : 500
Harga tertinggi: xxxx.xx
Harga terendah : xxxx.xx
...
```

---

## Troubleshooting Koneksi MT5

| Error | Kemungkinan Penyebab | Solusi |
|-------|---------------------|--------|
| `MT5 initialization failed` | MT5 desktop belum terbuka | Buka MT5 desktop terlebih dahulu |
| `Symbol XAUUSD not found` | Nama simbol berbeda di broker kamu | Cek nama simbol di Market Watch, update `.env` |
| `No data returned` | Simbol tidak aktif di Market Watch | Klik kanan Market Watch → Show All, cari dan tambah simbol |
| `Login failed` | MT5 belum login ke akun | Login manual di MT5 desktop |

---

## Struktur Folder

```
machine-to-win/
├── data/                    # Data sementara (tidak di-commit)
├── engine/
│   ├── __init__.py
│   ├── data_fetcher.py      # ✅ Koneksi & tarik data MT5
│   ├── indicators.py        # 🔲 EMA, RSI, S/R (Step 2)
│   └── rule_engine.py       # 🔲 Logika entry (Step 3)
├── web/                     # 🔲 Web UI (Step 5)
├── tests/
│   └── test_data_fetcher.py # Unit test koneksi MT5
├── scripts/
│   └── fetch_candles.py     # Script test tarik data
├── requirements.txt
├── .env.example
├── .env                     # Konfigurasi lokal (tidak di-commit)
└── README.md
```
