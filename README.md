# machine-to-win 🏆

**Sistem Analisis Trading XAUUSD M5 — Berbasis Aturan (Rule-Based), Tanpa AI**

Alat bantu pribadi untuk menganalisis peluang entry trading pada instrumen **XAUUSD (Gold/USD)** di timeframe **M5**. Sistem membaca data candle langsung dari MetaTrader 5 desktop, menghitung indikator teknikal, mengevaluasi kondisi entry melalui logika if-else murni, lalu menampilkan satu keputusan akhir — **BUY, SELL, atau WAIT** — beserta level Stop Loss, Take Profit, dan Risk-Reward Ratio.

Sistem ini adalah **otomatisasi dari framework analisis manual**: trend (EMA 9/21), RSI sebagai filter, serta SL berbasis ATR dan struktur swing. Tidak ada machine learning, tidak ada model prediksi, tidak ada black box — semua keputusan bisa ditelusuri alasannya baris per baris.

---

## ⚠️ Disclaimer

> Alat ini **tidak menjamin profit**. Sistem ini murni rule-based — ia hanya membaca kondisi indikator dan memberi output berdasarkan aturan yang telah didefinisikan. Kondisi pasar yang tidak terbaca oleh indikator (likuiditas, sentimen, news, manipulasi) **tidak diperhitungkan**.
>
> Gunakan sebagai **alat bantu analisis pribadi**, bukan sebagai sinyal trading otomatis. Keputusan trading tetap sepenuhnya di tangan kamu. Tidak ada AI/machine learning di dalam sistem ini.

---

## Arsitektur Sistem (5 Lapisan)

```
┌─────────────────────────────────────────────────────────────────┐
│  [MT5 Desktop]  →  [data_fetcher]  →  [indicators]             │
│       ↓                  ↓                  ↓                   │
│  Koneksi lokal    Candle OHLC M5    EMA 9/21, RSI 14,          │
│  (sudah login)    (hanya closed)    ATR 14, Trend               │
│                                           ↓                     │
│                                    [rule_engine]                 │
│                                           ↓                     │
│                               BUY / SELL / WAIT                 │
│                               + breakdown alasan                 │
│                                           ↓                     │
│                                   [risk_manager]                 │
│                                           ↓                     │
│                               SL, TP, RRR (hybrid ATR+Swing)   │
│                                           ↓                     │
│                                      [web/app.py]               │
│                                           ↓                     │
│                               Browser → hasil analisis           │
└─────────────────────────────────────────────────────────────────┘
```

| # | Lapisan | File | Fungsi |
|---|---------|------|--------|
| 1 | Data Candle | `engine/data_fetcher.py` | Koneksi ke MT5 desktop, ambil data OHLC, jaminan hanya candle closed |
| 2 | Indicator Engine | `engine/indicators.py` | Hitung EMA 9/21, RSI 14, ATR 14, deteksi trend (UPTREND / DOWNTREND / SIDEWAYS) |
| 3 | Rule Engine | `engine/rule_engine.py` | Evaluasi kondisi entry, output BUY/SELL/WAIT + breakdown alasan tiap kondisi |
| 4 | Risk Manager | `engine/risk_manager.py` | Hitung SL (hybrid ATR + Swing High/Low), TP berdasarkan RRR minimum |
| 5 | Web UI | `web/app.py` | Server Flask — satu tombol "Analisis Sekarang", tampilkan keputusan + breakdown |

---

## Logika Entry (Ringkasan Framework)

### Kondisi Entry

Sistem mengevaluasi dua hal secara berurutan:

**1. Trend + EMA Alignment** *(kondisi utama)*

| Label Trend | Syarat | Sinyal |
|-------------|--------|--------|
| `UPTREND` | EMA 9 > EMA 21 **DAN** Close > EMA 21 | Kandidat BUY |
| `DOWNTREND` | EMA 9 < EMA 21 **DAN** Close < EMA 21 | Kandidat SELL |
| `SIDEWAYS` | Kondisi lain | WAIT — tidak ada arah jelas |

**2. Filter RSI** *(veto, bukan sinyal utama)*

RSI bukan sinyal entry — ia hanya memblokir entry jika harga sudah terlalu ekstrem:

| Zona RSI | Kondisi | Efek |
|----------|---------|------|
| RSI > 70 (Overbought) | Saat kandidat BUY | ❌ Blokir BUY |
| RSI < 30 (Oversold) | Saat kandidat SELL | ❌ Blokir SELL |
| RSI 30–70 (Netral) | Semua arah | ✅ Entry tetap jalan |

### Kalkulasi SL/TP — Hybrid ATR + Swing

```
SL Versi ATR   = Entry ± (1.5 × ATR_14)
SL Versi Swing = Swing Low/High terdekat ± $0.50 buffer

SL Final (BUY)  = min(sl_atr, sl_swing)   ← yang lebih RENDAH (lebih jauh)
SL Final (SELL) = max(sl_atr, sl_swing)   ← yang lebih TINGGI (lebih jauh)

TP = Entry ± (jarak_SL × RRR minimum)     ← default RRR = 2.0
```

Jika tidak ada swing ditemukan dalam 50 candle terakhir, sistem otomatis fallback ke ATR saja tanpa error.

---

## Struktur Folder

```
machine-to-win/
├── engine/
│   ├── __init__.py
│   ├── data_fetcher.py       # Lapisan 1: Koneksi MT5, ambil data candle OHLC
│   ├── indicators.py         # Lapisan 2: EMA 9/21, RSI 14, ATR 14, deteksi trend
│   ├── rule_engine.py        # Lapisan 3: Evaluasi kondisi entry (if-else)
│   └── risk_manager.py       # Lapisan 4: Kalkulasi SL, TP, RRR
│
├── web/
│   ├── app.py                # Lapisan 5: Flask backend + routing
│   ├── templates/
│   │   └── index.html        # Tampilan web UI
│   └── static/
│       └── style.css         # Styling halaman
│
├── scripts/
│   ├── fetch_candles.py      # Test koneksi MT5 dan lihat data candle
│   ├── test_indicators.py    # Test kalkulasi semua indikator
│   ├── test_rule_engine.py   # Test logika rule engine
│   ├── test_risk_manager.py  # Test kalkulasi SL/TP/RRR
│   ├── compare_swing_wing.py # Diagnostik: perbandingan hasil swing dengan wing berbeda
│   └── _diag_timezone.py     # Diagnostik: investigasi isu timezone MT5
│
├── tests/
│   └── test_data_fetcher.py  # Unit test koneksi MT5
│
├── data/                     # Data sementara (tidak di-commit)
├── .env                      # Konfigurasi lokal — TIDAK di-commit ke git
├── .env.example              # Template konfigurasi (aman untuk di-commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prasyarat

- **Python** 3.10.6 atau lebih baru
- **MetaTrader 5** versi desktop untuk Windows (bukan MT5 Mobile) — sudah terinstall dan **sudah login** ke akun broker (demo maupun real, keduanya bisa)
- **pip** (package manager Python)

> MT5 Mobile tidak didukung. Library `MetaTrader5` hanya bisa berkomunikasi dengan proses MT5 desktop yang berjalan di komputer yang sama.

---

## Setup dari Nol

### 1. Clone Repository

```bash
git clone https://github.com/<username>/machine-to-win.git
cd machine-to-win
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Library yang akan terinstall:

| Library | Versi | Fungsi |
|---------|-------|--------|
| `MetaTrader5` | 5.0.5735 | Koneksi ke MT5 desktop |
| `pandas` | 2.2.2 | Olah data candle sebagai DataFrame |
| `numpy` | ≥2.0.0 | Kalkulasi numerik |
| `ta` | 0.11.0 | Kalkulasi RSI (Wilder's Smoothing, identik dengan MT5) |
| `python-dotenv` | 1.0.1 | Baca konfigurasi dari file `.env` |
| `tabulate` | 0.9.0 | Tampilan tabel di terminal |
| `Flask` | 3.0.3 | Web server untuk UI |

> **Catatan:** `pandas-ta` **tidak dipakai** karena tidak support Python 3.10 via PyPI. EMA dihitung via `pandas.ewm()` (identik dengan MT5), RSI via library `ta`.

### 3. Buat File `.env`

Salin file template dan sesuaikan:

```bash
copy .env.example .env
```

Isi file `.env`:

```env
MT5_SYMBOL=XAUUSD          # Nama simbol di MT5 (bisa berbeda per broker)
MT5_TIMEFRAME=M5
MT5_CANDLE_COUNT=500        # 500 candle M5 ≈ 41 jam data
```

**Cara cek nama simbol yang benar:** Buka MT5 → View → Market Watch → klik kanan simbol gold → Properties → lihat nama simbol. Beberapa broker menggunakan `XAUUSDm`, `GOLD`, atau varian lain.

### 4. Login ke MetaTrader 5

Buka aplikasi MT5 desktop dan login ke akun broker kamu (demo atau real). **MT5 harus tetap terbuka** selama sistem berjalan — Python berkomunikasi ke MT5 melalui socket lokal tanpa perlu username/password ulang.

---

## Cara Menjalankan

### A. Test Modul Satu per Satu (via `scripts/`)

Jalankan dari root folder `machine-to-win/`. Pastikan MT5 sudah terbuka dan login.

```bash
# Lapisan 1: Test koneksi MT5 dan lihat data candle
python scripts/fetch_candles.py

# Lapisan 2: Test kalkulasi semua indikator (EMA, RSI, ATR, trend)
python scripts/test_indicators.py

# Lapisan 3: Test logika rule engine (kondisi entry, output BUY/SELL/WAIT)
python scripts/test_rule_engine.py

# Lapisan 4: Test kalkulasi SL/TP/RRR
python scripts/test_risk_manager.py
```

Output yang diharapkan dari `fetch_candles.py`:

```
✅ Koneksi ke MT5 berhasil!
   MT5 build    : xxxx
   Path terminal: C:\...
📊 Menarik 500 candle XAUUSD M5 (hanya yang closed)...
✅ Data berhasil ditarik: 500 candle (semua sudah closed)
   Candle terbaru: 2026-07-25 04:15:00+00:00

=== 10 CANDLE TERBARU ===
[tabel data candle OHLC]
```

### B. Jalankan Web UI

```bash
python web/app.py
```

Buka browser dan akses: **http://localhost:5000**

Klik tombol **"Analisis Sekarang"** untuk menjalankan analisis. Sistem akan:
1. Menyambungkan ke MT5
2. Menarik 500 candle XAUUSD M5
3. Menghitung semua indikator
4. Mengevaluasi kondisi entry
5. Menampilkan keputusan + breakdown alasan + SL/TP/RRR (jika BUY/SELL)
6. Memutuskan koneksi MT5

---

## Troubleshooting

| Error | Kemungkinan Penyebab | Solusi |
|-------|---------------------|--------|
| `MT5 initialization failed` | MT5 desktop belum terbuka | Buka MT5 dan login ke akun broker |
| `Symbol XAUUSD not found` | Nama simbol berbeda di broker | Cek nama simbol di Market Watch, update `.env` |
| `No data returned` | Simbol tidak aktif di Market Watch | MT5 → klik kanan Market Watch → Show All → cari dan tambah simbol |
| `Login failed` | MT5 belum login ke akun | Login manual di MT5 desktop terlebih dahulu |
| Error saat pip install | Versi Python tidak kompatibel | Pastikan Python 3.10.6+ |

---

## Known Considerations

Beberapa keputusan teknis penting yang perlu diketahui siapa pun yang membaca atau mengembangkan kode ini:

### 1. Data Candle: Hanya Candle Closed (`start_pos=1`)

Semua penarikan data di `data_fetcher.py` menggunakan `copy_rates_from_pos(..., start_pos=1, ...)`, bukan `start_pos=0`.

**Kenapa ini penting:**
Candle ke-0 di MT5 adalah candle yang *sedang terbentuk* — high, low, dan close-nya masih bergerak setiap detik. Jika dianalisis, sistem bisa menghasilkan sinyal BUY atau SELL yang kemudian berubah saat candle itu menutup. Ini disebut *repainting signal* — sangat berbahaya untuk trading karena memberi ilusi akurasi yang tidak nyata.

Dengan `start_pos=1`, candle aktif otomatis dilewati. Seluruh codebase memakai konvensi: **`df.iloc[-1]` = candle closed paling baru, bukan candle yang sedang berjalan.**

### 2. Validasi Waktu: Referensi Waktu Server Broker

Waktu candle divalidasi menggunakan `mt5.symbol_info_tick().time` (waktu server broker), **bukan** `datetime.now()` atau jam komputer lokal.

**Kenapa ini penting:**
Library `MetaTrader5` mengembalikan timestamp dengan label timezone yang menyesatkan — tercatat sebagai UTC, padahal isinya adalah waktu server broker (misalnya UTC+3/EEST untuk sebagian besar broker Eropa). Jika kita bandingkan timestamp candle dengan `datetime.now(UTC)`, kita akan salah hitung selisih waktu 3 jam, yang bisa membuat sistem keliru menganggap candle sudah closed padahal belum (atau sebaliknya).

Solusi: selalu gunakan `mt5.symbol_info_tick().time` sebagai referensi "jam sekarang" untuk konsistensi dengan sumber data MT5.

> File diagnostik lengkap ada di `scripts/_diag_timezone.py` — berisi investigasi dan perbandingan berbagai cara baca timestamp MT5.

### 3. Swing Detection: `SWING_WING=5` (bukan 3)

Parameter `SWING_WING` di `risk_manager.py` — jumlah candle kiri dan kanan yang dibutuhkan untuk mengkonfirmasi sebuah swing — diatur ke **5**, bukan nilai default awal 3.

**Kenapa ini penting:**

| Wing | Window | Karakteristik |
|------|--------|---------------|
| `wing=3` | 35 menit | Banyak swing ditemukan, tapi sebagian adalah noise — lembah kecil yang tidak terlihat jelas secara visual |
| `wing=5` | 55 menit (~1 jam) | **Sweet spot** — menyaring noise, masih menemukan swing yang genuinely signifikan |
| `wing=8` | 85 menit | Terlalu ketat — melewatkan swing yang jelas di chart, terlalu sering fallback ke ATR |

`wing=5` dipilih setelah membandingkan hasil dengan data real XAUUSD M5 menggunakan `scripts/compare_swing_wing.py`. SL yang dihasilkan lebih bermakna secara visual dan lebih konsisten dengan analisis manual.

---

## Scope MVP Saat Ini

Ini adalah batasan yang disengaja — bukan kekurangan:

| Aspek | Status Saat Ini |
|-------|----------------|
| Instrumen | XAUUSD saja |
| Timeframe | M5 saja |
| Mode | Manual (klik tombol, bukan live monitoring) |
| Histori | Tidak ada — setiap analisis berdiri sendiri |
| Deployment | Lokal saja, belum ke server publik |
| Konteks non-teknikal | Tidak membaca news, sentimen, atau fundamental |
| Database | Belum ada |

---

## Roadmap

Fitur yang direncanakan tapi belum dikerjakan (urutan prioritas kasar):

- [ ] **Candle Pattern Detection** — deteksi pola seperti engulfing, pin bar, doji sebagai kondisi entry tambahan
- [ ] **Support/Resistance Level** — deteksi dan evaluasi level S/R dari struktur historis
- [ ] **Multi-Timeframe Trend** — konfirmasi trend dari H1 atau H4 sebelum entry di M5
- [ ] **Live Monitoring Mode** — polling otomatis setiap candle baru closed, bukan harus klik manual
- [ ] **Histori Analisis** — simpan setiap hasil analisis ke database lokal (SQLite)
- [ ] **Backtest Sederhana** — evaluasi apakah sinyal yang dihasilkan profitabel di data historis
- [ ] **Notifikasi** — kirim alert ke Telegram/Discord saat ada sinyal BUY/SELL
- [ ] **Multi-Symbol / Multi-Timeframe** — extend ke instrumen dan timeframe lain

---

## Cara Berkontribusi / Extend Kondisi Baru

Untuk menambah kondisi entry baru ke rule engine:

1. Buat fungsi baru di `engine/rule_engine.py`:
   ```python
   def _check_nama_kondisi(signals: dict) -> dict:
       # return dict dengan: "terpenuhi", "arah", "keterangan"
   ```

2. Panggil fungsi tersebut di dalam `evaluate_entry()` dan masukkan ke `kondisi_entry`:
   ```python
   kondisi_entry = [
       ("trend_and_ema", c_trend),
       ("nama_kondisi",  _check_nama_kondisi(signals)),  # ← tambah di sini
   ]
   ```

3. Update `MINIMUM_CONDITIONS_MET` jika jumlah kondisi yang harus terpenuhi berubah.

4. Di `web/app.py` bagian `_build_result()`, tambahkan kondisi baru ke `kondisi_list` agar tampil di UI.

---

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Data source | MetaTrader 5 Desktop (via library `MetaTrader5`) |
| Backend | Python 3.10.6+, Flask 3.0.3 |
| Data manipulation | pandas 2.2.2, numpy ≥2.0.0 |
| Indikator teknikal | pandas `.ewm()` (EMA), library `ta` v0.11.0 (RSI) |
| Frontend | HTML + Vanilla CSS (tanpa framework) |
| Konfigurasi | `python-dotenv` via file `.env` |

---

*Dibuat sebagai project belajar — dari framework analisis manual ke sistem otomatis rule-based.*
