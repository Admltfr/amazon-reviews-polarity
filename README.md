# Amazon Reviews Polarity — Klasifikasi Sentimen dengan PySpark

Project Big Data & AI yang mengklasifikasikan ulasan produk Amazon menjadi **negatif** atau **positif** menggunakan **PySpark**. Alur kerjanya mencakup perbandingan metode ingestion data (RDD vs DataFrame), eksplorasi dan pembersihan data, *feature engineering*, visualisasi, lalu pemodelan **TF-IDF + Linear SVM** (perbandingan unigram vs bigram).

## Daftar Isi

- [Struktur Project](#struktur-project)
- [Prasyarat](#prasyarat)
- [Cara Startup](#cara-startup)
- [Alur Notebook](#alur-notebook)
- [Tentang Dataset](#tentang-dataset)
- [Troubleshooting](#troubleshooting)

## Struktur Project

```
amazon-reviews-polarity/
├── datasets/                  # (dibuat manual, di-ignore git) letakkan train.csv & test.csv di sini
├── images/                    # Gambar hasil visualisasi (EDA, n-gram, panjang teks)
├── test/                      # Versi notebook yang dipecah per tahap
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_exploration_and_preprocessing.ipynb
│   ├── 03_model_pipelines_and_evaluation.ipynb
│   └── requirements.txt
├── main.ipynb                 # Notebook utama (end-to-end)
├── requirements.txt           # Dependensi Python
└── README.md
```

> Folder `datasets/` tercantum di `.gitignore`, jadi **tidak ikut ter-clone**. Folder ini harus dibuat sendiri dan diisi dengan dataset (lihat langkah 4).

## Prasyarat

| Kebutuhan | Keterangan |
|-----------|------------|
| **Python** | 3.10+ (notebook dikembangkan dengan Python 3.12) |
| **Java (JDK)** | JDK 17 atau 21 — wajib untuk menjalankan Spark. Pastikan `java -version` berjalan |
| **RAM** | Minimal 8 GB disarankan. Spark dikonfigurasi `spark.driver.memory = 4g` |
| **Disk** | Beberapa GB kosong untuk dataset dan direktori sementara Spark |
| **Jupyter** | Jupyter Notebook/Lab, atau VS Code dengan ekstensi Jupyter |

Cek versi yang terpasang:

```bash
python --version
java -version
```

## Cara Startup

### 1. Clone repository

```bash
git clone https://github.com/Admltfr/amazon-reviews-polarity.git
cd amazon-reviews-polarity
```

### 2. Buat dan aktifkan virtual environment

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
pip install notebook      # jika belum punya Jupyter
```

Dependensi utama: `pyspark==4.3.0`, `pandas==2.2.3`, `matplotlib==3.10.1`, `matplotlib-inline==0.1.7`, `scikit-learn==1.6.1`.

> Sel inisialisasi di `main.ipynb` juga menjalankan `%pip install -r requirements.txt`, tetapi tetap disarankan memasangnya lebih dulu di virtual environment.

### 4. Unduh dataset

1. Buka halaman dataset di Figshare:
   [Amazon Review Polarity — Figshare](https://figshare.com/articles/dataset/Amazon_Review_Polarity/13232501?file=25484027)
2. Unduh file dataset-nya. Jika berbentuk arsip (`.zip` / `.tgz`), ekstrak terlebih dahulu.
3. Buat folder `datasets/` di root project, lalu pindahkan `train.csv` dan `test.csv` ke dalamnya:

**Linux / macOS**

```bash
mkdir -p datasets
mv /path/ke/hasil-ekstrak/train.csv datasets/
mv /path/ke/hasil-ekstrak/test.csv  datasets/
```

**Windows (PowerShell)**

```powershell
mkdir datasets
Move-Item C:\path\ke\hasil-ekstrak\train.csv datasets\
Move-Item C:\path\ke\hasil-ekstrak\test.csv  datasets\
```

Struktur akhirnya harus seperti ini:

```
amazon-reviews-polarity/
└── datasets/
    ├── train.csv
    └── test.csv
```

### 5. Jalankan notebook

```bash
jupyter notebook main.ipynb
```

Atau buka `main.ipynb` di VS Code, pilih kernel dari `.venv`, lalu **Run All**.

> ⚠️ Jalankan sel **secara berurutan dari atas**. Variabel `spark` dan `sc` dibuat di sel inisialisasi, sehingga sel di bawahnya akan error (`NameError`) jika sel tersebut dilewati.

Jika ingin menjalankan per tahap, gunakan notebook di folder `test/` (urutan `01` → `02` → `03`). Notebook tersebut membaca data dari `../datasets/`, jadi `datasets/` tetap harus berada di root project.

## Alur Notebook

1. **Inisialisasi**: membuat `SparkSession` (mode `local[*]`, driver memory 4 GB).
2. **Data Ingestion**: membandingkan waktu baca data dengan RDD API vs DataFrame API.
3. **Data Exploration**: skema, statistik label, pengecekan *missing value* dan duplikat.
4. **Data Cleaning**: hapus null dan duplikat, `trim`, serta filter teks tidak valid.
5. **Feature Engineering**: kolom `text_length`, `length_category` (Short/Medium/Long), `sentiment`, dan `full_text` (gabungan title + text).
6. **Visualisasi**: distribusi sentimen, panjang teks, dan n-gram (hasil tersimpan di `images/`).
7. **Model**: `Tokenizer → (NGram) → HashingTF → IDF → Normalizer → LinearSVC`, dibandingkan untuk unigram (n=1) dan bigram (n=2).
8. **Evaluasi**: accuracy, F1, precision, recall, serta accuracy per kategori panjang teks.

## Tentang Dataset

- Ulasan produk Amazon dengan dua kelas: skor 1–2 sebagai **negatif**, skor 4–5 sebagai **positif** (skor 3 tidak disertakan).
- Umumnya terdiri dari 3.600.000 data latih dan 400.000 data uji.
- Kolom: `label` (1 = negatif, 2 = positif), `title`, dan `text`.
- Notebook memberi nama kolom secara manual lewat `.toDF("label", "title", "text")`.

> **Catatan header:** `main.ipynb` membaca `train.csv` dengan `header=True` dan `test.csv` dengan `header=False`. Pastikan file hasil unduhan sesuai dengan itu. Jika `train.csv` kamu tidak memiliki baris header, ubah menjadi `header=False` agar baris pertama tidak terbuang.

## Troubleshooting

**`JAVA_HOME is not set` / Java tidak ditemukan**
Pasang JDK 17 atau 21 dan pastikan `java -version` berjalan. Di Windows, jika masih bermasalah, *uncomment* sel pertama di `main.ipynb` dan isi variabel `JAVA` dengan path JDK kamu.

**`OutOfMemoryError` / Spark berjalan sangat lambat**
- Tutup aplikasi lain yang memakan RAM.
- Naikkan `spark.driver.memory` di sel inisialisasi (misalnya `"6g"`) jika RAM mencukupi.
- Arahkan direktori sementara Spark ke disk yang lega dengan meng-*uncomment* baris `os.makedirs(...)` dan `.config("spark.local.dir", ...)` di sel inisialisasi.

**`Path does not exist: datasets/train.csv`**
Folder `datasets/` belum dibuat atau notebook dijalankan dari direktori yang salah. Jalankan Jupyter dari root project dan cek isi `datasets/`.

**`NameError: name 'spark' is not defined` / `'sc' is not defined`**
Sel inisialisasi belum dijalankan. Jalankan ulang dari atas, atau gunakan *Restart & Run All*.

**Error versi Python / PySpark**
Pastikan kernel Jupyter memakai virtual environment `.venv` tempat dependensi terpasang, bukan Python global.
