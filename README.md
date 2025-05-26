
# Laporan Proyek Machine Learning - Deffin Purnama Noer

## Domain Proyek

Proyek ini berfokus pada **prediksi harga Bitcoin**, mata uang kripto terkemuka di dunia. Dalam beberapa tahun terakhir, Bitcoin telah mengalami lonjakan popularitas dan volatilitas harga yang signifikan, menjadikannya topik menarik untuk analisis dan prediksi. Fluktuasi harga Bitcoin yang cepat dapat dipengaruhi oleh berbagai faktor, termasuk sentimen pasar, berita ekonomi global, adopsi teknologi blockchain, regulasi pemerintah, dan dinamika penawaran-permintaan. Bitcoin dikenal memiliki tingkat volatilitas yang jauh lebih tinggi dibandingkan aset investasi tradisional seperti saham atau emas, yang menambah kompleksitas dalam prediksinya.

Kemampuan untuk memprediksi pergerakan harga Bitcoin memiliki implikasi besar bagi investor, _trader_, dan analis pasar. Prediksi yang akurat dapat membantu dalam pengambilan keputusan investasi, manajemen risiko portofolio, dan pengembangan strategi _trading_ yang lebih efektif. Namun, sifat non-linear dan kompleks dari data _time series_ harga Bitcoin menghadirkan tantangan besar dalam membangun model prediksi yang andal. Oleh karena itu, penerapan teknik _machine learning_, khususnya model _deep learning_ yang dirancang untuk data deret waktu, menjadi krusial dalam mengatasi tantangan ini karena kemampuannya dalam mengidentifikasi pola kompleks dan non-linear yang tidak dapat ditangkap oleh metode tradisional.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan:** Masalah prediksi harga Bitcoin perlu diselesaikan karena adanya potensi keuntungan finansial dan mitigasi risiko yang signifikan. Dengan memahami pola dan tren harga di masa lalu, kita dapat membangun model yang belajar dari data tersebut untuk memproyeksikan pergerakan di masa depan. Solusi ini dapat membantu:

-   **Investor:** Mengidentifikasi waktu terbaik untuk membeli atau menjual Bitcoin, memaksimalkan potensi keuntungan.
-   **Trader:** Mengembangkan strategi _trading_ jangka pendek yang lebih menguntungkan.
-   **Institusi Keuangan:** Mengevaluasi risiko investasi terkait Bitcoin dan mengembangkan produk keuangan yang relevan.

**Referensi:**

-   O. Omole and D. Enke, "Deep learning for Bitcoin price direction prediction: models and trading strategies empirically compared," _Financial Innovation_, vol. 10, no. 117, Aug. 2024. [Online]. Available: [https://doi.org/10.1186/s40854-024-00643-1](https://doi.org/10.1186/s40854-024-00643-1)
-   L. Yin, J. Nie, and L. Han, "Understanding cryptocurrency volatility: The role of oil market shocks," _International Review of Economics & Finance_, vol. 72, pp. 233–253, Mar. 2021. [Online]. Available: [https://doi.org/10.1016/j.iref.2020.11.013](https://doi.org/10.1016/j.iref.2020.11.013)
-   L. Pichl and T. Kaizoji, "Volatility analysis of bitcoin," _Quantitative Finance and Economics_, vol. 1, no. 4, pp. 474–485, 2017.

----------

## Business Understanding

### Problem Statements

1.  **Tingginya volatilitas harga Bitcoin** membuat pendekatan tradisional seperti regresi linier atau ARIMA kurang efektif dalam menghasilkan prediksi yang akurat.
2.  **Kurangnya alat bantu prediktif yang cerdas** membuat investor kesulitan dalam mengambil keputusan berbasis data historis yang kompleks dan dinamis.

### Goals

1.  Membangun model _deep learning_ yang mampu memprediksi harga Bitcoin dalam jangka pendek (misalnya 5 hari ke depan), untuk membantu investor mengantisipasi pergerakan pasar.
2.  Mengevaluasi dan membandingkan performa dua arsitektur Recurrent Neural Network (LSTM dan GRU) untuk menemukan model dengan performa terbaik, berdasarkan metrik MSE, RMSE, MAE, dan R².

### Solution Statements

1.  Menerapkan dua model _time series deep learning_: **LSTM dan GRU**, masing-masing dengan arsitektur dua _layer_ dan jumlah neuron yang seimbang.
2.  Melakukan **normalisasi data dan _windowing_ (_lookback_ 30 hari)** untuk membentuk urutan _input_ model.
3.  Menggunakan data harga historis dari Yahoo Finance, dilatih dan diuji dengan pembagian 80:20, dan dievaluasi menggunakan metrik **MSE, RMSE, MAE, dan R²**.
4.  Memilih model terbaik berdasarkan metrik evaluasi dan menginterpretasikan hasil prediksi secara visual melalui grafik aktual vs prediksi.
----------

## Data Understanding

Data yang digunakan dalam proyek ini adalah **riwayat harga Bitcoin (BTC-USD)** yang diambil dari **Yahoo Finance** melalui library **yfinance** di Python. Dataset ini mencakup informasi harga harian Bitcoin selama periode waktu dari **01 Januari 2016** hingga **25 Mei 2025**, serta : 
1. **Date**: Tanggal pencatatan data harga (format: YYYY-MM-DD).
2. **Close**: Harga penutupan Bitcoin pada tanggal tersebut. Ini adalah variabel target yang akan diprediksi oleh model.
3. **High**: Harga tertinggi Bitcoin yang dicapai pada tanggal tersebut.
4. **Low**: Harga terendah Bitcoin yang dicapai pada tanggal tersebut.
5. **Open**: Harga pembukaan Bitcoin pada tanggal tersebut.
6. **Volume**: Total volume perdagangan Bitcoin pada tanggal tersebut, menunjukkan jumlah unit Bitcoin yang diperdagangkan.

Contoh Pengambilan Data: 

    import yfinance as yf
    df = yf.download("BTC-USD", start="2016-01-01", end="2025-05-26")
    
### Exploratory Data Analysis

-   Visualisasi tren harga penutupan 
    
-   Visualisasi moving average (MA7, MA30) 
    
-   Log Return Harian Harga BTC
    
-   Korelasi Antar Fitur
----------

## Data Preparation

Pada bagian ini, serangkaian teknik persiapan data diterapkan untuk mentransformasi data mentah menjadi format yang sesuai dan optimal untuk pelatihan model _deep learning_ (LSTM dan GRU). Urutan tahapan ini sangat penting untuk memastikan integritas dan kualitas data.

### 1. Penambahan Fitur Derivatif (Feature Engineering)

Langkah pertama adalah membuat fitur-fitur baru yang dapat memberikan informasi tambahan mengenai dinamika harga Bitcoin. Fitur-fitur ini meliputi:

-   **Moving Average 7 hari (MA_7):** Rata-rata harga penutupan selama 7 hari terakhir.
-   **Moving Average 30 hari (MA_30):** Rata-rata harga penutupan selama 30 hari terakhir.
-   **Log Return Harian (Log_Return):** Logaritma natural dari rasio harga penutupan saat ini dengan harga penutupan hari sebelumnya.

**Alasan:** Penambahan _Moving Averages_ berfungsi sebagai indikator _lagging_ yang menghaluskan fluktuasi harga jangka pendek dan membantu mengidentifikasi tren harga yang mendasari. MA jangka pendek lebih responsif, sementara MA jangka panjang memberikan gambaran tren yang lebih stabil. Sementara itu, _Log Return_ digunakan untuk menormalisasi distribusi _return_ dan membuat data lebih stasioner, yang penting untuk model _time series_. Fitur-fitur ini secara kolektif membantu model menangkap pola _time series_ yang lebih kompleks.

### 2. Pemilihan Fitur dan Penanganan Missing Values

Setelah fitur-fitur baru ditambahkan, subset fitur yang relevan untuk pemodelan dipilih, yaitu `Open`, `High`, `Low`, `Close`, `Volume`, `MA_7`, `MA_30`, dan `Log_Return`. Selanjutnya, baris yang mengandung nilai `NaN` (Not a Number), yang muncul akibat perhitungan _moving average_ dan _log return_ pada baris-baris awal data, dihilangkan. Selain itu, kolom 'Date' disimpan secara terpisah sebelum proses normalisasi untuk keperluan visualisasi di kemudian hari.

**Alasan:** Memfokuskan model pada fitur-fitur yang paling informatif dapat meningkatkan akurasi prediksi dan mengurangi _noise_. Penanganan _missing values_ penting karena model _deep learning_ tidak dapat memproses nilai `NaN`, sehingga penghapusan baris yang relevan memastikan semua _input_ model valid dan mencegah _error_ selama pelatihan.

### 3. Normalisasi Data

Data numerik dari fitur-fitur yang telah dipilih dinormalisasi menggunakan **MinMaxScaler**. Teknik ini mengubah nilai-nilai fitur ke dalam rentang [0, 1].

**Alasan:** Fitur-fitur seperti harga dan volume memiliki skala yang sangat berbeda. Model _deep learning_, terutama yang menggunakan _gradient-based optimization_, sangat sensitif terhadap skala data. Normalisasi mencegah fitur dengan nilai yang lebih besar mendominasi proses pembelajaran dan membantu _optimizer_ menemukan bobot optimal dengan lebih efisien, sehingga mempercepat konvergensi model.

### 4. Pembentukan Urutan Waktu (Sequence Creation)

Data yang telah dinormalisasi diubah menjadi format urutan (_sequences_) yang sesuai untuk model RNN (LSTM dan GRU). Ini dilakukan dengan menetapkan _window size_ 60, yang berarti untuk memprediksi harga penutupan pada suatu hari, model akan melihat data dari 60 hari sebelumnya (dari hari `t-60` hingga `t-1`) untuk semua fitur yang digunakan. _Output_ yang menjadi target prediksi adalah harga penutupan pada hari `t`.

**Alasan:** Model RNN dirancang khusus untuk memproses data sekuensial dan menangkap dependensi temporal. Membentuk data menjadi _sequences_ memungkinkan model untuk "mempelajari" pola dan hubungan antar data sepanjang waktu, yang esensial untuk prediksi deret waktu. Selain itu, arsitektur RNN membutuhkan _input_ dalam bentuk 3D: `(sampel, time_step, fitur)`, dan pembentukan _sequence_ ini memenuhi persyaratan tersebut.

### 5. Pembagian Data Latih dan Uji

Data yang telah di-sequencing dibagi menjadi _training set_ dan _testing set_ secara kronologis. Sekitar **80% data awal digunakan untuk pelatihan**, dan **20% sisanya digunakan untuk pengujian**. Tanggal yang sesuai dengan data pengujian juga disimpan untuk keperluan visualisasi hasil prediksi.

**Alasan:** Pembagian ini krusial untuk mengevaluasi kemampuan model dalam menggeneralisasi pada data yang belum pernah dilihat sebelumnya, memberikan estimasi yang lebih realistis tentang kinerja model di dunia nyata. Pembagian _time series_ harus dilakukan secara kronologis (data yang lebih awal untuk pelatihan, data yang lebih baru untuk pengujian) untuk menghindari _data leakage_, di mana informasi dari masa depan bocor ke dalam _training set_, menyebabkan evaluasi kinerja yang terlalu optimis.

----------
## Modeling

Tahapan pemodelan ini membahas implementasi dua arsitektur _deep learning_, yaitu **Long Short-Term Memory (LSTM)** dan **Gated Recurrent Unit (GRU)**, untuk memprediksi harga penutupan Bitcoin. Kedua model ini dipilih karena efektivitasnya dalam menangani data _time series_ dan dependensi temporal.

### 1. Arsitektur dan Implementasi Model LSTM

Model LSTM dirancang untuk menangani masalah _vanishing_ atau _exploding gradient_ yang sering terjadi pada _Recurrent Neural Network_ (RNN) tradisional, sehingga sangat cocok untuk mempelajari pola jangka panjang dalam data deret waktu.

**Kelebihan LSTM:**

-   Mampu menangkap dependensi jangka panjang dalam _sequence_ data.
-   Efektif dalam mencegah masalah _vanishing/exploding gradient_ melalui mekanisme "gerbang" (_gates_) internalnya (_input_, _forget_, _output gates_).
-   Sangat baik untuk tugas-tugas prediksi deret waktu yang kompleks seperti harga finansial.

**Kekurangan LSTM:**

-   Memiliki kompleksitas komputasi yang tinggi dan membutuhkan lebih banyak parameter, sehingga waktu pelatihan bisa lebih lama.
-   Struktur internalnya (gerbang) membuatnya lebih sulit untuk diinterpretasikan dibandingkan model yang lebih sederhana.
-   Cenderung membutuhkan dataset yang cukup besar agar dapat dilatih secara optimal.

**Tahapan Pemodelan LSTM:**

1.  **Inisialisasi Model:** Model dibangun menggunakan Keras `Sequential` API.
2.  **Lapisan LSTM (Pertama):** Menambahkan lapisan `LSTM` dengan 64 unit. `return_sequences=True` diset karena akan ada lapisan LSTM lain di bawahnya. `input_shape` disesuaikan dengan bentuk data pelatihan (`X_train.shape[1]` untuk _time step_ dan `X_train.shape[2]` untuk jumlah fitur).
3.  **Lapisan Dropout (Pertama):** Menambahkan `Dropout` dengan _rate_ 0.2 setelah lapisan LSTM pertama untuk mengurangi _overfitting_ dengan secara acak menonaktifkan 20% neuron selama pelatihan.
4.  **Lapisan LSTM (Kedua):** Menambahkan lapisan `LSTM` kedua dengan 64 unit. `return_sequences=False` karena ini adalah lapisan LSTM terakhir sebelum _output_.
5.  **Lapisan Dropout (Kedua):** Menambahkan `Dropout` dengan _rate_ 0.2 untuk _regularization_ lebih lanjut.
6.  **Lapisan Output (Dense):** Lapisan `Dense` dengan 1 neuron digunakan sebagai lapisan _output_ untuk menghasilkan prediksi harga penutupan tunggal.
7.  **Kompilasi Model:** Model dikompilasi dengan _optimizer_ 'adam' dan _loss function_ 'mean_squared_error' (MSE), yang merupakan metrik umum untuk masalah regresi.

**Parameter Pelatihan LSTM:**

-   **Epochs:** 50 (namun dihentikan lebih awal oleh _EarlyStopping_).
-   **Batch Size:** 32.
-   **Optimizer:** Adam.
-   **Loss Function:** Mean Squared Error (MSE).
-   **Callbacks:** `EarlyStopping` diimplementasikan dengan `monitor='val_loss'` dan `patience=10`. Ini berarti pelatihan akan berhenti jika _validation loss_ tidak membaik selama 10 _epoch_ berturut-utut, dan bobot model terbaik akan dikembalikan.

**Proses Prediksi dan _Inverse Transform_:** Setelah pelatihan, model digunakan untuk membuat prediksi pada data uji (`X_test`). Hasil prediksi dan nilai aktual kemudian di-_inverse transform_ dari skala [0, 1] kembali ke skala harga Bitcoin asli menggunakan `MinMaxScaler` yang sama yang digunakan saat normalisasi. Ini penting agar hasil prediksi dapat diinterpretasikan dalam unit harga yang sebenarnya.

### 2. Arsitektur dan Implementasi Model GRU

Gated Recurrent Unit (GRU) adalah varian lain dari RNN yang juga dirancang untuk mengatasi masalah _vanishing gradient_. GRU memiliki arsitektur yang lebih sederhana dibandingkan LSTM karena hanya memiliki dua gerbang (_update gate_ dan _reset gate_), sehingga lebih ringan secara komputasi.

**Kelebihan GRU:**

-   **Komputasi Lebih Cepat:** Memiliki lebih sedikit parameter dibandingkan LSTM, sehingga lebih cepat dilatih dan membutuhkan lebih sedikit sumber daya komputasi.
-   **Efektif untuk Dependensi Jangka Panjang:** Tetap mampu menangkap dependensi jangka panjang dalam data deret waktu meskipun lebih sederhana dari LSTM.
-   **Mudah Diimplementasikan:** Relatif lebih mudah dipahami dan diimplementasikan karena arsitekturnya yang ringkas.

**Kekurangan GRU:**

-   Meskipun sering berkinerja setara dengan LSTM, dalam beberapa kasus _time series_ yang sangat kompleks atau memiliki dependensi yang sangat panjang, GRU mungkin sedikit kalah dalam akurasi dari LSTM.
-   Sama seperti LSTM, GRU masih membutuhkan data yang cukup untuk pelatihan yang efektif.

**Tahapan Pemodelan GRU:**

1.  **Inisialisasi Model:** Model dibangun menggunakan Keras `Sequential` API.
2.  **Lapisan GRU (Pertama):** Menambahkan lapisan `GRU` dengan 64 unit. `return_sequences=True` diset. `input_shape` disesuaikan.
3.  **Lapisan Dropout (Pertama):** Menambahkan `Dropout` dengan _rate_ 0.2.
4.  **Lapisan GRU (Kedua):** Menambahkan lapisan `GRU` kedua dengan 64 unit. `return_sequences=False`.
5.  **Lapisan Dropout (Kedua):** Menambahkan `Dropout` dengan _rate_ 0.2.
6.  **Lapisan Output (Dense):** Lapisan `Dense` dengan 1 neuron untuk menghasilkan prediksi harga penutupan tunggal.
7.  **Kompilasi Model:** Model dikompilasi dengan _optimizer_ 'adam' dan _loss function_ 'mean_squared_error' (MSE).

**Parameter Pelatihan GRU:**

-   **Epochs:** 50 (namun dihentikan lebih awal oleh _EarlyStopping_).
-   **Batch Size:** 32.
-   **Optimizer:** Adam.
-   **Loss Function:** Mean Squared Error (MSE).
-   **Callbacks:** `EarlyStopping` diimplementasikan dengan `monitor='val_loss'` dan `patience=10`.

**Proses Prediksi dan _Inverse Transform_:** Sama seperti model LSTM, model GRU digunakan untuk memprediksi pada data uji, dan hasil prediksi beserta nilai aktual kemudian di-_inverse transform_ kembali ke skala harga Bitcoin asli.

### 3. Pemilihan Model Terbaik

Setelah melatih kedua model (LSTM dan GRU) dengan arsitektur dan parameter yang sebanding, pemilihan model terbaik didasarkan pada perbandingan metrik evaluasi pada _testing set_.

**Alasan Memilih Model Terbaik:**

-   **Semua metrik menunjukkan bahwa GRU menghasilkan _error_ yang lebih kecil dibandingkan LSTM.** Ini berarti GRU memiliki akurasi prediksi yang lebih tinggi dalam mengukur jarak antara nilai prediksi dan nilai aktual.
-   **GRU memiliki R2 yang lebih tinggi, artinya model GRU menjelaskan variansi data target lebih baik.** Nilai R2 yang lebih tinggi mengindikasikan bahwa model GRU lebih mampu menangkap dan menjelaskan fluktuasi harga Bitcoin dibandingkan LSTM.

Berdasarkan perbandingan metrik evaluasi secara menyeluruh (MSE, RMSE, MAE, dan R²), **model GRU akan dipilih sebagai solusi terbaik** untuk memprediksi harga Bitcoin. Kinerja GRU yang superior dalam hal _error_ yang lebih rendah dan kemampuan menjelaskan variansi data yang lebih tinggi menjadikannya pilihan yang lebih unggul untuk proyek ini. Selain itu, aspek interpretasi grafik _Loss_ selama pelatihan juga dipertimbangkan untuk melihat stabilitas dan konvergensi model.

----------

## Evaluation

Pada tahap evaluasi ini, kinerja model yang telah dilatih diukur menggunakan metrik-metrik yang relevan untuk masalah regresi. Metrik yang digunakan adalah **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, dan **R-squared (R²)**. Evaluasi dilakukan pada data *testing set* yang belum pernah dilihat oleh model selama pelatihan, untuk memberikan gambaran yang objektif tentang kemampuan generalisasi model.

### 1. Penjelasan Metrik Evaluasi

#### a. Mean Squared Error (MSE)

- **Penjelasan:** MSE mengukur rata-rata dari kuadrat perbedaan antara nilai prediksi ($\hat{y_i}$) dan nilai aktual ($y_i$). Metrik ini memberikan bobot yang lebih besar pada kesalahan prediksi yang besar karena adanya pengkuadratan, sehingga sensitif terhadap *outlier*.
- **Formula:**  
MSE = (1/n) × Σ(yᵢ − ŷᵢ)²
- **Cara Kerja:** Semakin kecil nilai MSE, semakin baik model dalam memprediksi nilai aktual, karena menunjukkan rata-rata kesalahan kuadrat yang lebih rendah.

#### b. Root Mean Squared Error (RMSE)

- **Penjelasan:** RMSE adalah akar kuadrat dari MSE. Metrik ini sangat populer karena menghasilkan unit yang sama dengan variabel target (harga Bitcoin), sehingga lebih mudah diinterpretasikan dibandingkan MSE. Seperti MSE, RMSE juga sensitif terhadap *outlier*.
- **Formula:**  
RMSE = √[ (1/n) × Σ(yᵢ − ŷᵢ)² ]
- **Cara Kerja:** Semakin kecil nilai RMSE, semakin akurat model dalam memprediksi harga Bitcoin.

#### c. Mean Absolute Error (MAE)

- **Penjelasan:** MAE mengukur rata-rata dari nilai absolut perbedaan antara nilai prediksi ($\hat{y_i}$) dan nilai aktual ($y_i$). Berbeda dengan MSE dan RMSE, MAE tidak mengkuadratkan kesalahan, sehingga kurang sensitif terhadap *outlier*.
- **Formula:**  
MAE = (1/n) × Σ|yᵢ − ŷᵢ|
- **Cara Kerja:** Semakin kecil nilai MAE, semakin baik model. MAE memberikan gambaran langsung tentang rata-rata besar kesalahan prediksi dalam unit target.

#### d. R-squared (R²)

- **Penjelasan:** R², atau koefisien determinasi, mengukur proporsi varians dalam variabel dependen (harga penutupan Bitcoin) yang dapat dijelaskan oleh model regresi. Nilainya berkisar dari 0 hingga 1 (atau bisa negatif jika model sangat buruk).
- **Formula:**  
<div align="center">

R² = 1 − [ Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² ]

</div>


Di mana ȳ adalah rata-rata dari nilai aktual (yᵢ).
- **Cara Kerja:** Nilai R² yang mendekati 1 menunjukkan bahwa model sangat baik dalam menjelaskan variasi data target.

---

### 2. Hasil Proyek Berdasarkan Metrik Evaluasi

Berikut adalah hasil evaluasi kedua model (LSTM dan GRU) pada _testing set_ setelah _inverse transform_ ke skala harga Bitcoin asli:

**a. Model LSTM:**

```
Evaluasi Model pada Data Tes:
Mean Squared Error (MSE): 6061932.40
Root Mean Squared Error (RMSE): 2462.10
Mean Absolute Error (MAE): 1774.98
R-squared (R2): 0.9893

```
**b. Model GRU:**

```
Evaluasi Model pada Data Tes:
Mean Squared Error (MSE): 3697483.18
Root Mean Squared Error (RMSE): 1922.88
Mean Absolute Error (MAE): 1321.34
R-squared (R2): 0.9935
```
### 3. Interpretasi Hasil dan Pemilihan Model Terbaik

Berdasarkan perbandingan metrik evaluasi dari kedua model:

-   **Kesalahan Prediksi:** Model GRU menunjukkan nilai MSE (3,697,483.18), RMSE (1,922.88), dan MAE (1,321.34) yang secara signifikan lebih rendah dibandingkan model LSTM (MSE: 6,061,932.40, RMSE: 2,462.10, MAE: 1,774.98). Ini berarti bahwa, secara rata-rata, prediksi yang dihasilkan oleh model GRU memiliki deviasi yang lebih kecil dari harga Bitcoin aktual. Dengan kata lain, model GRU menghasilkan _error_ prediksi yang lebih kecil.
    
-   **Kemampuan Menjelaskan Variansi Data (R2):** Model GRU mencapai nilai R2 sebesar 0.9935, yang lebih tinggi daripada model LSTM yang memiliki R2 sebesar 0.9893. Nilai R2 yang lebih tinggi pada GRU mengindikasikan bahwa model ini mampu menjelaskan sekitar 99.35% dari variasi harga Bitcoin, menunjukkan kecocokan yang sangat baik dengan data aktual. Hal ini menegaskan bahwa model GRU lebih baik dalam menangkap pola dan tren yang mendasari pergerakan harga Bitcoin.
    

**Kesimpulan:** Semua metrik evaluasi menunjukkan bahwa **model GRU berkinerja lebih unggul dibandingkan model LSTM** dalam memprediksi harga Bitcoin pada dataset ini. GRU menghasilkan _error_ yang lebih kecil di semua metrik berbasis kesalahan (MSE, RMSE, MAE) dan memiliki kemampuan menjelaskan variansi data yang lebih baik (R² lebih tinggi). Oleh karena itu, **Model GRU akan dipilih sebagai model terbaik** untuk memprediksi harga Bitcoin dalam proyek ini. Kemampuan GRU untuk memberikan prediksi yang lebih akurat dengan _error_ yang lebih rendah menjadikannya solusi yang lebih andal untuk membantu investor dan _trader_.
