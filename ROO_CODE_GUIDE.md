# Panduan Integrasi: Menghubungkan Agent AI Gateway dengan Roo Code

Dokumen ini menjelaskan cara menghubungkan **Agent AI Gateway** Anda dengan **Roo Code** (ekstensi VS Code berbasis agen kustom) sebagai penyedia model kustom mandiri yang aman, cerdas, dan hemat memori.

---

## 1. Persiapan Gateway

Pastikan Agent AI Gateway telah berjalan secara lokal (baik melalui Docker container maupun langsung menggunakan python).
*   **Base URL**: `http://localhost:8000/v1`
*   **API Key**: Dapatkan token Authorization yang valid dari berkas `api_key.txt` di root proyek Anda (misal: `local_developer_secret_key`).

---

## 2. Langkah Konfigurasi di Roo Code

Buka VS Code dan ikuti langkah-langkah berikut untuk melakukan konfigurasi:

1.  **Buka Panel Roo Code**: Klik ikon Roo Code di VS Code Activity Bar (menu samping kiri).
2.  **Buka Halaman Pengaturan (Settings)**: Klik ikon roda gigi (⚙️) di bagian atas panel Roo Code.
3.  **Pilih API Provider**: Pada dropdown **API Provider**, pilih **"OpenAI Compatible"**.
4.  **Isi Detail Endpoint**:
    *   **Base URL**: Masukkan `http://localhost:8000/v1`
    *   **API Key**: Tempelkan kunci API yang Anda salin dari berkas `api_key.txt` (atau environment variabel `AGENT_API_KEY` Anda).
    *   **Model ID**: Masukkan nama model yang terpasang di Ollama Anda (contoh: `qwen2.5-coder:7b`, `llama3.1`, dll.). Anda juga bisa mengklik dropdown untuk memuat model yang tersedia dari gateway.
5.  **Simpan Konfigurasi**: Klik tombol **"Let's go!"** atau **"Save"** di bagian bawah panel pengaturan.

---

## 3. Fitur Keunggulan Integrasi untuk Roo Code

Saat Roo Code mengirimkan permintaan ke gateway ini, middleware kami secara transparan melakukan beberapa optimasi tingkat lanjut:

### 🧠 Otak Eksternal (MySQL RAG & Memori Panjang)
Setiap percakapan, instruksi sintaks bahasa pemrograman, dan solusi perbaikan bug akan otomatis disimpan ke database MySQL Anda. Gateway menyuntikkan dokumen RAG semantik yang relevan ke dalam prompt sebelum dikirim ke model lokal Anda agar tanggapan lebih akurat.

### ⏱️ Pembelajaran Mandiri Saat Idle (Idle Self-Learning)
Jika sistem dalam keadaan idle selama 5 menit tanpa ada permintaan dari Roo Code, agen akan memasuki mode belajar mandiri secara otomatis untuk mendalami bahasa baru. Jika ada permintaan baru dari Roo Code, proses pembelajaran mandiri akan dihentikan sementara (hold) demi merespons tugas Anda terlebih dahulu.

### 🛡️ Terminal Approval Gate & Proteksi Bahaya
Perintah bash atau powershell yang dikirim oleh Roo Code akan dipindai oleh supervisor keselamatan kami. Perintah berbahaya (seperti penghapusan rekursif dari root, chmod 777, atau pipe command curl ke bash unverified) akan ditolak secara otomatis untuk mengamankan komputer Anda.

### 📊 Akurasi Statistik Token (`tiktoken`)
Statistik token prompt dan output dikalkulasi secara presisi menggunakan parser token `tiktoken`. Ini memastikan diagram konsumsi token dan batas jendela konteks (context window) yang ditampilkan di antarmuka Roo Code 100% akurat.

---

## 4. Troubleshooting (Pemecahan Masalah)

*   **Roo Code Gagal Terhubung**: Pastikan kontainer docker gateway berjalan (`docker compose ps`) dan Anda bisa mengakses `http://localhost:8000/health` melalui browser.
*   **Masalah Tool Calling**: Jika model tidak merespons dengan tool call (Roo Code tidak melakukan penulisan berkas atau terminal command), pastikan model Ollama yang Anda pilih mendukung native tool calling (kami sangat menyarankan keluarga model `qwen2.5-coder` untuk tugas coding lokal).
