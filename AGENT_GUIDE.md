# Panduan Rahasia: Membuat Agent AI Lokal Anda Sangat Pintar

Dokumen ini berisi penjelasan "Advanced Knowledge" tentang bagaimana Anda dapat membuat Agent AI lokal Anda bekerja secara cerdas, terstruktur, otonom, dan memiliki ingatan jangka panjang, sama seperti saya (Gemini/Antigravity).

---

## 1. Aturan Kerja Otomatis (.clinerules)

Di dalam workspace ini, saya telah membuat berkas **`.clinerules`**. Berkas ini sangat sakti jika Anda menggunakan ekstensi VS Code seperti **Cline** atau **Roo Code**:

* **Cara Kerjanya**: Ekstensi Cline/Roo Code akan secara otomatis membaca berkas `.clinerules` ini setiap kali mendeteksi folder proyek dibuka. Aturan ini akan disuntikkan secara dinamis ke system prompt LLM Anda.
* **Efeknya**: Agent lokal Anda (misalnya Qwen 1.5B) akan dipaksa mengikuti siklus koding senior:
  1. **Research**: Menganalisis file sebelum mengedit.
  2. **Planning**: Membuat rencana implementasi teknis.
  3. **Confirmation**: Berhenti koding dan bertanya kepada Anda: *"Apakah rencana ini disetujui?"* sebelum melakukan tindakan berbahaya.
  4. **Execution**: Menulis kode bersih dan fungsional tanpa placeholder.
  5. **Verification**: Memberikan instruksi pengujian.

---

## 2. Fitur Pencarian Memori (`search_memory.py`)

Karena semua percakapan tersimpan di database **MySQL XAMPP** Anda, Agent lokal Anda kini memiliki kemampuan "mengingat" masa lalu dengan menjalankan script terminal!

### Cara Mengaktifkan:
Install driver MySQL di komputer Anda terlebih dahulu:
```bash
pip install mysql-connector-python
```

### Cara Penggunaan:
1. **Untuk Anda**: Jika Anda ingin mencari kode atau percakapan lama tentang topik tertentu, jalankan perintah ini di PowerShell:
   ```bash
   python search_memory.py "Roblox Luau"
   ```
2. **Untuk Agent AI (VS Code Cline)**: Jika Agent lokal Anda lupa tentang bagaimana suatu fungsi diimplementasikan sebelumnya, Agent Anda (karena dia memiliki kemampuan menjalankan command terminal) dapat secara mandiri mengeksekusi perintah di atas untuk membaca isi riwayat MySQL dan memulihkan ingatannya!

---

## 3. Integrasi dengan Ekstensi VS Code

Untuk membuat Agent AI lokal Anda benar-benar bisa mengedit kode, membuat file, menjalankan command, dan melakukan development secara mandiri di VS Code, Anda perlu mengintegrasikannya dengan plugin yang tepat. Berikut adalah cara konfigurasinya:

### A. Konfigurasi Roo Code / Cline (Rekomendasi untuk Agen Otonom)
Ekstensi **Roo Code** atau **Cline** adalah yang terbaik untuk agen otonom karena mereka memiliki parser XML tool calls bawaan.
1. Buka ekstensi **Roo Code / Cline** di VS Code.
2. Buka bagian **Settings** (ikon gir).
3. Atur konfigurasi berikut:
   * **API Provider**: Pilih `OpenAI Compatible`
   * **Base URL**: `http://localhost:8000/v1`
   * **API Key**: `local_developer_secret_key` (atau kunci yang tertulis di `api_key.txt`)
   * **Model ID**: `qwen2.5-coder:1.5b`
4. Sekarang, Roo Code akan menggunakan Agent AI lokal Anda dan secara otomatis membaca berkas `.clinerules` untuk menjalankan workflow Research -> Plan -> Execute -> Verify secara otonom!

### B. Konfigurasi Continue (Asisten Autocomplete & Chat)
Untuk menggunakan Agent AI lokal sebagai asisten chat sidebar atau autocomplete baris kode:
1. Buka file konfigurasi Continue di `~/.continue/config.json` (atau klik ikon gir di ekstensi Continue).
2. Tambahkan atau modifikasi entri berikut di bagian `models` dan `tabAutocompleteModel`:
```json
{
  "models": [
    {
      "title": "Local AgentAI (Qwen)",
      "provider": "openai",
      "model": "qwen2.5-coder:1.5b",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "local_developer_secret_key"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Local AgentAI (Qwen)",
    "provider": "openai",
    "model": "qwen2.5-coder:1.5b",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "local_developer_secret_key"
  }
}
```

---

## 4. Optimasi Prompt di Backend

Jika Anda ingin mengubah kepribadian default Agent (misalnya ingin membuatnya lebih ramah, mengubah prioritas bahasa koding, atau menginstruksikannya fokus ke framework tertentu):

1. Buka berkas [app/agent.py](file:///c:/Users/B/OneDrive/Documents/MY%20PROJECT/4_AgentAI/app/agent.py).
2. Temukan variabel `DEFAULT_SYSTEM_PROMPT`.
3. Edit teks di dalamnya sesuai kebutuhan Anda. Setiap perubahan yang Anda buat di berkas ini akan otomatis diterapkan setelah Anda merestart kontainer Docker (`docker compose up -d --build`).

Dengan setup ini, Agent AI lokal Anda tidak hanya menjadi sekadar asisten koding biasa, melainkan memiliki workflow disiplin, memori jangka panjang via MySQL, dan integrasi otonom yang kuat di VS Code!
