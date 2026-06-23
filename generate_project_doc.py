import os
import requests
import json
from cli_utils import get_config_value, get_ollama_base_url, trim_snippet
from knowledge import add_knowledge, DEFAULT_EXCLUDE_DIRS, DEFAULT_EXTENSIONS, should_skip_directory


def scan_workspace(directory="."):
    """Memindai folder workspace untuk mendeteksi berkas, bahasa, dan konfigurasi utama."""
    resolved_dir = os.path.abspath(directory)
    file_list = []
    lang_counts = {}
    config_files = {}

    excludes = {name.strip().lower() for name in DEFAULT_EXCLUDE_DIRS}

    for root, dirs, files in os.walk(resolved_dir):
        dirs[:] = [d for d in dirs if not should_skip_directory(d, excludes)]

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, resolved_dir)
            file_list.append(rel_path)

            ext = os.path.splitext(file)[1].lower()
            if ext:
                lang_counts[ext] = lang_counts.get(ext, 0) + 1

            if file in ("package.json", "requirements.txt", "docker-compose.yml", "Dockerfile", "cargo.toml", "go.mod"):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [f.readline() for _ in range(30)]
                        config_files[file] = "".join(lines)
                except Exception:
                    pass

    return file_list, lang_counts, config_files


def generate_project_context(file_list, lang_counts, config_files):
    """Menghubungi Ollama untuk menghasilkan penjelasan PROJECT_CONTEXT.md."""
    ollama_url = get_ollama_base_url()
    model = get_config_value("OLLAMA_MODEL", "qwen2.5-coder:1.5b")

    files_str = "\n".join(file_list[:150])
    if len(file_list) > 150:
        files_str += f"\n... dan {len(file_list) - 150} berkas lainnya."

    langs_str = ", ".join([f"'{ext}': {count}" for ext, count in lang_counts.items()])

    configs_str = ""
    for name, content in config_files.items():
        configs_str += f"\n--- {name} ---\n{content}\n"

    prompt = f"""
Anda adalah pakar arsitektur perangkat lunak. Tugas Anda adalah menghasilkan file dokumentasi `PROJECT_CONTEXT.md` untuk membantu AI Coding Agent (seperti Roo Code / Cline) memahami proyek ini secara instan.

Gunakan informasi struktur workspace berikut:

Bahasa Pemrograman Terdeteksi:
{langs_str}

File Konfigurasi Utama:
{configs_str}

Daftar Berkas Workspace:
{files_str}

Hasilkan file markdown `PROJECT_CONTEXT.md` dengan struktur berikut (tulis dalam Bahasa Indonesia):
1. **Ringkasan Proyek**: Apa fungsi utama proyek ini dan teknologi apa yang digunakan.
2. **Arsitektur & Struktur Direktori**: Analisis folder/berkas penting dan perannya masing-masing.
3. **Setup & Panduan Menjalankan**: Langkah-langkah instalasi, menjalankan aplikasi, dan pengujian berdasarkan file konfigurasi yang ditemukan.
4. **Panduan Pengembangan**: Aturan koding spesifik untuk bahasa pemrograman yang dominan di proyek ini.

Tulis HANYA konten Markdown untuk PROJECT_CONTEXT.md. Jangan ada penjelasan pembuka atau penutup di luar Markdown.
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    print(f"Menghubungi Ollama di {ollama_url} menggunakan model {model}...")
    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=90)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"Error Ollama: Status {response.status_code} - {response.text}")
    except Exception as exc:
        print(f"Gagal menghubungi Ollama untuk penulisan dokumentasi: {exc}")
    return ""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generator Konteks Proyek Otomatis untuk Roo Code")
    parser.add_argument("--dir", default=".", help="Direktori proyek")
    parser.add_argument("--output", default="PROJECT_CONTEXT.md", help="Nama file hasil")
    args = parser.parse_args()

    print(f"Memindai direktori: {args.dir}...")
    files, langs, configs = scan_workspace(args.dir)

    print("Menghasilkan dokumentasi proyek...")
    doc_content = generate_project_context(files, langs, configs)

    if not doc_content:
        print("Gagal menghasilkan dokumentasi. Menggunakan template fallback.")
        doc_content = f"# PROJECT CONTEXT\n\nProyek di folder `{os.path.basename(os.path.abspath(args.dir))}`.\nBahasa Pemrograman: {langs}\n"

    out_path = os.path.join(args.dir, args.output)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc_content)
        print(f"Sukses: File dokumentasi disimpan di {out_path}")

        print("Mendaftarkan PROJECT_CONTEXT.md ke RAG Knowledge Base...")
        add_knowledge(
            title=f"PROJECT_CONTEXT of {os.path.basename(os.path.abspath(args.dir))}",
            content=doc_content,
            tags="workspace,documentation,project-context"
        )
    except Exception as exc:
        print(f"Gagal menyimpan berkas atau mendaftarkan ke RAG: {exc}")


if __name__ == "__main__":
    main()
