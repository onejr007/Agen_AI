import argparse
import json
import os

import requests
try:
    import mysql.connector as mysql_connector
except ImportError:
    mysql_connector = None

from cli_utils import (
    build_gateway_headers,
    chunk_text,
    get_gateway_base_url,
    get_mysql_connection_config,
    get_ollama_base_url,
    get_ollama_embed_model,
    normalize_csv_list,
    trim_snippet,
)

DEFAULT_EXTENSIONS = [".py", ".lua", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt", ".go", ".rs"]
DEFAULT_EXCLUDE_DIRS = [
    "node_modules", ".git", "__pycache__", ".venv", "env", "dist", "build",
    ".agents", ".gemini", ".system_generated", "ollama_data"
]


def get_db_connection():
    """Membuka koneksi MySQL lokal berdasarkan konfigurasi project."""
    if mysql_connector is None:
        raise RuntimeError("mysql-connector-python belum terpasang. Jalankan: pip install mysql-connector-python")
    return mysql_connector.connect(**get_mysql_connection_config())


def get_embedding(text: str) -> list:
    """Mengambil embedding dari Ollama lokal dengan konfigurasi host-side."""
    ollama_url = get_ollama_base_url()
    payload = {
        "model": get_ollama_embed_model(),
        "prompt": text,
    }
    try:
        response = requests.post(f"{ollama_url}/api/embeddings", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("embedding", [])
        print(f"Error Ollama: Status {response.status_code} - {trim_snippet(response.text, 200)}")
    except Exception as exc:
        print(f"Error: Tidak bisa terhubung ke Ollama di {ollama_url} ({exc})")
    return []


def add_knowledge(title: str, content: str, tags: str):
    """Menyimpan pengetahuan baru ke knowledge base dengan embedding lokal."""
    print("Menghasilkan vektor embedding untuk konten...")
    vector = get_embedding(content)
    if not vector:
        print("Gagal membuat embedding. Penyimpanan dibatalkan.")
        return

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO knowledge_base (title, content, tags, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (title.strip(), content.strip(), tags.strip(), json.dumps(vector)),
        )
        conn.commit()
        print(f"Sukses: Pengetahuan '{title}' berhasil disimpan dan diindeks.")
    except Exception as exc:
        print(f"Database Error: {exc}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


def list_knowledge(limit: int = 20):
    """Menampilkan daftar pengetahuan terakhir dari database."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, tags, created_at
            FROM knowledge_base
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (max(1, limit),),
        )
        rows = cursor.fetchall()

        if not rows:
            print("Belum ada pengetahuan yang disimpan di database.")
            return

        print("=== Daftar Pengetahuan (RAG Knowledge Base) ===\n")
        print(f"{'ID':<5} | {'Judul':<35} | {'Tags':<25} | {'Dibuat':<20}")
        print("-" * 92)
        for idx, title, tags, created_at in rows:
            tags_display = tags if tags else "-"
            print(f"{idx:<5} | {title[:35]:<35} | {tags_display[:25]:<25} | {str(created_at):<20}")
    except Exception as exc:
        print(f"Database Error: {exc}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


def delete_knowledge(entry_id: int):
    """Menghapus satu pengetahuan berdasarkan ID."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM knowledge_base WHERE id = %s", (entry_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Sukses: Pengetahuan dengan ID {entry_id} berhasil dihapus.")
        else:
            print(f"ID {entry_id} tidak ditemukan.")
    except Exception as exc:
        print(f"Database Error: {exc}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


def should_skip_directory(path_name: str, excluded_names: set[str]) -> bool:
    """Memeriksa apakah sebuah nama direktori harus diabaikan."""
    return path_name.strip().lower() in excluded_names


def index_workspace(
    directory: str,
    extensions: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
    chunk_size: int = 1500,
    max_chunks_per_file: int = 8,
    max_files: int | None = None,
):
    """Mengindeks file workspace ke knowledge base melalui gateway lokal."""
    resolved_dir = os.path.abspath(directory)
    if not os.path.isdir(resolved_dir):
        print(f"Folder tidak ditemukan: {resolved_dir}")
        return

    normalized_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or DEFAULT_EXTENSIONS)}
    normalized_excludes = {name.strip().lower() for name in (exclude_dirs or DEFAULT_EXCLUDE_DIRS)}
    gateway_url = f"{get_gateway_base_url().rstrip('/')}/knowledge"
    headers = build_gateway_headers()

    print(f"Memulai indeks ruang kerja di folder: {resolved_dir}")
    print(f"Ekstensi berkas yang dipindai: {sorted(normalized_extensions)}")
    print(f"Folder yang diabaikan: {sorted(normalized_excludes)}")
    print(f"Gateway target: {gateway_url}\n")

    files_indexed = 0
    chunks_created = 0
    files_with_errors = 0

    for root, dirs, files in os.walk(resolved_dir):
        dirs[:] = [d for d in dirs if not should_skip_directory(d, normalized_excludes)]

        for file_name in files:
            if max_files is not None and files_indexed >= max_files:
                print("Batas jumlah file tercapai. Proses indexing dihentikan lebih awal.")
                print(f"\nSelesai! Berhasil mengindeks {files_indexed} berkas menjadi {chunks_created} bagian. File gagal: {files_with_errors}.")
                return

            extension = os.path.splitext(file_name)[1].lower()
            if extension not in normalized_extensions:
                continue

            file_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(file_path, resolved_dir)

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
            except Exception as exc:
                files_with_errors += 1
                print(f"Gagal membaca file {rel_path}: {exc}")
                continue

            if not content.strip():
                continue

            chunks = chunk_text(content, chunk_size=chunk_size, max_chunks=max_chunks_per_file)
            if not chunks:
                continue

            files_indexed += 1
            print(f"Memproses {rel_path} ({len(chunks)} bagian)...")

            for idx, chunk in enumerate(chunks, 1):
                payload = {
                    "title": f"[Workspace] {rel_path} (Part {idx}/{len(chunks)})",
                    "content": chunk,
                    "tags": f"workspace,{extension[1:]}",
                }
                try:
                    response = requests.post(gateway_url, headers=headers, json=payload, timeout=15)
                    if response.status_code == 200:
                        chunks_created += 1
                    else:
                        files_with_errors += 1
                        print(f"  Gagal mengirim bagian {idx}: {response.status_code} - {trim_snippet(response.text, 180)}")
                except Exception as exc:
                    files_with_errors += 1
                    print(f"  Koneksi API Error pada bagian {idx}: {exc}")

    print(
        f"\nSelesai! Berhasil mengindeks {files_indexed} berkas menjadi {chunks_created} bagian. "
        f"File/operasi gagal: {files_with_errors}."
    )


def build_parser() -> argparse.ArgumentParser:
    """Membangun parser CLI untuk knowledge tool."""
    parser = argparse.ArgumentParser(description="Kelola Pengetahuan RAG (Memori Semantik) untuk AgentAI")
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")

    add_parser = subparsers.add_parser("add", help="Tambah pengetahuan baru")
    add_parser.add_argument("--title", required=True, help="Judul dokumen / API")
    add_parser.add_argument("--content", required=True, help="Isi detail dokumentasi / potongan kode")
    add_parser.add_argument("--tags", default="", help="Tag pendukung, dipisahkan koma")

    list_parser = subparsers.add_parser("list", help="Daftar pengetahuan")
    list_parser.add_argument("--limit", type=int, default=20, help="Jumlah item yang ditampilkan")

    delete_parser = subparsers.add_parser("delete", help="Hapus pengetahuan berdasarkan ID")
    delete_parser.add_argument("--id", type=int, required=True, help="ID pengetahuan yang ingin dihapus")

    index_parser = subparsers.add_parser("index-workspace", help="Indeks otomatis berkas di ruang kerja proyek")
    index_parser.add_argument("--dir", default=".", help="Direktori yang ingin diindeks")
    index_parser.add_argument("--extensions", default="", help="Daftar ekstensi, dipisahkan koma")
    index_parser.add_argument("--exclude-dirs", default="", help="Daftar folder yang diabaikan, dipisahkan koma")
    index_parser.add_argument("--chunk-size", type=int, default=1500, help="Ukuran maksimal karakter per chunk")
    index_parser.add_argument("--max-chunks-per-file", type=int, default=8, help="Batas chunk per file")
    index_parser.add_argument("--max-files", type=int, default=0, help="Batas jumlah file yang diindeks, 0 = tanpa batas")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "add":
        add_knowledge(args.title, args.content, args.tags)
    elif args.command == "list":
        list_knowledge(limit=args.limit)
    elif args.command == "delete":
        delete_knowledge(args.id)
    elif args.command == "index-workspace":
        index_workspace(
            args.dir,
            extensions=normalize_csv_list(args.extensions, DEFAULT_EXTENSIONS),
            exclude_dirs=normalize_csv_list(args.exclude_dirs, DEFAULT_EXCLUDE_DIRS),
            chunk_size=args.chunk_size,
            max_chunks_per_file=max(1, args.max_chunks_per_file),
            max_files=None if args.max_files <= 0 else args.max_files,
        )
    else:
        parser.print_help()
