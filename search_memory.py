import argparse

try:
    import mysql.connector as mysql_connector
except ImportError:
    mysql_connector = None

from cli_utils import get_mysql_connection_config, trim_snippet


def get_db_connection():
    """Membuka koneksi MySQL untuk pencarian memori lokal."""
    if mysql_connector is None:
        raise RuntimeError("mysql-connector-python belum terpasang. Jalankan: pip install mysql-connector-python")
    return mysql_connector.connect(**get_mysql_connection_config())


def search_memory(query: str, limit: int = 5, snippet_chars: int = 300):
    """Mencari riwayat chat berdasarkan kata kunci dengan output ringkas."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as exc:
        print(f"Error: Tidak bisa terhubung ke MySQL lokal: {exc}")
        print("Pastikan layanan database lokal Anda aktif dan konfigurasi project benar.")
        return

    print(f"=== Mencari Memori Agent untuk Kata Kunci: '{query}' ===\n")

    sql = """
        SELECT m.role, m.content, m.created_at, c.title
        FROM messages m
        JOIN chats c ON m.chat_id = c.id
        WHERE m.content LIKE %s
        ORDER BY m.created_at DESC
        LIMIT %s
    """
    like_query = f"%{query}%"

    try:
        cursor.execute(sql, (like_query, max(1, limit)))
        rows = cursor.fetchall()

        if not rows:
            print("Tidak ditemukan memori atau riwayat chat yang cocok.")
            return

        for index, row in enumerate(rows, 1):
            role, content, created_at, chat_title = row
            role_display = "Developer" if role == "user" else "AgentAI"
            snippet = trim_snippet(content, max(80, snippet_chars))

            print(f"[{index}] Sesi Chat: '{chat_title}' ({created_at})")
            print(f"    Peran: {role_display}")
            print(f"    Isi Memori:\n    {snippet}\n")
            print("-" * 60)
    except Exception as exc:
        print(f"Gagal mengeksekusi pencarian: {exc}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


def build_parser() -> argparse.ArgumentParser:
    """Membangun parser CLI untuk pencarian memori lokal."""
    parser = argparse.ArgumentParser(description="Cari memori percakapan AgentAI dari database lokal")
    parser.add_argument("query", help="Kata kunci pencarian")
    parser.add_argument("--limit", type=int, default=5, help="Jumlah hasil yang ditampilkan")
    parser.add_argument("--snippet", type=int, default=300, help="Panjang snippet isi memori")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    search_memory(args.query, limit=args.limit, snippet_chars=args.snippet)
