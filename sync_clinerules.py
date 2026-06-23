import os
from knowledge import get_db_connection, execute_sql, close_connection, get_config_value

ext_map = {
    ".lua": "luau",
    ".luau": "luau",
    ".py": "python",
    ".php": "php",
    ".sql": "mysql",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".html": "web",
    ".css": "web",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sh": "bash",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json"
}


def analyze_workspace():
    """Memindai workspace untuk mendeteksi bahasa pemrograman yang aktif."""
    exclude_dirs = {".git", "node_modules", "venv", ".venv", "__pycache__", ".gemini", ".agents", "build", "dist"}
    lang_counts = {}

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ext_map:
                lang = ext_map[ext]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
    return lang_counts


def sync_rules():
    """Mengambil aturan koding dari database lokal dan memperbarui file .clinerules."""
    lang_counts = analyze_workspace()
    active_langs = [lang for lang, count in lang_counts.items() if count > 0]

    print(f"Bahasa pemrograman aktif di workspace: {lang_counts}")

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        execute_sql(cursor, "SELECT language_name, instructions FROM language_guidelines WHERE is_active = 1")
        rows = cursor.fetchall()

        guidelines_map = {}
        for row in rows:
            lang_name, instructions = row
            guidelines_map[lang_name.lower()] = instructions

        compiled_rules = []
        compiled_rules.append("# Roo Code (Cline) Coding Rules & Guidelines")
        compiled_rules.append("Ini adalah berkas aturan koding yang disinkronkan secara dinamis berdasarkan profil bahasa yang aktif di workspace.\n")

        found_any = False
        for lang in active_langs:
            if lang in guidelines_map:
                found_any = True
                compiled_rules.append(f"## Aturan Gaya Koding untuk: {lang.upper()}")
                compiled_rules.append(guidelines_map[lang].strip())
                compiled_rules.append("\n" + "=" * 40 + "\n")

        if not found_any:
            print("Tidak ditemukan panduan spesifik di database. Menggunakan aturan umum bawaan.")
            compiled_rules.append("## Aturan Umum")
            compiled_rules.append("Ikuti standar koding best-practices untuk bahasa yang aktif di dalam workspace.")

        rules_content = "\n".join(compiled_rules)

        with open(".clinerules", "w", encoding="utf-8") as f:
            f.write(rules_content)

        print("Sukses: Berkas .clinerules berhasil disinkronkan.")
    except Exception as exc:
        print(f"Gagal melakukan sinkronisasi aturan koding: {exc}")
    finally:
        close_connection(conn, cursor)


if __name__ == "__main__":
    sync_rules()
