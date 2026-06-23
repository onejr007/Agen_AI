import argparse
import json
import datetime
from knowledge import (
    get_db_connection,
    execute_sql,
    close_connection,
    calculate_cosine_similarity,
    get_config_value,
)


def deduplicate_rag(threshold: float = 0.95, dry_run: bool = False):
    """Mendeteksi dan membersihkan entri RAG yang serupa di tabel knowledge_base."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        execute_sql(cursor, "SELECT id, title, tags, embedding FROM knowledge_base ORDER BY id ASC")
        rows = cursor.fetchall()
        if not rows:
            print("Tabel knowledge_base kosong.")
            return

        print(f"Membaca {len(rows)} entri dari knowledge_base...")
        entries = []
        for row in rows:
            entry_id, title, tags, raw_embed = row
            vector = []
            if raw_embed:
                try:
                    vector = json.loads(raw_embed) if isinstance(raw_embed, str) else raw_embed
                except Exception:
                    pass
            if vector:
                entries.append({
                    "id": entry_id,
                    "title": title,
                    "tags": tags or "",
                    "vector": vector
                })

        deleted_ids = set()
        merged_updates = {}  # id -> new_tags

        for i in range(len(entries)):
            id_i = entries[i]["id"]
            if id_i in deleted_ids:
                continue

            for j in range(i + 1, len(entries)):
                id_j = entries[j]["id"]
                if id_j in deleted_ids:
                    continue

                sim = calculate_cosine_similarity(entries[i]["vector"], entries[j]["vector"])
                if sim >= threshold:
                    print(f"Mendeteksi duplikat (>={threshold:.2f}):")
                    print(f"  - [{id_i}] {entries[i]['title']}")
                    print(f"  - [{id_j}] {entries[j]['title']} (Sim: {sim:.4f})")

                    # Gabungkan tags
                    tags_i = merged_updates.get(id_i, entries[i]["tags"])
                    tags_j = entries[j]["tags"]

                    set_i = {t.strip().lower() for t in tags_i.split(",") if t.strip()}
                    set_j = {t.strip().lower() for t in tags_j.split(",") if t.strip()}
                    merged_set = set_i.union(set_j)
                    merged_tags = ",".join(sorted(merged_set))

                    merged_updates[id_i] = merged_tags
                    deleted_ids.add(id_j)
                    print(f"  -> Akan digabung ke [{id_i}] dengan tag baru: '{merged_tags}', dan [{id_j}] akan dihapus.")

        if not dry_run:
            for keep_id, new_tags in merged_updates.items():
                execute_sql(cursor, "UPDATE knowledge_base SET tags = %s WHERE id = %s", (new_tags, keep_id))
            for del_id in deleted_ids:
                execute_sql(cursor, "DELETE FROM knowledge_base WHERE id = %s", (del_id,))
            conn.commit()
            print(f"\nSukses: Berhasil memperbarui {len(merged_updates)} entri dan menghapus {len(deleted_ids)} duplikat.")
        else:
            print(f"\n[Dry Run] Akan memperbarui {len(merged_updates)} entri dan menghapus {len(deleted_ids)} duplikat.")

    except Exception as exc:
        print(f"Error saat deduplikasi: {exc}")
    finally:
        close_connection(conn, cursor)


def prune_chat_history(prune_days: int, dry_run: bool = False):
    """Menghapus sesi chat dan pesan lama yang berumur lebih dari prune_days."""
    if prune_days <= 0:
        return

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=prune_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\nMencari riwayat chat yang dibuat sebelum {cutoff_str} (lebih dari {prune_days} hari lalu)...")

        execute_sql(cursor, "SELECT id, title, created_at FROM chats WHERE created_at < %s", (cutoff_str,))
        rows = cursor.fetchall()

        if not rows:
            print("Tidak ditemukan riwayat chat lama yang perlu dipangkas.")
            return

        print(f"Menemukan {len(rows)} sesi chat lama untuk dihapus.")

        for row in rows:
            chat_id, title, created_at = row
            print(f"  - [{chat_id}] '{title}' (Dibuat: {created_at})")

        if not dry_run:
            chat_ids = [r[0] for r in rows]
            for c_id in chat_ids:
                execute_sql(cursor, "DELETE FROM messages WHERE chat_id = %s", (c_id,))
                execute_sql(cursor, "DELETE FROM chats WHERE id = %s", (c_id,))
            conn.commit()
            print("Sukses: Riwayat chat lama berhasil dipangkas.")
        else:
            print(f"[Dry Run] Riwayat chat lama ({len(rows)} sesi) akan dihapus.")

    except Exception as exc:
        print(f"Error saat memangkas riwayat chat: {exc}")
    finally:
        close_connection(conn, cursor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pembersih & Pengurangan Duplikasi Memori RAG AgentAI")
    parser.add_argument("--threshold", type=float, default=0.95, help="Ambang batas kesamaan kosinus duplikasi (default: 0.95)")
    parser.add_argument("--prune-days", type=int, default=0, help="Hapus riwayat chat yang lebih tua dari N hari (0 = nonaktif)")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan simulasi tanpa melakukan penulisan ke database")

    args = parser.parse_args()
    deduplicate_rag(threshold=args.threshold, dry_run=args.dry_run)
    if args.prune_days > 0:
        prune_chat_history(prune_days=args.prune_days, dry_run=args.dry_run)
