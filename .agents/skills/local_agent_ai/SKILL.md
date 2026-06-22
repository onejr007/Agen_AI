---
name: local-agent-ai
description: Gunakan skill ini untuk memanggil Agent AI lokal (Qwen 1.5b) atau mencari memori dan database pengetahuan lokal di XAMPP MySQL.
---
# Skill Agent AI Lokal

Skill ini digunakan oleh Antigravity untuk berkonsultasi dengan Agent AI lokal pengguna dan memori RAG di database MySQL XAMPP.

## Kapan Menggunakan
- Gunakan jika Anda ingin mencari riwayat chat lama di database `agent_db` (XAMPP MySQL).
- Gunakan jika Anda ingin mencari pengetahuan kustom (RAG) yang telah ditambahkan pengguna.
- Gunakan jika Anda ingin mendelegasikan tugas pemrosesan lokal ke LLM `qwen2.5-coder:1.5b`.

## Cara Menggunakan
Jalankan script `query_local_agent.py` di terminal dengan prompt pencarian Anda:
```bash
python .agents/skills/local_agent_ai/scripts/query_local_agent.py "Cari tahu tentang task.defer() di Luau"
```
