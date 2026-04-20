# inbox

Personal Gmail triage pipeline. Syncs mail to SQLite, downloads PDF/XLS
attachments, classifies each message with Gemini (structured `Gist` schema:
category × intent × four triage flags), prints a week-by-week priority view.

## Prereqs

- `uv` (`brew install uv`)
- `libreoffice` (`brew install --cask libreoffice` / `apt install libreoffice`)
  — used to convert `.xls`/`.xlsx` attachments to PDF so Gemini can read them.
- `GOOGLE_API_KEY` in the environment (Gemini API key).
- `credentials.json` — OAuth desktop-app credentials from Google Cloud
  Console (Gmail API enabled). First run authorizes and writes `token.json`.

## Common commands

```bash
uv run sync.py sync                   # fetch new mail + classify (flash-lite, parallel, all sources)
uv run sync.py sync --days 14 --pro   # deeper window + upgrade to Gemini 3 Pro
uv run sync.py sync --no-analyze      # fetch only, skip the classifier
uv run sync.py gists                  # replay cached gists, week-by-week
uv run sync.py gists --grep fly       # filter by substring
uv run sync.py analyze --dry-run --limit 10   # dry-run classifier (no DB write)
uv run sync.py list                   # recent messages + 8-char IDs
uv run sync.py search <term>          # local-DB substring search on from/subject
uv run sync.py search-remote <query>  # live Gmail API search (no sync)
uv run sync.py reply <id> "..."       # reply by 8-char ID prefix (preview + confirm)
uv run sync.py backfill               # rescan synced messages for PDFs/XLSs
```

## Layout

- `sync.py` — click CLI. Gmail API fetch, SQLite writes, attachment
  downloads (PDF + XLS + XLSX + ODS, non-PDFs auto-converted via `soffice`),
  and the triage subcommands (`list`, `search`, `reply`, …).
- `extract.py` — Gemini call, `Gist` schema, renderer.
- `gmail.db` — SQLite (ignored). Tables: `messages`, `attachments`,
  `invoice_extractions` (the gist cache).

## Data that lives outside git

Copy these from an existing install:

- `gmail.db` (cached messages + gists)
- `credentials.json` (OAuth app)
- `token.json` (OAuth user)
- `~/pdf/` (downloaded attachments; optional — can re-sync)
