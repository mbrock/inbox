# inbox

Personal Gmail triage pipeline. Syncs mail to SQLite, downloads PDF/XLS
attachments, classifies each message with Gemini (structured `Gist` schema:
category × intent × four triage flags), prints a week-by-week priority view.

## Prereqs

- `uv` (`brew install uv`)
- `libreoffice` (`brew install --cask libreoffice` / `apt install libreoffice`)
  — used to convert `.xls`/`.xlsx` attachments to PDF so Gemini can read them.
- `GEMINI_API_KEY` in the environment.
- `gmail-sync/credentials.json` — OAuth desktop-app credentials from
  Google Cloud Console (Gmail API enabled). First run authorizes and writes
  `gmail-sync/token.json`.

## Common commands

```bash
./life mail sync              # fetch new mail + gist the new stuff
./life mail gists             # replay cached gists, week-by-week
./life mail gists --grep fly  # filter by substring
./life mail gist --test       # dry-run of the classifier (no DB write)
./life mail reply <id> "..."  # reply to a message by 8-char ID
./life mail backfill          # rescan already-synced messages for PDFs/XLSs
```

## Layout

- `life` — click-based CLI. Only the `mail` subcommands are used here;
  the `bank`/`telegram`/`journal` groups are no-ops unless their env vars
  are set.
- `gmail-sync/sync.py` — Gmail API fetch, SQLite writes, attachment
  downloads (PDF + XLS + XLSX + ODS, non-PDFs auto-converted via
  `soffice`).
- `gmail-sync/extract.py` — Gemini call, `Gist` schema, renderer.
- `gmail-sync/gmail.db` — SQLite (ignored). Tables: `messages`,
  `attachments`, `invoice_extractions` (the gist cache).

## Data that lives outside git

Copy these from an existing install:

- `gmail-sync/gmail.db` (cached messages + gists)
- `gmail-sync/credentials.json` (OAuth app)
- `gmail-sync/token.json` (OAuth user)
- `~/pdf/` (downloaded attachments; optional — can re-sync)
