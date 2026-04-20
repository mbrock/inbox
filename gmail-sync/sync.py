#!/usr/bin/env python3
"""Sync Gmail messages, download PDF attachments, and send replies."""

import os
import re
import json
import time
import random
import sqlite3
import base64
import hashlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime

from email.utils import parsedate

from dotenv import load_dotenv


def parse_date(s):
    try:
        t = parsedate(s)
        return f"{t[0]}-{t[1]:02d}-{t[2]:02d}" if t else (s or '')[:10]
    except Exception:
        return (s or '')[:10]

# Load environment from ~/.env
load_dotenv(Path.home() / '.env')

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Need both read and send scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send'
]
SCRIPT_DIR = Path(__file__).parent
CREDENTIALS_FILE = SCRIPT_DIR / 'credentials.json'
TOKEN_FILE = SCRIPT_DIR / 'token.json'
DB_FILE = SCRIPT_DIR / 'gmail.db'
PDF_DIR = Path.home() / 'pdf'

DOC_EXTENSIONS = ('.pdf', '.xls', '.xlsx', '.ods')


def get_gmail_service():
    """Authenticate and return Gmail API service."""
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def compute_body_hash(subject: str, body: str) -> str | None:
    text = f"Subject: {subject or ''}\n\n{body or ''}".strip()
    if not text:
        return None
    return hashlib.sha256(text.encode()).hexdigest()


def _backfill_internal_dates(conn):
    rows = conn.execute(
        "SELECT id, date FROM messages WHERE internal_date IS NULL AND date IS NOT NULL AND date != ''"
    ).fetchall()
    if not rows:
        return
    for mid, date_str in rows:
        try:
            ts = int(parsedate_to_datetime(date_str).timestamp())
        except Exception:
            continue
        conn.execute('UPDATE messages SET internal_date = ? WHERE id = ?', (ts, mid))
    conn.commit()


def _backfill_body_hashes(conn):
    rows = conn.execute(
        "SELECT id, subject, body_text FROM messages WHERE body_hash IS NULL AND body_text IS NOT NULL AND body_text != ''"
    ).fetchall()
    if not rows:
        return
    for mid, subj, body in rows:
        h = compute_body_hash(subj, body)
        if h:
            conn.execute('UPDATE messages SET body_hash = ? WHERE id = ?', (h, mid))
    conn.commit()


def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            label_ids TEXT,
            snippet TEXT,
            subject TEXT,
            from_addr TEXT,
            to_addr TEXT,
            date TEXT,
            body_text TEXT,
            body_html TEXT,
            raw_payload TEXT,
            synced_at TEXT
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_thread_id ON messages(thread_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON messages(date)')

    # PDF attachments table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY,
            message_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            size_bytes INTEGER,
            local_path TEXT,
            downloaded_at TEXT,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_attachments_hash ON attachments(content_hash)')

    # Invoice extraction cache
    conn.execute('''
        CREATE TABLE IF NOT EXISTS invoice_extractions (
            id INTEGER PRIMARY KEY,
            content_hash TEXT NOT NULL UNIQUE,
            model_name TEXT NOT NULL,
            extracted_json TEXT,
            created_at TEXT
        )
    ''')

    # Migrate to compound unique key if needed (allows multiple models per PDF)
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='invoice_extractions'"
    ).fetchone()
    if schema and 'UNIQUE(content_hash, model_name)' not in schema[0]:
        conn.execute('''
            CREATE TABLE invoice_extractions_new (
                id INTEGER PRIMARY KEY,
                content_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                extracted_json TEXT,
                created_at TEXT,
                UNIQUE(content_hash, model_name)
            )
        ''')
        conn.execute('INSERT OR IGNORE INTO invoice_extractions_new SELECT * FROM invoice_extractions')
        conn.execute('DROP TABLE invoice_extractions')
        conn.execute('ALTER TABLE invoice_extractions_new RENAME TO invoice_extractions')

    # Add message_id column if not present (for email gists)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(invoice_extractions)").fetchall()]
    if 'message_id' not in cols:
        conn.execute('ALTER TABLE invoice_extractions ADD COLUMN message_id TEXT')

    # Add body_hash column on messages (sha256 of "Subject: ...\n\nbody") so the
    # "which emails still need a gist?" query can dedupe identical bodies that
    # otherwise each re-hit Gemini on every run. Backfill any missing rows now.
    msg_cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if 'body_hash' not in msg_cols:
        conn.execute('ALTER TABLE messages ADD COLUMN body_hash TEXT')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_body_hash ON messages(body_hash)')
    _backfill_body_hashes(conn)

    # Parsed timestamp from the `date` header (unix seconds) so SQL ORDER BY
    # actually sorts chronologically instead of lex-sorting RFC 2822 strings.
    if 'internal_date' not in msg_cols:
        conn.execute('ALTER TABLE messages ADD COLUMN internal_date INTEGER')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_internal_date ON messages(internal_date)')
    _backfill_internal_dates(conn)

    conn.commit()
    return conn


def get_header(headers, name):
    """Extract header value by name."""
    for h in headers:
        if h['name'].lower() == name.lower():
            return h['value']
    return None


def get_body_parts(payload):
    """Recursively extract text and html body parts."""
    text = None
    html = None

    if 'body' in payload and payload['body'].get('data'):
        data = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
        mime = payload.get('mimeType', '')
        if 'text/plain' in mime:
            text = data
        elif 'text/html' in mime:
            html = data

    if 'parts' in payload:
        for part in payload['parts']:
            t, h = get_body_parts(part)
            if t and not text:
                text = t
            if h and not html:
                html = h

    return text, html


def extract_doc_attachments(payload):
    """Recursively extract attachment info for DOC_EXTENSIONS (pdf, xls, xlsx, ods)."""
    attachments = []

    def match(filename):
        return filename.lower().endswith(DOC_EXTENSIONS)

    def process_parts(parts):
        for part in parts:
            if 'parts' in part:
                process_parts(part['parts'])
                continue
            filename = part.get('filename', '')
            if match(filename) and part.get('body', {}).get('attachmentId'):
                attachments.append({
                    'attachment_id': part['body']['attachmentId'],
                    'filename': filename,
                    'size': part['body'].get('size', 0)
                })

    if 'parts' in payload:
        process_parts(payload['parts'])
    elif match(payload.get('filename', '')):
        if payload.get('body', {}).get('attachmentId'):
            attachments.append({
                'attachment_id': payload['body']['attachmentId'],
                'filename': payload['filename'],
                'size': payload['body'].get('size', 0)
            })

    return attachments


def parse_email_date(msg):
    """Parse email date from message headers or internal date."""
    payload = msg.get('payload', {})
    headers = payload.get('headers', [])
    date_header = get_header(headers, 'Date')

    if date_header:
        try:
            return parsedate_to_datetime(date_header)
        except (ValueError, TypeError):
            pass

    # Fallback to internal date
    internal_date = msg.get('internalDate')
    if internal_date:
        return datetime.fromtimestamp(int(internal_date) / 1000)

    return datetime.now()


def extract_sender_domain(from_addr):
    """Extract domain from sender email address."""
    if not from_addr:
        return 'unknown.com'

    # Extract email from "Name <email@domain.com>" format
    match = re.search(r'<([^>]+)>', from_addr)
    email = match.group(1) if match else from_addr.strip()

    if '@' in email:
        return email.split('@')[-1].lower()
    return 'unknown.com'


def generate_doc_path(msg, filename):
    """Generate file path preserving the attachment's original extension."""
    email_date = parse_email_date(msg)
    year_month = email_date.strftime('%Y-%m')
    date_str = email_date.strftime('%Y%m%d')

    from_addr = get_header(msg.get('payload', {}).get('headers', []), 'From')
    domain = extract_sender_domain(from_addr)

    msg_id = msg.get('id', 'unknown')[:8]
    ext = Path(filename).suffix.lower() or '.bin'
    stem = Path(filename).stem
    clean_stem = re.sub(r'[<>:"/\\|?*]', '_', stem)[:30]

    final_filename = f"{date_str}-{domain}-{msg_id}-{clean_stem}{ext}"
    return PDF_DIR / year_month / domain / final_filename


def convert_to_pdf(raw_path: Path) -> Path | None:
    """Run LibreOffice headless to convert any supported doc to PDF alongside the raw file."""
    if raw_path.suffix.lower() == '.pdf':
        return raw_path
    try:
        subprocess.run(
            ['soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(raw_path.parent), str(raw_path)],
            check=True, capture_output=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ! soffice conversion failed for {raw_path.name}: {e}")
        return None
    pdf_path = raw_path.with_suffix('.pdf')
    return pdf_path if pdf_path.exists() else None


# Gmail enforces a per-user concurrent-request ceiling (roughly 10-ish) and the
# batch endpoint fans sub-requests out in parallel server-side, so a big batch
# quickly trips "Too many concurrent requests for user" 429s. Keep chunks
# comfortably under the ceiling and lean on per-sub-request retry for the rest.
BATCH_SIZE = 10
MAX_RETRIES = 6


def _is_rate_limited(exc):
    if isinstance(exc, HttpError):
        return exc.resp.status in (403, 429, 500, 503)
    return False


def _batch_execute(service, req_factories):
    """req_factories: list of (request_id, callable_returning_fresh_HttpRequest).

    Dispatches through Gmail's batch endpoint in chunks of BATCH_SIZE. On 429 /
    rate-limit errors, retries the failing sub-requests (rebuilt via the factory)
    with full-jitter exponential backoff. Returns (results_by_id, errors_by_id).
    """
    factory_by_id = dict(req_factories)
    results: dict[str, dict] = {}
    errors: dict[str, Exception] = {}
    pending = [rid for rid, _ in req_factories]

    for attempt in range(MAX_RETRIES + 1):
        if not pending:
            break

        round_errors: dict[str, Exception] = {}

        def _cb(request_id, response, exception):
            if exception is not None:
                round_errors[request_id] = exception
            else:
                results[request_id] = response

        for i in range(0, len(pending), BATCH_SIZE):
            chunk_ids = pending[i:i + BATCH_SIZE]
            batch = service.new_batch_http_request(callback=_cb)
            for rid in chunk_ids:
                batch.add(factory_by_id[rid](), request_id=rid)
            batch.execute()

        retryable = []
        for rid, exc in round_errors.items():
            if attempt < MAX_RETRIES and _is_rate_limited(exc):
                retryable.append(rid)
            else:
                errors[rid] = exc

        if not retryable:
            break

        # Full-jitter exponential backoff, capped at 30s.
        delay = random.uniform(0, min(2 ** (attempt + 1), 30))
        print(
            f"  ~ rate-limited on {len(retryable)} req(s); retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s",
            flush=True,
        )
        time.sleep(delay)
        pending = retryable

    return results, errors


def _batch_get_messages(service, msg_ids):
    reqs = [
        (mid, (lambda mid=mid: service.users().messages().get(
            userId='me', id=mid, format='full')))
        for mid in msg_ids
    ]
    return _batch_execute(service, reqs)


def _batch_get_attachments(service, att_keys):
    """att_keys: list of (key, message_id, attachment_id). Returns results_by_key."""
    reqs = [
        (
            key,
            (lambda mid=mid, aid=aid: service.users().messages().attachments().get(
                userId='me', messageId=mid, id=aid)),
        )
        for key, mid, aid in att_keys
    ]
    return _batch_execute(service, reqs)


def _save_attachment(conn, msg, att, raw_bytes):
    """Persist a single decoded attachment. Returns a status dict matching the
    shape that download_doc_attachments used to return."""
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    existing = conn.execute(
        'SELECT local_path FROM attachments WHERE content_hash = ?',
        (content_hash,)
    ).fetchone()
    if existing:
        return {'filename': att['filename'], 'path': existing[0], 'status': 'exists'}

    raw_path = generate_doc_path(msg, att['filename'])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_bytes)

    pdf_path = convert_to_pdf(raw_path)
    stored_path = pdf_path if pdf_path else raw_path

    conn.execute('''
        INSERT INTO attachments (message_id, filename, content_hash, size_bytes, local_path, downloaded_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (msg['id'], att['filename'], content_hash, len(raw_bytes), str(stored_path), datetime.now().isoformat()))

    return {'filename': att['filename'], 'path': str(stored_path), 'status': 'downloaded'}


def download_doc_attachments(service, conn, msg):
    """Download PDF/XLS/XLSX/ODS attachments and store in database.

    Non-PDF attachments are additionally converted to PDF via soffice so the
    downstream Gemini pipeline (which expects application/pdf) works uniformly.
    The DB's local_path points at the PDF (converted or native); raw originals
    sit alongside.

    Used by backfill (per-message); sync_messages uses the batched path below.
    """
    payload = msg.get('payload', {})
    attachments = extract_doc_attachments(payload)

    if not attachments:
        return []

    downloaded = []
    msg_id = msg['id']

    for att in attachments:
        att_data = service.users().messages().attachments().get(
            userId='me',
            messageId=msg_id,
            id=att['attachment_id']
        ).execute()
        data = base64.urlsafe_b64decode(att_data['data'])
        downloaded.append(_save_attachment(conn, msg, att, data))

    return downloaded


def sync_messages(service, conn, days=7, with_pdfs=True):
    """Sync messages from the past N days. PDF/XLS/XLSX/ODS attachments are
    always downloaded (and non-PDFs converted to PDF via soffice) unless
    with_pdfs=False is passed explicitly."""
    after_date = (datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')
    query = f'after:{after_date}'

    print(f"Listing message IDs from the past {days} days...", flush=True)

    ids = []
    page_token = None
    while True:
        results = service.users().messages().list(
            userId='me',
            q=query,
            pageToken=page_token,
            maxResults=500,
        ).execute()
        if 'messages' in results:
            ids.extend(m['id'] for m in results['messages'])
            print(f"  Found {len(ids)} message IDs so far...", flush=True)
        page_token = results.get('nextPageToken')
        if not page_token:
            break

    # Filter out already-synced messages up front — one SQL round-trip instead of
    # one Gmail API round-trip per message.
    if ids:
        placeholders = ','.join('?' * len(ids))
        existing = {row[0] for row in conn.execute(
            f'SELECT id FROM messages WHERE id IN ({placeholders})', ids
        ).fetchall()}
    else:
        existing = set()

    new_ids = [i for i in ids if i not in existing]
    print(f"Total: {len(ids)} in range, {len(existing)} already synced, {len(new_ids)} to fetch")

    new_count = 0
    att_count = 0
    processed = 0
    for batch_start in range(0, len(new_ids), BATCH_SIZE):
        chunk = new_ids[batch_start:batch_start + BATCH_SIZE]
        msgs, msg_errs = _batch_get_messages(service, chunk)
        for mid, exc in msg_errs.items():
            print(f"  ! fetch failed for {mid}: {exc}", flush=True)

        # Collect all attachment fetches for this chunk into one batch.
        att_payloads = {}
        if with_pdfs:
            att_keys = []
            att_meta = {}
            for mid, msg in msgs.items():
                for att in extract_doc_attachments(msg.get('payload', {})):
                    key = f"{mid}:{att['attachment_id']}"
                    att_keys.append((key, mid, att['attachment_id']))
                    att_meta[key] = (mid, att)
            if att_keys:
                att_payloads, att_errs = _batch_get_attachments(service, att_keys)
                for key, exc in att_errs.items():
                    _, att = att_meta[key]
                    print(f"  ! attachment fetch failed ({att['filename']}): {exc}", flush=True)

        for mid in chunk:
            processed += 1
            if mid not in msgs:
                continue
            msg = msgs[mid]
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])

            subject = get_header(headers, 'Subject')
            from_addr = get_header(headers, 'From')
            to_addr = get_header(headers, 'To')
            date = get_header(headers, 'Date')

            body_text, body_html = get_body_parts(payload)

            try:
                internal_date = int(parsedate_to_datetime(date).timestamp()) if date else None
            except Exception:
                internal_date = None

            conn.execute('''
                INSERT OR REPLACE INTO messages
                (id, thread_id, label_ids, snippet, subject, from_addr, to_addr, date, body_text, body_html, raw_payload, synced_at, body_hash, internal_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mid,
                msg.get('threadId'),
                json.dumps(msg.get('labelIds', [])),
                msg.get('snippet'),
                subject,
                from_addr,
                to_addr,
                date,
                body_text,
                body_html,
                json.dumps(payload),
                datetime.now().isoformat(),
                compute_body_hash(subject, body_text),
                internal_date,
            ))

            if with_pdfs:
                for att in extract_doc_attachments(payload):
                    key = f"{mid}:{att['attachment_id']}"
                    resp = att_payloads.get(key)
                    if not resp or 'data' not in resp:
                        continue
                    data = base64.urlsafe_b64decode(resp['data'])
                    d = _save_attachment(conn, msg, att, data)
                    status = "✓" if d['status'] == 'downloaded' else "○"
                    print(f"  {status} {d['filename']}")
                    if d['status'] == 'downloaded':
                        att_count += 1

            new_count += 1
            print(f"  [{processed}/{len(new_ids)}] {subject[:50] if subject else '(no subject)'}", flush=True)

        # One commit per batch instead of per message.
        conn.commit()

    pdf_total = conn.execute('SELECT COUNT(*) FROM attachments').fetchone()[0]
    print(f"Done! {new_count} new message(s), {att_count} new attachment(s). DB now holds {pdf_total} attachments.")


def backfill_attachments(service, conn, days=7):
    """Scan messages already in DB (within `days`) for DOC_EXTENSIONS attachments
    that haven't been downloaded yet. Uses the stored raw_payload instead of
    re-fetching from Gmail."""
    since = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        '''SELECT id, raw_payload FROM messages
           WHERE synced_at >= ? OR date >= ?''',
        (since, since)
    ).fetchall()

    # First pass: walk stored payloads and build a flat list of attachments that
    # still need bytes. Second pass: one batched fetch for everything.
    pending = []  # list of (key, msg_stub, att)
    for msg_id, raw in rows:
        try:
            payload = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            continue
        atts = extract_doc_attachments(payload)
        if not atts:
            continue
        # Skip if we already have at least one attachment row for this message —
        # a good-enough heuristic to avoid re-scanning everything.
        have_any = conn.execute(
            'SELECT 1 FROM attachments WHERE message_id = ? LIMIT 1',
            (msg_id,)
        ).fetchone()
        if have_any:
            continue
        # Fake a minimal msg dict for generate_doc_path (needs headers + id)
        msg_stub = {'id': msg_id, 'payload': payload}
        for att in atts:
            key = f"{msg_id}:{att['attachment_id']}"
            pending.append((key, msg_stub, att))

    if not pending:
        return 0

    att_keys = [(key, stub['id'], att['attachment_id']) for key, stub, att in pending]
    results, errs = _batch_get_attachments(service, att_keys)
    for key, exc in errs.items():
        print(f"  ! backfill fetch failed ({key}): {exc}", flush=True)

    new_att = 0
    for key, stub, att in pending:
        resp = results.get(key)
        if not resp or 'data' not in resp:
            continue
        data = base64.urlsafe_b64decode(resp['data'])
        d = _save_attachment(conn, stub, att, data)
        status = "✓" if d['status'] == 'downloaded' else "○"
        print(f"  backfill {status} {d['filename']}")
        if d['status'] == 'downloaded':
            new_att += 1

    conn.commit()
    return new_att


def send_reply(service, conn, original_message_id: str, body_text: str):
    """Send a reply to an existing message."""
    # Get original message details from DB
    row = conn.execute(
        'SELECT thread_id, subject, from_addr, id FROM messages WHERE id = ?',
        (original_message_id,)
    ).fetchone()

    if not row:
        raise ValueError(f"Message {original_message_id} not found in database")

    thread_id, subject, from_addr, msg_id = row

    # Extract email address from "Name <email>" format
    if '<' in from_addr:
        to_email = from_addr.split('<')[1].rstrip('>')
    else:
        to_email = from_addr

    # Ensure subject has Re: prefix
    if not subject.lower().startswith('re:'):
        subject = f"Re: {subject}"

    # Create the email message
    message = MIMEText(body_text)
    message['to'] = to_email
    message['subject'] = subject
    message['In-Reply-To'] = msg_id
    message['References'] = msg_id

    # Encode the message
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

    # Send as reply in the same thread
    result = service.users().messages().send(
        userId='me',
        body={
            'raw': raw,
            'threadId': thread_id
        }
    ).execute()

    print(f"Reply sent! Message ID: {result['id']}")
    return result


def get_message_by_search(conn, search_term: str):
    """Find a message by searching from_addr or subject."""
    row = conn.execute(
        '''SELECT id, thread_id, subject, from_addr, date
           FROM messages
           WHERE from_addr LIKE ? OR subject LIKE ?
           ORDER BY date DESC
           LIMIT 1''',
        (f'%{search_term}%', f'%{search_term}%')
    ).fetchone()

    if row:
        return {
            'id': row[0],
            'thread_id': row[1],
            'subject': row[2],
            'from_addr': row[3],
            'date': row[4]
        }
    return None


def describe_all_documents(conn, source='pdfs', model=None, parallel=False, workers=4, limit=None, gist=False, dry_run=False, thinking='medium', emoji=False, show_noise=False):
    from extract import describe_pdf, describe_email, describe_documents_parallel, print_gist_row, print_gist_week, _week_bucket, build_id_shortener, MODEL_PRO

    if model is None:
        model = MODEL_PRO

    cache_key = f"{model}:gist:v3" if gist else model
    tasks = []

    if source in ('pdfs', 'all'):
        pdf_query = '''
            SELECT a.local_path, a.content_hash, a.filename, m.date, m.from_addr
            FROM attachments a
            JOIN messages m ON a.message_id = m.id
            LEFT JOIN invoice_extractions e ON a.content_hash = e.content_hash AND e.model_name = ?
            WHERE e.id IS NULL
            ORDER BY COALESCE(m.internal_date, 0) DESC
        '''
        if limit and source == 'pdfs':
            pdf_query += f' LIMIT {limit}'
        for path, _, filename, date, from_addr in conn.execute(pdf_query, (cache_key,)).fetchall():
            if Path(path).exists():
                tasks.append({'type': 'pdf', 'path': path, 'label': filename, 'model': model,
                              'gist': gist, 'date': date, 'sender': from_addr or '', 'dry_run': dry_run,
                              'thinking': thinking})

    if source in ('emails', 'all'):
        # Dedupe by body_hash so identical emails (e.g. repeated FLY.IO payment
        # failures) aren't re-analyzed. A cached row for ANY prior sibling with
        # the same hash counts as "already analyzed" for this message too.
        #
        # In --dry-run we bypass both filters so you see results on real recent
        # traffic even if it's already been gisted.
        # Skip outgoing mail (messages I sent) — not useful to triage my own stuff.
        # Accept HTML-only messages too — many automated senders (Tuul, Revolut,
        # AWS) omit text/plain. We'll fall back to body_html → snippet below.
        SENT_FILTER = "AND (m.label_ids IS NULL OR m.label_ids NOT LIKE '%\"SENT\"%')"
        HAS_BODY = (
            "(COALESCE(NULLIF(m.body_text,''), NULLIF(m.body_html,''), NULLIF(m.snippet,'')) IS NOT NULL)"
        )
        if dry_run:
            email_query = f'''
                SELECT m.id, m.subject, m.from_addr, m.body_text, m.body_html, m.snippet, m.date
                FROM messages m
                WHERE {HAS_BODY}
                  {SENT_FILTER}
                ORDER BY COALESCE(m.internal_date, 0) DESC
            '''
        else:
            email_query = f'''
                SELECT m.id, m.subject, m.from_addr, m.body_text, m.body_html, m.snippet, m.date
                FROM messages m
                WHERE {HAS_BODY}
                  {SENT_FILTER}
                  AND NOT EXISTS (
                        SELECT 1 FROM invoice_extractions e
                        WHERE e.model_name = ?
                          AND (e.message_id = m.id OR (m.body_hash IS NOT NULL AND e.content_hash = m.body_hash))
                  )
                ORDER BY COALESCE(m.internal_date, 0) DESC
            '''
        if limit and source == 'emails':
            email_query += f' LIMIT {limit}'
        params = () if dry_run else (cache_key,)
        for msg_id, subject, from_addr, body_text, body_html, snippet, date in conn.execute(email_query, params).fetchall():
            body = body_text or body_html or snippet or ''
            tasks.append({
                'type': 'email',
                'message_id': msg_id,
                'msg_id': msg_id,
                'subject': subject or '',
                'body': body,
                'label': subject or msg_id,
                'model': model,
                'gist': gist,
                'date': date,
                'sender': from_addr or '',
                'dry_run': dry_run,
                'thinking': thinking,
            })

    if limit and source == 'all':
        tasks = tasks[:limit]

    if not tasks:
        print("Nothing left to describe.")
        return

    error_count = 0
    if parallel and len(tasks) > 1:
        from rich.console import Console as RichConsole
        rc = RichConsole()

        if gist:
            # Compute a stable id-shortener across ALL messages in the DB so
            # the short ID width is consistent across runs / across weeks.
            all_ids = [r[0] for r in conn.execute('SELECT id FROM messages').fetchall()]
            id_map = build_id_shortener(all_ids, safety=1)

            # Process one week at a time: oldest week first, "this week" last,
            # so each week's rows print together as they complete.
            from collections import defaultdict
            buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
            for t in tasks:
                week_idx, week_label = _week_bucket(t.get('date', ''))
                buckets[(week_idx, week_label)].append(t)

            for (week_idx, week_label) in sorted(buckets.keys(), reverse=True):
                week_tasks = buckets[(week_idx, week_label)]
                results, errs = describe_documents_parallel(week_tasks, workers=workers, emoji=emoji)
                error_count += errs
                if not dry_run:
                    for task, description in results:
                        if description and task['type'] == 'pdf':
                            suffix = '.gist.txt'
                            txt_path = Path(task['path']).with_name(Path(task['path']).stem + suffix)
                            txt_path.write_text(description, encoding='utf-8')
                print_gist_week(rc, week_label, results, emoji=emoji, show_noise=show_noise, id_map=id_map)
        else:
            results, error_count = describe_documents_parallel(tasks, workers=workers, emoji=emoji)
            if not dry_run:
                for task, description in results:
                    if description and task['type'] == 'pdf':
                        suffix = '.txt'
                        txt_path = Path(task['path']).with_name(Path(task['path']).stem + suffix)
                        txt_path.write_text(description, encoding='utf-8')
    else:
        from rich.console import Console as RichConsole
        rc = RichConsole()
        for i, task in enumerate(tasks):
            if task['type'] == 'pdf':
                description = describe_pdf(task['path'], model=model, gist=gist, quiet=True, dry_run=dry_run, thinking=thinking)
            else:
                description = describe_email(task['message_id'], task['subject'], task['body'],
                                             model=model, gist=gist, quiet=True, dry_run=dry_run,
                                             from_addr=task.get('sender') or None, thinking=thinking)

            if gist:
                if print_gist_row(rc, description, date=task.get('date', ''),
                                  sender=task.get('sender', ''), label=task['label'], emoji=emoji):
                    error_count += 1
            elif description:
                if task['type'] == 'pdf' and not dry_run:
                    suffix = '.gist.txt' if gist else '.txt'
                    txt_path = Path(task['path']).with_name(Path(task['path']).stem + suffix)
                    txt_path.write_text(description, encoding='utf-8')
                rc.print(f"  [green]✓[/green] [{i+1}/{len(tasks)}] {task['label']}")
            else:
                rc.print(f"  [red]✗[/red] [{i+1}/{len(tasks)}] {task['label']}")
                error_count += 1

    if error_count:
        print(f"\n[red]{error_count}/{len(tasks)} errors.[/red]")
    return error_count


def describe_all_pdfs(conn, **kwargs):
    """Backwards-compat wrapper."""
    describe_all_documents(conn, source='pdfs', **kwargs)


def search_gmail(service, query, max_results=20):
    """Search Gmail directly via API without syncing."""
    print(f"Searching Gmail for: {query}\n")

    results = service.users().messages().list(
        userId='me',
        q=query,
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])

    if not messages:
        print("No messages found.")
        return []

    print(f"Found {len(messages)} messages:\n")

    found = []
    for msg_info in messages:
        msg = service.users().messages().get(
            userId='me',
            id=msg_info['id'],
            format='metadata',
            metadataHeaders=['Subject', 'From', 'Date']
        ).execute()

        headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
        subject = headers.get('Subject', '(no subject)')
        from_addr = headers.get('From', '(unknown)')
        date = headers.get('Date', '')

        print(f"  {msg_info['id'][:12]}  {date[:16] if date else ''}")
        print(f"    From: {from_addr[:60]}")
        print(f"    Subject: {subject[:70]}")
        print()

        found.append({
            'id': msg_info['id'],
            'subject': subject,
            'from': from_addr,
            'date': date
        })

    return found


def main():
    import sys

    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  uv run sync.py sync [--days N] [--with-pdfs] [--analyze] [--flash] [--parallel] [--workers N]")
        print("                      Sync emails, download PDFs, optionally analyze them")
        print("  uv run sync.py search <query>")
        print("                      Search Gmail directly (e.g., 'contact lens' or 'from:amazon')")
        print("  uv run sync.py analyze [--flash] [--parallel] [--workers N]")
        print("                      Analyze all unprocessed PDFs with Gemini")
        print("  uv run sync.py extract [file.pdf]")
        print("                      Extract/describe a specific PDF")
        print("  uv run sync.py gists")
        print("                      List all one-line gist summaries in a table")
        print("  uv run sync.py pdfs")
        print("                      List downloaded PDFs")
        print("  uv run sync.py reply <message_id> <body>")
        print("                      Reply to a message")
        print()
        print("Options:")
        print("  --days N      Number of days to look back (default: 7)")
        print("  --with-pdfs   Download PDF attachments")
        print("  --analyze     Analyze PDFs after downloading")
        print("  --flash           Use Gemini 3 Flash instead of Pro")
        print("  --flash-lite      Use Gemini 3.1 Flash Lite (cheapest)")
        print("  --gist            One-line telegraph summary per document")
        print("  --emails          Analyze email bodies instead of PDFs")
        print("  --all             Analyze both PDFs and emails")
        print("  --parallel        Process PDFs in parallel")
        print("  --workers N       Number of parallel workers (default: 4)")
        print("  --limit N         Process only N PDFs, newest first")
        return

    cmd = args[0]
    # PDF/XLS download is on by default; --no-pdfs disables it. --with-pdfs kept as a no-op alias.
    with_pdfs = '--no-pdfs' not in args
    analyze = '--analyze' in args
    parallel = '--parallel' in args

    # Parse --days N, --workers N, --limit N
    days = 7
    workers = 4
    limit = None
    for i, arg in enumerate(args):
        if arg == '--days' and i + 1 < len(args):
            try:
                days = int(args[i + 1])
            except ValueError:
                pass
        if arg == '--workers' and i + 1 < len(args):
            try:
                workers = int(args[i + 1])
            except ValueError:
                pass
        if arg == '--limit' and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                pass

    gist = '--gist' in args
    dry_run = '--dry-run' in args
    emoji = '--emoji' in args
    show_noise = '--show-noise' in args

    # --thinking LEVEL   (minimal | low | medium | high)
    thinking = 'medium'
    for i, arg in enumerate(args):
        if arg == '--thinking' and i + 1 < len(args):
            if args[i + 1] in ('minimal', 'low', 'medium', 'high'):
                thinking = args[i + 1]

    # Source selection
    if '--emails' in args:
        source = 'emails'
    elif '--all' in args:
        source = 'all'
    else:
        source = 'pdfs'

    # Model selection
    from extract import MODEL_PRO, MODEL_FLASH, MODEL_FLASH_LITE
    if '--flash-lite' in args:
        model = MODEL_FLASH_LITE
    elif '--flash' in args:
        model = MODEL_FLASH
    else:
        model = MODEL_PRO

    if cmd == 'sync':
        service = get_gmail_service()
        conn = init_db()
        sync_messages(service, conn, days=days, with_pdfs=with_pdfs)

        if analyze:
            print(f"\nAnalyzing {source}...\n")
            describe_all_documents(conn, source=source, model=model, parallel=parallel, workers=workers, limit=limit, gist=gist)

        conn.close()

    elif cmd == 'analyze':
        conn = init_db()
        errs = describe_all_documents(conn, source=source, model=model, parallel=parallel, workers=workers, limit=limit, gist=gist, dry_run=dry_run, thinking=thinking, emoji=emoji, show_noise=show_noise) or 0
        conn.close()
        if errs:
            sys.exit(1)

    elif cmd == 'backfill':
        service = get_gmail_service()
        conn = init_db()
        n = backfill_attachments(service, conn, days=days)
        print(f"Backfilled {n} attachment(s).")
        conn.close()

    elif cmd == 'test-gist':
        # Alias for: analyze --emails --gist --dry-run --limit N --parallel
        conn = init_db()
        errs = describe_all_documents(conn, source='emails', model=model, parallel=True,
                                      workers=workers, limit=(limit or 10), gist=True, dry_run=True,
                                      thinking=thinking, emoji=emoji, show_noise=show_noise) or 0
        conn.close()
        if errs:
            sys.exit(1)

    elif cmd == 'pdfs':
        conn = init_db()
        rows = conn.execute('''
            SELECT a.filename, a.local_path, a.size_bytes, a.downloaded_at,
                   m.from_addr, m.subject
            FROM attachments a
            JOIN messages m ON a.message_id = m.id
            ORDER BY a.downloaded_at DESC
            LIMIT 20
        ''').fetchall()
        if rows:
            print(f"\nRecent PDFs ({len(rows)} shown):\n")
            for r in rows:
                size_kb = r[2] / 1024 if r[2] else 0
                print(f"  {r[0]} ({size_kb:.1f}KB)")
                print(f"    From: {r[4][:50]}")
                print(f"    Path: {r[1]}")
                print()
        else:
            print("No PDFs downloaded yet. Run: uv run sync.py sync --with-pdfs")
        conn.close()

    elif cmd == 'extract':
        # Delegate to extract.py
        from extract import main as extract_main
        extract_main(args[1:])

    elif cmd == 'reply':
        if len(args) < 3:
            print("Usage: uv run sync.py reply <message_id> <body>")
            return
        message_id = args[1]
        body = args[2]
        service = get_gmail_service()
        conn = init_db()
        send_reply(service, conn, message_id, body)
        conn.close()
        print("Reply sent!")

    elif cmd == 'gists':
        # Print all cached v3-schema gists, week-by-week, same layout as the
        # live `analyze --gist` run.
        from extract import print_gist_week, _week_bucket, build_id_shortener
        from rich.console import Console as RichConsole
        from collections import defaultdict
        rc = RichConsole()

        # Optional filters from argv
        grep_pat = None
        max_weeks = None
        for i, a in enumerate(args):
            if a == '--grep' and i + 1 < len(args):
                grep_pat = args[i + 1].lower()
            if a == '--weeks' and i + 1 < len(args):
                try: max_weeks = int(args[i + 1])
                except ValueError: pass

        conn = init_db()
        rows = conn.execute('''
            SELECT m.date, m.from_addr, e.extracted_json, m.id
            FROM invoice_extractions e
            JOIN messages m ON e.message_id = m.id
            WHERE e.model_name LIKE '%:gist:v3'
        ''').fetchall()

        if not rows:
            print("No structured gists cached yet. Run: uv run sync.py analyze --emails --gist")
            conn.close()
            return

        all_ids = [r[0] for r in conn.execute('SELECT id FROM messages').fetchall()]
        id_map = build_id_shortener(all_ids, safety=1)
        conn.close()

        buckets: dict = defaultdict(list)
        for date_str, from_addr, description, msg_id in rows:
            if grep_pat and grep_pat not in (description or '').lower():
                continue
            week_idx, week_label = _week_bucket(date_str or '')
            if max_weeks is not None and week_idx >= max_weeks:
                continue
            task_stub = {'date': date_str, 'sender': from_addr or '', 'label': msg_id, 'msg_id': msg_id}
            buckets[(week_idx, week_label)].append((task_stub, description))

        printed = 0
        for (week_idx, week_label) in sorted(buckets.keys(), reverse=True):
            printed += print_gist_week(rc, week_label, buckets[(week_idx, week_label)],
                                       emoji=emoji, show_noise=show_noise, id_map=id_map)
        if not printed:
            print("Nothing matches.")

    elif cmd == 'search':
        if len(args) < 2:
            print("Usage: uv run sync.py search <query>")
            print("Examples:")
            print("  uv run sync.py search 'contact lens'")
            print("  uv run sync.py search 'from:amazon'")
            print("  uv run sync.py search 'bio toric'")
            return
        query = ' '.join(args[1:])
        service = get_gmail_service()
        search_gmail(service, query)

    else:
        print(f"Unknown command: {cmd}")


if __name__ == '__main__':
    main()
