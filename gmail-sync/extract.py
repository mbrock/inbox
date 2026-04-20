#!/usr/bin/env python3
"""Describe PDF documents using Gemini AI."""

import os
import sqlite3
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


class Gist(BaseModel):
    """You are the executive assistant of a user with executive-function
    difficulties who is easily overwhelmed by email volume. Your job is to
    triage: surface the few things that truly matter, fade everything that
    doesn't. Telegraphic concision is appreciated.

    `category` is the broad domain. `intent` is the shape of the message.
    The four booleans (frivolous/broadcast/obligation/critical) are the
    main signal — they're independent and can co-occur (a critical
    obligation is the most urgent; a frivolous broadcast is the most
    ignorable).
    """

    category: Literal[
        "billing", "accounts",
        "booking", "orders",
        "newsletter", "marketing",
        "personal", "institution",
        "other",
    ] = Field(
        description=(
            "Broad domain. "
            "billing = invoices/payments/receipts. "
            "accounts = auth, 2FA codes, login/security alerts, password resets — anything about access to an account. "
            "booking = hotels, flights, restaurants, event reservations. "
            "orders = shipping notifications, order confirmations from shops. "
            "newsletter = periodic bulk content I opted into. "
            "marketing = promos, deals, product pushes from companies. "
            "personal = real humans writing to me directly. "
            "institution = banks, tax authorities, landlords, embassies, employers, schools "
            "(but if the email is specifically about my account's auth/security, use `accounts`). "
            "other = none of the above fit."
        )
    )
    intent: Literal[
        "failure", "success", "reminder", "request",
        "reply", "warning", "info",
    ] = Field(
        description=(
            "Shape of the message. "
            "failure = something went wrong (payment failed, build failed, delivery failed). "
            "success = something completed well (receipt, order shipped, payment confirmed, renewed). "
            "reminder = nudge to do a known thing (pay this invoice, renew subscription). "
            "request = a party wants something new from me (KYC docs, respond please, approve). "
            "reply = response inside an ongoing thread I am part of. "
            "warning = looming problem, not yet broken (threshold crossed, expiration approaching, deprecation). "
            "info = passive notification (newsletter, product update, policy update, announcement, 2FA code delivery)."
        )
    )
    frivolous: bool = Field(
        description=(
            "Engagement bait, mildly amusing content, sales, startup product pushes, "
            "promotional offers, holiday greetings — nothing that truly needs the user's "
            "attention. Skipping it entirely has zero consequence."
        )
    )
    broadcast: bool = Field(
        description=(
            "Sent to a vast list with nothing user-specific — newsletters, mass policy "
            "updates, product announcements, substacks, bulletins. Often worth a glance "
            "but the user is not personally the subject."
        )
    )
    obligation: bool = Field(
        description=(
            "The user actually has to do something concrete: pay a bill, respond, verify "
            "identity, renew, fix a failing payment, upload a document, show up. Someone "
            "or something is waiting, and there's a pending consequence for inaction."
        )
    )
    critical: bool = Field(
        description=(
            "Stakes are serious: large monetary amount, imminent account closure, service "
            "termination, debt collection, court notices, legal threats, final warnings, "
            "eviction, anything where inaction causes real damage to finances, housing, "
            "or legal standing. Lawyers could plausibly get involved."
        )
    )
    sender: str = Field(
        description=(
            "Brand / company / person who sent this, short form. "
            "Prefer the human-readable display name over the email address. "
            "For automated senders, the brand (e.g. 'Fly.io', 'Bank Frick', 'Microsoft')."
        )
    )
    amount: float | None = Field(
        default=None,
        description=(
            "Monetary amount if one is the main figure on the document "
            "(invoice total, receipt total, billing threshold). Null otherwise. "
            "Use the single most salient amount, not a subtotal."
        )
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code if `amount` is set; null otherwise."
    )
    due_date: str | None = Field(
        default=None,
        description=(
            "Payment or response deadline in YYYY-MM-DD format, if stated. "
            "Use the explicit due date on the document, not the sent date. "
            "Null if there is no deadline."
        )
    )
    clue: str = Field(
        max_length=42,
        description=(
            "Short natural-language phrase saying what this message is ABOUT. "
            "Must NOT repeat the sender name — the sender is shown separately. "
            "Must NOT restate category/intent. Write like a normal person writes. "
            "HARD LIMIT: 42 characters. "
            "Good examples: "
            "'CI failure for mbrock/foobar', "
            "'Rate hotel check-in', "
            "'Demand for office rent payment', "
            "'New browser added to Click to Pay', "
            "'Electricity tariff update', "
            "'Reading list: philosophy 2025'. "
            "Bad examples (sender-leaking or boilerplate): "
            "'payment reminder from Bite Latvija', "
            "'Tucker Carlson video update', "
            "'Newsletter from Robin Sloan'."
        )
    )
    code: str | None = Field(
        default=None,
        description=(
            "If this message is a one-time 2FA / OTP / verification code delivery, "
            "put the actual code digits here (e.g. '382915'). Null otherwise. "
            "Messages with a code are by definition not critical — they are transient "
            "confirmations, not demands on the user."
        )
    )
    error: str | None = Field(
        default=None,
        description=(
            "Set this ONLY if you cannot confidently classify the message — "
            "e.g. content is unreadable, empty after stripping, in a language you can't parse, "
            "or obviously corrupted. Explain the problem in one short sentence. "
            "Null when classification is fine."
        )
    )

# Load environment from ~/.env
load_dotenv(Path.home() / '.env')

# google-genai warns if both keys are set; blank out the duplicate
if os.environ.get('GEMINI_API_KEY') and os.environ.get('GOOGLE_API_KEY'):
    os.environ['GEMINI_API_KEY'] = ''


def get_api_key() -> str | None:
    """Read the Gemini API key. Accepts GOOGLE_API_KEY (preferred) or
    GEMINI_API_KEY (what the older README asked for)."""
    return os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')


console = Console()

SCRIPT_DIR = Path(__file__).parent
DB_FILE = SCRIPT_DIR / 'gmail.db'

MODEL_PRO = "gemini-3-pro-preview"
MODEL_FLASH = "gemini-3-flash-preview"
MODEL_FLASH_LITE = "gemini-3.1-flash-lite-preview"

MAX_WORKERS = 4  # Parallel workers for batch processing
MAX_RETRIES = 3  # Retry attempts for failed API calls
RETRY_DELAY = 2  # Seconds between retries


def get_prompt(gist: bool = False) -> str:
    """Generate prompt with current date."""
    today = datetime.now().strftime("%B %d, %Y")
    if gist:
        return (
            f"Today's date is {today}.\n\n"
            "Act as the executive assistant of a user who struggles with executive "
            "function and drowns in email. The purpose of this gist is triage: so the "
            "user can see — without having to wade through everything — which messages "
            "truly require them and which can be ignored. Telegraphic concision, please.\n\n"
            "Classify the message above by returning a JSON Gist following the schema. "
            "The four booleans are the key signal; set them independently and honestly. "
            "If you genuinely cannot classify the content, set `error` with a short "
            "explanation instead of guessing wildly."
        )

    return f"""Today's date is {today}.

Analyze this document. No markdown formatting.

If it's an invoice, bill, or receipt:
FROM: who sent it
TO: who it's addressed to
AMOUNT: total with currency
DUE: payment deadline (or PAID if already paid)
ITEMS: main charges, one per line
IBAN: payment reference if present
NOTES: warnings or overdue notices if any

If it's something else (email, boarding pass, contract, etc.), use the same plain-label style for its key details.

Be complete but ruthlessly concise. Plain text only, no markdown."""


def get_content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_cached_description(conn: sqlite3.Connection, content_hash: str, model: str) -> str | None:
    row = conn.execute(
        'SELECT extracted_json FROM invoice_extractions WHERE content_hash = ? AND model_name = ?',
        (content_hash, model)
    ).fetchone()
    return row[0] if row else None


def save_description(conn: sqlite3.Connection, content_hash: str, description: str, model: str, message_id: str | None = None):
    conn.execute('''
        INSERT OR REPLACE INTO invoice_extractions (content_hash, model_name, extracted_json, created_at, message_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (content_hash, model, description, datetime.now().isoformat(), message_id))
    conn.commit()


def model_label(model: str) -> str:
    if model == MODEL_FLASH_LITE:
        return "Flash Lite"
    if model == MODEL_FLASH:
        return "Flash"
    return "Pro"


def describe_document(
    content_hash: str,
    parts: list,
    label: str,
    model: str = MODEL_FLASH,
    gist: bool = False,
    quiet: bool = False,
    use_cache: bool = True,
    message_id: str | None = None,
    dry_run: bool = False,
    thinking: str = "medium",
) -> str | None:
    """Core: check cache, call Gemini, save result. Parts are already-constructed Gemini content parts.

    When gist=True the model is constrained to the Gist pydantic schema and the
    stored value is a JSON blob. When dry_run=True both cache read and write
    are skipped — used for previewing the prompt/schema on real samples.
    """
    cache_key = f"{model}:gist:v3" if gist else model

    conn = sqlite3.connect(DB_FILE)
    if use_cache and not dry_run:
        cached = get_cached_description(conn, content_hash, cache_key)
        if cached:
            if not quiet:
                console.print(f"[dim]Cached: {label}[/dim]")
            conn.close()
            return cached

    api_key = get_api_key()
    if not api_key:
        if not quiet:
            console.print("[red]No API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY)[/red]")
        conn.close()
        return None

    client = genai.Client(api_key=api_key)
    mode = f"{model_label(model)} gist" if gist else model_label(model)
    if not quiet:
        console.print(f"[blue]Analyzing {label} with Gemini {mode}{' [dry]' if dry_run else ''}...[/blue]")

    safety_off = [
        types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.OFF)
        for c in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
    ]

    config_kwargs: dict = {"safety_settings": safety_off}
    if gist:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_json_schema"] = Gist.model_json_schema()
    if thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts + [get_prompt(gist=gist)],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            description = response.text
            if description and not dry_run:
                save_description(conn, content_hash, description, cache_key, message_id=message_id)
            conn.close()
            return description
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                if not quiet:
                    console.print(f"[yellow]Retry {attempt + 1}/{MAX_RETRIES}: {e}[/yellow]")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                if not quiet:
                    console.print(f"[red]Failed after {MAX_RETRIES} attempts: {last_error}[/red]")

    conn.close()
    return None


def describe_pdf(pdf_path: str, api_key: str | None = None, use_cache: bool = True, model: str = MODEL_FLASH, quiet: bool = False, gist: bool = False, dry_run: bool = False, thinking: str = "medium") -> str | None:
    path = Path(pdf_path)
    if not path.exists():
        if not quiet:
            console.print(f"[red]File not found: {path}[/red]")
        return None
    pdf_data = path.read_bytes()
    content_hash = get_content_hash(pdf_data)
    parts = [types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")]
    return describe_document(content_hash, parts, path.name, model=model, gist=gist, quiet=quiet, use_cache=use_cache, dry_run=dry_run, thinking=thinking)


BODY_MAX_CHARS = 8000


def describe_email(message_id: str, subject: str, body: str, model: str = MODEL_FLASH, quiet: bool = False, gist: bool = False, dry_run: bool = False, from_addr: str | None = None, thinking: str = "medium") -> str | None:
    header = ""
    if from_addr:
        header += f"From: {from_addr}\n"
    if subject:
        header += f"Subject: {subject}\n"
    body = body or ""
    truncated = False
    if len(body) > BODY_MAX_CHARS:
        body = body[:BODY_MAX_CHARS]
        truncated = True
    text = (header + "\n" + body).strip()
    if truncated:
        text += "\n\n[… body truncated at 8000 chars]"
    if not text:
        return None
    content_hash = get_content_hash(text.encode())
    parts = [text]
    return describe_document(content_hash, parts, subject or message_id, model=model, gist=gist, quiet=quiet, message_id=message_id, dry_run=dry_run, thinking=thinking)


def describe_document_task(args: dict) -> tuple[str, str | None, str | None]:
    """Worker task for parallel processing."""
    try:
        dry = args.get('dry_run', False)
        thinking = args.get('thinking', 'low')
        if args['type'] == 'pdf':
            desc = describe_pdf(args['path'], model=args['model'], gist=args['gist'], quiet=True, dry_run=dry, thinking=thinking)
        else:
            desc = describe_email(args['message_id'], args['subject'], args['body'],
                                  model=args['model'], gist=args['gist'], quiet=True, dry_run=dry,
                                  from_addr=args.get('sender') or None, thinking=thinking)
        return (args['label'], desc, None)
    except Exception as e:
        return (args['label'], None, str(e))


# ── shared formatter for gist output (used by both parallel and serial paths) ──

def _status_of(g) -> tuple[str, int]:
    """(stripe_style, group). `stripe_style` is the rich style for a single
    colored space placed at the leftmost column — a priority stripe, not a
    full row background (which fought with low-contrast terminals).
    Priority: critical > obligation > broadcast > frivolous > routine."""
    if getattr(g, "critical", False):
        return ("on red",    3)
    if getattr(g, "obligation", False):
        return ("on yellow", 2)
    if getattr(g, "broadcast", False):
        return ("",          0)  # no stripe for bulk
    if getattr(g, "frivolous", False):
        return ("",          0)  # no stripe for fluf
    return ("on grey35",     1)   # routine


_EMOJI_TAG = {
    "CRIT": "🚨",
    "TODO": "📋",
    "BULK": "📰",
    "FLUF": "🎈",
    "    ": "📎",
}
_EMOJI_CAT = {
    "billing":     "💰",
    "accounts":    "🔐",
    "booking":     "🏨",
    "orders":      "📦",
    "newsletter":  "📰",
    "marketing":   "📢",
    "personal":    "👤",
    "institution": "🏛️",
    "other":       "❓",
}
_EMOJI_INTENT = {
    "failure":  "❌",
    "success":  "✅",
    "reminder": "⏰",
    "request":  "📨",
    "reply":    "💬",
    "warning":  "⚠️",
    "info":     "ℹ️",
}
_CAT_COLOR = {
    "billing":     "magenta",
    "accounts":    "red",
    "booking":     "cyan",
    "orders":      "cyan",
    "newsletter":  "dim",
    "marketing":   "dim",
    "personal":    "green",
    "institution": "blue",
    "other":       "white",
}
_INTENT_COLOR = {
    "failure":  "red",
    "success":  "green",
    "reminder": "yellow",
    "request":  "yellow",
    "reply":    "cyan",
    "warning":  "bold red",
    "info":     "dim",
}


_CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}


def _format_money(amount: float, currency: str | None) -> str:
    cur = (currency or "").upper()
    n = int(round(amount))
    sym = _CURRENCY_SYMBOL.get(cur)
    if sym:
        return f"{sym}{n}"
    return f"{n} {cur}" if cur else str(n)


def _base32_hash(mid: str) -> str:
    """Lowercase base32 of sha256(msg_id) with trailing '=' stripped."""
    import base64
    return base64.b32encode(hashlib.sha256(mid.encode()).digest()).decode("ascii").lower().rstrip("=")


def build_id_shortener(all_ids, safety: int = 1) -> dict:
    """Given an iterable of Gmail message IDs, return {full_id → short_id}
    using the shortest base32(sha256) prefix that is collision-free across
    the input set, plus `safety` extra chars as headroom for future growth."""
    hashes = {mid: _base32_hash(mid) for mid in all_ids if mid}
    n = 4
    while n < 52:
        shorts = {h[:n] for h in hashes.values()}
        if len(shorts) == len(hashes):
            break
        n += 1
    n = min(52, n + safety)
    return {mid: h[:n] for mid, h in hashes.items()}


def _short_date(s: str) -> str:
    """Current-year dates render as 'Mon Apr 20'; other years as '2025-04-16'."""
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(s)
    except Exception:
        return (s or '')[:10]
    if dt.year == datetime.now().year:
        return dt.strftime("%a %b %d")
    return dt.strftime("%Y-%m-%d")


def _date_sort_key(s: str) -> str:
    """Strict ISO date for sorting (the display format does not sort correctly)."""
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(s).isoformat()
    except Exception:
        return ""


def _relative_due(due_str: str | None) -> str:
    """Render a YYYY-MM-DD due date as a relative phrase."""
    if not due_str:
        return ""
    try:
        due = datetime.strptime(due_str, "%Y-%m-%d").date()
    except Exception:
        return f"due {due_str}"
    today = datetime.now().date()
    delta = (due - today).days
    if delta == 0:
        return "due today"
    if delta == 1:
        return "due tomorrow"
    if delta == -1:
        return "due yesterday"
    if delta < 0:
        return f"due {-delta} days ago"
    return f"due in {delta} days"


def print_gist_row(console_, raw: str | None, *, date: str = "", sender: str = "", label: str = "", msg_id: str = "", id_map: dict | None = None, emoji: bool = False) -> bool:
    """Returns True if this row was an error (no response or `error` field set)."""
    """Render one gist result (structured JSON or legacy free text) as two aligned lines.

    Line 1:  imp  date        category/subtype              sender            amount  due
    Line 2:      one_liner (or raw free text / error)
    """
    date_s = _short_date(date) if date else "          "
    sender_s = (sender or label or '').split('<')[0].strip().strip('"') or (sender or label or '')

    W_SENDER = 22
    W_CATSUB = 30
    INDENT2 = "       "

    if not raw:
        head = f"  [red]✗[/red]  [dim]{date_s}[/dim]  {sender_s[:W_SENDER].upper():<{W_SENDER}}  [red]NO RESPONSE[/red]"
        console_.print(head)
        console_.print(f"{INDENT2}[red]{label}[/red]")
        return True

    try:
        g = Gist.model_validate_json(raw)
    except Exception as e:
        head = f"  [red]✗[/red]  [dim]{date_s}[/dim]  {sender_s[:W_SENDER].upper():<{W_SENDER}}  [red]SCHEMA PARSE FAILED[/red]"
        console_.print(head)
        console_.print(f"{INDENT2}[red]{e}[/red] — raw: [dim]{raw.strip()[:200]}[/dim]")
        return True

    stripe_style, group = _status_of(g)
    sender_disp = (g.sender or sender_s or "")[:W_SENDER].ljust(W_SENDER)

    meta_bits = []
    if g.amount is not None:
        meta_bits.append(f"[yellow]{_format_money(g.amount, g.currency)}[/yellow]")
    due_phrase = _relative_due(g.due_date)
    if due_phrase:
        meta_bits.append(f"[yellow]{due_phrase}[/yellow]")
    if getattr(g, "code", None):
        meta_bits.append(f"[bold magenta]{g.code}[/bold magenta]")
    meta_inline = (" ".join(meta_bits) + "  ") if meta_bits else ""

    if g.error:
        head = (
            f"  [red]⚠⚠⚠⚠[/red]  [dim]{date_s}[/dim]  "
            f"{sender_disp}  [red]{cat_abbr} {intent_abbr}[/red]"
        )
        console_.print(head)
        console_.print(f"{INDENT2}[red]{g.error}[/red]  [dim]({g.clue})[/dim]")
        return True

    sender_pad = (sender_disp.rstrip()[:20]).ljust(20)
    stripe = f"[{stripe_style}] [/]" if stripe_style else " "
    date_pad = date_s.ljust(10)
    if id_map and msg_id in id_map:
        short_id = id_map[msg_id]
    elif msg_id:
        short_id = _base32_hash(msg_id)[:7]
    else:
        short_id = "       "
    body = f"[bold]{sender_pad}[/bold]  {meta_inline}{g.clue or ''}"
    kwargs = {"no_wrap": True, "overflow": "ellipsis"}
    if group <= 1:
        console_.print(f"{stripe} [dim]{short_id}  {date_pad}  {body}[/dim]", **kwargs)
    else:
        console_.print(f"{stripe} [cyan]{short_id}[/cyan]  {date_pad}  {body}", **kwargs)
    return False


# Keep old name as alias for callers that import it directly
def describe_pdf_task(args: tuple) -> tuple[str, str | None, str | None]:
    path, model, gist = args
    return describe_document_task({'type': 'pdf', 'path': path, 'model': model, 'gist': gist, 'label': Path(path).name})


def describe_documents_parallel(tasks: list[dict], workers: int = MAX_WORKERS, emoji: bool = False) -> tuple[list[tuple[dict, str | None]], int]:
    """Run tasks in parallel. Shows ONLY a progress bar while working (cleared
    on exit); caller prints results afterwards (see print_gist_sorted)."""
    results: list[tuple[dict, str | None]] = []
    error_count = 0
    gist = tasks[0]['gist'] if tasks else False

    # Fail fast: without a key, every worker returns None silently (quiet=True
    # suppresses the per-task error), so you'd otherwise see "N/N errors" with
    # no indication why.
    if tasks and not get_api_key():
        raise SystemExit(
            "No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
            "in ~/.env and re-run."
        )

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        ptask = progress.add_task("", total=len(tasks))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(describe_document_task, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                label, description, error = future.result()
                if error:
                    error_count += 1
                elif not description:
                    error_count += 1
                elif gist:
                    # Pre-parse for error counting without printing.
                    try:
                        g = Gist.model_validate_json(description)
                        if g.error:
                            error_count += 1
                    except Exception:
                        error_count += 1
                results.append((task, description))
                progress.advance(ptask)

    return results, error_count


def _week_bucket(s: str) -> tuple[int, str]:
    """Returns (week_index, label) where week 0 is the current calendar week.
    Week boundaries are Monday-based. Older weeks increment the index."""
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(s).date()
    except Exception:
        return (9999, "sometime")
    today = datetime.now().date()
    mon_today = today - timedelta(days=today.weekday())
    mon_that = dt - timedelta(days=dt.weekday())
    weeks = (mon_today - mon_that).days // 7
    if weeks < 0:
        return (-1, "upcoming")
    if weeks == 0:
        return (0, "this week")
    if weeks == 1:
        return (1, "last week")
    if weeks < 8:
        return (weeks, f"{weeks} weeks ago")
    months = weeks // 4
    return (weeks, f"{months} month" + ("s" if months != 1 else "") + " ago")


def print_gist_week(console_, week_label: str, results: list[tuple[dict, str | None]], *, emoji: bool = False, show_noise: bool = False, id_map: dict | None = None) -> int:
    """Print a single week's rows, already sorted priority→sender→date.
    Returns number of rows printed."""
    rows: list[tuple[int, str, str, dict, str]] = []
    for task, description in results:
        if not description:
            continue
        try:
            g = Gist.model_validate_json(description)
        except Exception:
            continue
        _, group = _status_of(g)
        if group == 0 and not show_noise:
            continue
        date_key = _date_sort_key(task.get('date', ''))
        sender_key = (g.sender or task.get('sender', '') or '').strip().lower()
        rows.append((group, sender_key, date_key, task, description))

    if not rows:
        return 0

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    console_.print()
    console_.print(f"[bold dim]— {week_label} —[/bold dim]")
    for _, _, _, task, description in rows:
        print_gist_row(
            console_, description,
            date=task.get('date', ''),
            sender=task.get('sender', ''),
            label=task.get('label', ''),
            msg_id=task.get('message_id') or task.get('msg_id') or '',
            id_map=id_map,
            emoji=emoji,
        )
    return len(rows)


def describe_pdfs_parallel(pdf_paths: list[tuple[str, str]], model: str = MODEL_FLASH, workers: int = MAX_WORKERS, gist: bool = False) -> list[tuple[str, str | None]]:
    tasks = [{'type': 'pdf', 'path': path, 'label': Path(path).name, 'model': model, 'gist': gist} for path, _ in pdf_paths]
    return describe_documents_parallel(tasks, workers=workers)


def display_description(description: str, filename: str = ""):
    """Display the description in a nice panel."""
    title = filename if filename else "Document"
    console.print(Panel(Markdown(description), title=title, border_style="blue"))


def main(args: list[str] | None = None):
    """Main entry point."""
    import sys
    args = args if args is not None else sys.argv[1:]

    # Parse flags
    use_pro = '--pro' in args
    use_cache = '--no-cache' not in args
    parallel = '--parallel' in args
    model = MODEL_PRO if use_pro else MODEL_FLASH

    # Parse --workers N
    workers = MAX_WORKERS
    for i, arg in enumerate(args):
        if arg == '--workers' and i + 1 < len(args):
            try:
                workers = int(args[i + 1])
            except ValueError:
                pass

    # Remove flags from args
    args = [a for a in args if a not in ('--pro', '--no-cache', '--all', '--parallel') and not a.startswith('--workers')]

    if not args:
        # Process unprocessed PDFs from database
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute('''
            SELECT a.local_path, a.content_hash, a.filename
            FROM attachments a
            LEFT JOIN invoice_extractions e ON a.content_hash = e.content_hash
            WHERE e.id IS NULL
            ORDER BY a.downloaded_at DESC
            LIMIT 10
        ''').fetchall()
        conn.close()

        if not rows:
            console.print("[yellow]No unprocessed PDFs. Run: uv run sync.py sync --with-pdfs[/yellow]")
            return

        console.print(f"[blue]Found {len(rows)} unprocessed PDFs[/blue]\n")

        if parallel:
            pdf_paths = [(path, filename) for path, _, filename in rows if Path(path).exists()]
            results = describe_pdfs_parallel(pdf_paths, model=model, workers=workers)
        else:
            for path, content_hash, filename in rows:
                if not Path(path).exists():
                    console.print(f"[dim]Skipping missing file: {filename}[/dim]")
                    continue

                description = describe_pdf(path, model=model)
                if description:
                    display_description(description, filename)
                    console.print()

    elif '--all' in sys.argv:
        # Process all PDFs (re-describe)
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute('SELECT local_path, filename FROM attachments ORDER BY downloaded_at DESC').fetchall()
        conn.close()

        if parallel:
            pdf_paths = [(path, filename) for path, filename in rows if Path(path).exists()]
            results = describe_pdfs_parallel(pdf_paths, model=model, workers=workers)
        else:
            for path, filename in rows:
                if not Path(path).exists():
                    continue
                description = describe_pdf(path, use_cache=False, model=model)
                if description:
                    display_description(description, filename)
                    console.print()

    else:
        # Process specific file
        pdf_path = args[0]
        description = describe_pdf(pdf_path, use_cache=use_cache, model=model)
        if description:
            display_description(description, Path(pdf_path).name)


if __name__ == '__main__':
    main()
