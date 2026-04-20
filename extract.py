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


class PromptBlockedError(RuntimeError):
    """Gemini refused the request at the safety pre-filter (PROHIBITED_CONTENT
    etc.). Separate from generic RuntimeError because retrying another model
    rarely helps — Google's pre-filter is broadly consistent across tiers —
    and because the triage pipeline would rather bucket these as newsletter
    noise than keep re-queueing them forever. See describe_email for the
    auto-classify fallback."""


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
            # Cached rows whose Gist has `.error` set represent a previous
            # failed attempt. Don't short-circuit on them — let the live
            # call run again so that e.g. a safety-block can now fall
            # through to describe_email's newsletter-fallback path. A
            # successful retry will overwrite the error row via the
            # INSERT OR REPLACE in save_description.
            stale_error = False
            if gist:
                try:
                    if Gist.model_validate_json(cached).error:
                        stale_error = True
                except Exception:
                    pass
            if not stale_error:
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

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts + [get_prompt(gist=gist)],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            description = response.text
            if description:
                if not dry_run:
                    save_description(conn, content_hash, description, cache_key, message_id=message_id)
                conn.close()
                return description

            # Empty response.text: the call succeeded but Gemini returned no
            # output. Distinguish "hard-blocked by safety" from other oddities
            # (MAX_TOKENS, RECITATION, etc.) so callers can treat safety blocks
            # as a real classification result (usually: newsletter noise) and
            # other empties as genuine failures worth retrying.
            pf = getattr(response, "prompt_feedback", None)
            candidate_blocked = False
            candidate_block_bits: list[str] = []
            for cand in getattr(response, "candidates", None) or []:
                sr = getattr(cand, "safety_ratings", None) or []
                blocked = [r for r in sr if getattr(r, "blocked", False)]
                if blocked:
                    candidate_blocked = True
                    cats = ",".join(str(getattr(r, "category", "?")) for r in blocked)
                    candidate_block_bits.append(f"safety blocked: {cats}")
            if pf is not None and getattr(pf, "block_reason", None):
                raise PromptBlockedError(f"prompt blocked: {pf.block_reason}")
            if candidate_blocked:
                raise PromptBlockedError("; ".join(candidate_block_bits))

            reason_bits: list[str] = []
            for cand in getattr(response, "candidates", None) or []:
                fr = getattr(cand, "finish_reason", None)
                if fr and str(fr).rsplit(".", 1)[-1] != "STOP":
                    reason_bits.append(f"finish_reason={fr}")
            reason = "; ".join(reason_bits) or "empty response.text"
            raise RuntimeError(f"Gemini returned no text ({reason})")
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
    # Re-raise so the parallel worker captures a real error string instead of
    # silently returning None (which the UI renders as the uninformative
    # "NO RESPONSE").
    raise last_error if last_error else RuntimeError("Gemini call failed with no captured exception")


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


def build_email_text(from_addr: str | None, subject: str | None, body: str | None) -> str:
    """Canonical serialization of an email for the Gemini classifier.

    Both `describe_email` (live) and the sync-side body_hash pre-filter must
    use this exact string so that the content_hash cache lookup and the
    NOT EXISTS filter agree on what "this email" is."""
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
    return text


def email_content_hash(from_addr: str | None, subject: str | None, body: str | None) -> str | None:
    """sha256 of the Gemini input text. None for empty emails."""
    text = build_email_text(from_addr, subject, body)
    if not text:
        return None
    return get_content_hash(text.encode())


def task_content_hash(task: dict) -> str | None:
    """content_hash that describe_{email,pdf} would use as its cache key."""
    if task.get('type') == 'pdf':
        path = task.get('path')
        if not path or not Path(path).exists():
            return None
        return get_content_hash(Path(path).read_bytes())
    return email_content_hash(
        task.get('sender') or None,
        task.get('subject') or None,
        task.get('body') or None,
    )


def make_error_gist_json(error_msg: str, task: dict) -> str:
    """Synthesize a Gist JSON blob with `error` set so a classifier failure
    can be cached in `invoice_extractions` and picked up by `--retry-failed`
    without silently re-burning tokens on every `analyze` run.

    The required non-error fields are filled with inert defaults (broadcast
    noise in the `other` category) so that if a stray renderer ever reads
    the row without honoring `.error`, it shows up as muted, not as a
    fake-confident classification."""
    sender = (task.get('sender') or '').split('<')[0].strip().strip('"')[:40] or 'unknown'
    label = task.get('label') or task.get('subject') or task.get('message_id') or 'n/a'
    g = Gist(
        category='other', intent='info',
        frivolous=True, broadcast=False, obligation=False, critical=False,
        sender=sender,
        clue=str(label)[:42] or 'n/a',
        error=error_msg[:500],
    )
    return g.model_dump_json()


def make_blocked_gist_json(from_addr: str | None, subject: str | None, fallback_label: str | None = None) -> str:
    """Gist for an email whose content Gemini refused at the safety pre-filter.

    The triage pipeline is for finding bills/obligations; if Google won't let
    us inspect a message, it's almost always a Substack-style newsletter with
    racy content, not an actionable invoice. Bucket as newsletter/broadcast/
    frivolous so it's hidden by the default noise filter, and keep `error=None`
    so `--retry-failed` doesn't pick it back up (retrying won't help — the
    block is consistent across Gemini tiers). Real non-newsletter edge cases
    can still be seen with `--show-noise` or pulled from the sender directly."""
    sender = (from_addr or '').split('<')[0].strip().strip('"')[:40] or 'unknown'
    clue_src = subject or fallback_label or 'blocked content'
    g = Gist(
        category='newsletter', intent='info',
        frivolous=True, broadcast=True, obligation=False, critical=False,
        sender=sender,
        clue=str(clue_src)[:42] or 'blocked content',
        error=None,
    )
    return g.model_dump_json()


def describe_email(message_id: str, subject: str, body: str, model: str = MODEL_FLASH, quiet: bool = False, gist: bool = False, dry_run: bool = False, from_addr: str | None = None, thinking: str = "medium") -> str | None:
    text = build_email_text(from_addr, subject, body)
    if not text:
        return None
    content_hash = get_content_hash(text.encode())
    parts = [text]
    try:
        return describe_document(content_hash, parts, subject or message_id, model=model, gist=gist, quiet=quiet, message_id=message_id, dry_run=dry_run, thinking=thinking)
    except PromptBlockedError as e:
        if not gist:
            raise  # non-gist flows (raw description) have no sensible fallback
        if not quiet:
            console.print(f"[dim yellow]{subject or message_id}: {e} — auto-classifying as newsletter noise[/dim yellow]")
        desc_json = make_blocked_gist_json(from_addr, subject, message_id)
        if not dry_run:
            cache_key = f"{model}:gist:v3"
            conn = sqlite3.connect(DB_FILE)
            try:
                save_description(conn, content_hash, desc_json, cache_key, message_id=message_id)
            finally:
                conn.close()
        return desc_json


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


def print_gist_row(console_, raw: str | None, *, date: str = "", sender: str = "", label: str = "", msg_id: str = "", id_map: dict | None = None, emoji: bool = False, error: str | None = None) -> bool:
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
        reason = "TASK FAILED" if error else "NO RESPONSE"
        head = f"  [red]✗[/red]  [dim]{date_s}[/dim]  {sender_s[:W_SENDER].upper():<{W_SENDER}}  [red]{reason}[/red]"
        console_.print(head)
        console_.print(f"{INDENT2}[red]{label}[/red]")
        if error:
            console_.print(f"{INDENT2}[red]{error}[/red]")
        return True

    try:
        g = Gist.model_validate_json(raw)
    except Exception as e:
        head = f"  [red]✗[/red]  [dim]{date_s}[/dim]  {sender_s[:W_SENDER].upper():<{W_SENDER}}  [red]SCHEMA PARSE FAILED[/red]"
        console_.print(head)
        console_.print(f"{INDENT2}[red]{label}[/red]")
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
            f"  [red]⚠[/red]  [dim]{date_s}[/dim]  "
            f"{sender_disp}  [red]GIST ERROR[/red]"
        )
        console_.print(head)
        console_.print(f"{INDENT2}[red]{label}[/red]")
        console_.print(f"{INDENT2}[red]{g.error}[/red]  [dim]({g.clue or ''})[/dim]")
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


def print_gist_week(console_, week_label: str, results: list, *, emoji: bool = False, show_noise: bool = False, id_map: dict | None = None) -> int:
    """Print a single week's rows, already sorted priority→sender→date.

    `results` items may be (task, description) or (task, description, error).
    Errored/unparseable rows are surfaced at the top of the week with their
    exception message, so `analyze` doesn't silently swallow failures.

    Returns number of rows printed."""
    rows: list[tuple[int, str, str, dict, str | None, str | None]] = []
    for item in results:
        if len(item) == 3:
            task, description, error = item
        else:
            task, description = item
            error = None

        group: int
        sender_key: str
        if not description:
            # Surface task failure (exception, empty response) at the top of
            # the week with priority group -2 so they sort above real rows.
            group = -2
            sender_key = (task.get('sender', '') or '').strip().lower()
        else:
            try:
                g = Gist.model_validate_json(description)
            except Exception:
                group = -1  # parse failures right below task failures
                sender_key = (task.get('sender', '') or '').strip().lower()
                rows.append((group, sender_key, _date_sort_key(task.get('date', '')),
                             task, description, error))
                continue
            if g.error:
                # Cached classifier failure. Treat like a fresh task failure
                # (group -1) so it surfaces at the top of the week regardless
                # of noise filtering — erroring is not a "broadcast" even if
                # make_error_gist_json's placeholder booleans would say so.
                group = -1
                sender_key = (g.sender or task.get('sender', '') or '').strip().lower()
                rows.append((group, sender_key, _date_sort_key(task.get('date', '')),
                             task, description, error))
                continue
            _, group = _status_of(g)
            if group == 0 and not show_noise:
                continue
            sender_key = (g.sender or task.get('sender', '') or '').strip().lower()

        date_key = _date_sort_key(task.get('date', ''))
        rows.append((group, sender_key, date_key, task, description, error))

    if not rows:
        return 0

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    console_.print()
    console_.print(f"[bold dim]— {week_label} —[/bold dim]")
    for _, _, _, task, description, error in rows:
        print_gist_row(
            console_, description,
            date=task.get('date', ''),
            sender=task.get('sender', ''),
            label=task.get('label', ''),
            msg_id=task.get('message_id') or task.get('msg_id') or '',
            id_map=id_map,
            emoji=emoji,
            error=error,
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
    show_all = '--all' in args
    model = MODEL_PRO if use_pro else MODEL_FLASH

    # Parse --workers N while preserving positional args.
    workers = MAX_WORKERS
    cleaned_args: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--workers' and i + 1 < len(args):
            try:
                workers = int(args[i + 1])
                i += 2
                continue
            except ValueError:
                cleaned_args.append(arg)
                i += 1
                continue
        if arg in ('--pro', '--no-cache', '--all', '--parallel'):
            i += 1
            continue
        cleaned_args.append(arg)
        i += 1
    args = cleaned_args

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

                try:
                    description = describe_pdf(path, model=model)
                except Exception as e:
                    console.print(f"[red]✗ {filename}: {e}[/red]")
                    continue
                if description:
                    display_description(description, filename)
                    console.print()

    elif show_all:
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
                try:
                    description = describe_pdf(path, use_cache=False, model=model)
                except Exception as e:
                    console.print(f"[red]✗ {filename}: {e}[/red]")
                    continue
                if description:
                    display_description(description, filename)
                    console.print()

    else:
        # Process specific file
        pdf_path = args[0]
        try:
            description = describe_pdf(pdf_path, use_cache=use_cache, model=model)
        except Exception as e:
            console.print(f"[red]✗ {pdf_path}: {e}[/red]")
            return
        if description:
            display_description(description, Path(pdf_path).name)


if __name__ == '__main__':
    main()
