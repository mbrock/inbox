"""Pydantic models for invoice data extraction."""

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, StringConstraints

# Strip whitespace from all strings
StrippedStr = Annotated[str, StringConstraints(strip_whitespace=True)]


class LineItem(BaseModel):
    """A single line item from an invoice."""

    icon: StrippedStr
    """Emoji icon representing the item type."""

    kind: StrippedStr
    """Item type in 2-3 words (e.g., 'Power Tool', 'Building Material')."""

    brand: StrippedStr | None = None
    """Manufacturer or brand name if specified."""

    size: StrippedStr | None = None
    """Dimensions or size specification (e.g., '50L', '45x145x2400mm')."""

    attributes: list[StrippedStr] = []
    """Concise specs: material, color, standard, count."""

    quantity: Decimal
    """Numeric quantity."""

    unit: StrippedStr | None = None
    """Unit of measure (pcs, m, kg, m2, m3)."""

    unit_price: Decimal | None = None
    """Price per unit."""

    subtotal: Decimal
    """Total line item cost."""


class InvoiceData(BaseModel):
    """Structured data extracted from an invoice PDF."""

    iban: list[StrippedStr] = []
    """International Bank Account Numbers found in the document."""

    company_name: StrippedStr
    """Vendor/seller company name."""

    customer_name: StrippedStr | None = None
    """Recipient/buyer name if present."""

    amount: Decimal
    """Total invoice amount."""

    currency: StrippedStr
    """Currency code (EUR, USD, SEK, etc.)."""

    payment_reference: StrippedStr
    """Order number, invoice number, or payment reference."""

    line_items: list[LineItem] = []
    """Itemized charges if available."""

    invoice_date: StrippedStr | None = None
    """Invoice date if present."""

    due_date: StrippedStr | None = None
    """Payment due date if present."""
