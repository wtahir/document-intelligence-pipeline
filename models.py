"""
Pydantic models for validated LLM extraction output.

Each model defines the exact schema expected from GPT-4o when classifying
and extracting insurance documents. Validation catches bad LLM output
immediately — before it silently corrupts downstream stages.

The pipeline processes three damage types (water, storm, glass) and three
document categories (claim_email, invoice, photo_documentation).
"""

from typing import Optional
from pydantic import BaseModel, Field


class ClaimEmail(BaseModel):
    """Schema for claim notification emails reporting damage."""
    document_type: str = Field(default="claim_email")
    language: str = Field(description="ISO 639-1 code, e.g. 'de', 'en'")
    claim_number: Optional[str] = None
    date: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    claimant_name: Optional[str] = None
    policy_number: Optional[str] = None
    damage_type: Optional[str] = Field(default=None, description="water | storm | glass")
    damaged_object: Optional[str] = None
    damage_date: Optional[str] = None
    damage_description: Optional[str] = None
    summary_en: str = Field(description="English summary, 2-3 sentences")
    urgency: str = Field(default="normal", description="low | normal | high")
    confidence: float = Field(ge=0.0, le=1.0)


class InvoiceDocument(BaseModel):
    """Schema for repair invoices with claimed amounts."""
    document_type: str = Field(default="invoice")
    language: str
    invoice_number: Optional[str] = None
    claim_number: Optional[str] = None
    claimant_name: Optional[str] = None
    policy_number: Optional[str] = None
    damage_type: Optional[str] = Field(default=None, description="water | storm | glass")
    damaged_object: Optional[str] = None
    vendor: Optional[str] = None
    invoice_date: Optional[str] = None
    subtotal_eur: Optional[float] = None
    tax_eur: Optional[float] = None
    total_amount_eur: Optional[float] = None
    line_items: list[str] = Field(default_factory=list, description="List of service descriptions")
    summary_en: str
    confidence: float = Field(ge=0.0, le=1.0)


class PhotoDocumentation(BaseModel):
    """Schema for damage photo documentation reports."""
    document_type: str = Field(default="photo_documentation")
    language: str
    claim_number: Optional[str] = None
    claimant_name: Optional[str] = None
    damage_type: Optional[str] = Field(default=None, description="water | storm | glass")
    damaged_object: Optional[str] = None
    photo_date: Optional[str] = None
    damage_severity: Optional[str] = Field(default=None, description="minor | moderate | severe | total_loss")
    repair_recommendation: Optional[str] = None
    summary_en: str
    confidence: float = Field(ge=0.0, le=1.0)


# --- Kept for backward compatibility in tests ---

class ClaimCommunication(BaseModel):
    """Legacy schema — alias for ClaimEmail for backward compatibility."""
    document_type: str = Field(default="claim_communication")
    language: str = Field(default="en", description="ISO 639-1 code")
    claim_number: Optional[str] = None
    date: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    subject: Optional[str] = None
    summary_en: str = Field(default="", description="English summary")
    attachments_mentioned: list[str] = Field(default_factory=list)
    action_required: Optional[str] = None
    urgency: str = Field(default="normal")
    confidence: float = Field(ge=0.0, le=1.0)


class PolicyDocument(BaseModel):
    """Schema for insurance policy documents."""
    document_type: str = Field(default="policy_document")
    language: str
    policy_number: Optional[str] = None
    policyholder_name: Optional[str] = None
    coverage_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    premium_amount: Optional[str] = None
    summary_en: str
    confidence: float = Field(ge=0.0, le=1.0)


class UnknownDocument(BaseModel):
    """Fallback schema for unclassifiable documents."""
    document_type: str = Field(default="unknown")
    language: str = "unknown"
    summary_en: str = "Could not summarize."
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
