"""
Unit tests for the Insurance Document Intelligence Pipeline.

Tests cover core logic (chunking, validation, metadata building)
without requiring API keys or external services.

Run:  python -m pytest tests/ -v
"""

import pytest
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Model validation tests ──────────────────────────────────

class TestPydanticModels:
    """Validate that Pydantic models enforce schema correctly."""

    def test_claim_email_valid(self):
        from models import ClaimEmail

        data = {
            "document_type": "claim_email",
            "language": "en",
            "claim_number": "WD-2024-12345",
            "date": "2024-06-15",
            "sender": "test@example.com",
            "recipient": "claims@insurance.com",
            "claimant_name": "Thomas Mueller",
            "policy_number": "POL-100000",
            "damage_type": "water",
            "damaged_object": "kitchen ceiling",
            "damage_date": "2024-06-10",
            "damage_description": "Burst pipe caused flooding",
            "summary_en": "Water damage claim for kitchen ceiling.",
            "urgency": "high",
            "confidence": 0.92,
        }
        claim = ClaimEmail(**data)
        assert claim.damage_type == "water"
        assert claim.damaged_object == "kitchen ceiling"
        assert claim.confidence == 0.92

    def test_claim_email_optional_fields(self):
        from models import ClaimEmail

        data = {
            "document_type": "claim_email",
            "language": "en",
            "summary_en": "Minimal claim document.",
            "urgency": "normal",
            "confidence": 0.5,
        }
        claim = ClaimEmail(**data)
        assert claim.claim_number is None
        assert claim.damage_type is None
        assert claim.damaged_object is None

    def test_invoice_document_valid(self):
        from models import InvoiceDocument

        data = {
            "document_type": "invoice",
            "language": "en",
            "invoice_number": "INV-10001",
            "claim_number": "WD-2024-12345",
            "claimant_name": "Thomas Mueller",
            "policy_number": "POL-100000",
            "damage_type": "water",
            "damaged_object": "kitchen ceiling",
            "vendor": "Rohrfix GmbH",
            "invoice_date": "2024-07-01",
            "subtotal_eur": 3500.00,
            "tax_eur": 665.00,
            "total_amount_eur": 4165.00,
            "line_items": ["Pipe repair", "Ceiling replacement"],
            "summary_en": "Invoice for water damage repair.",
            "confidence": 0.88,
        }
        inv = InvoiceDocument(**data)
        assert inv.total_amount_eur == 4165.00
        assert inv.damage_type == "water"
        assert len(inv.line_items) == 2

    def test_photo_documentation_valid(self):
        from models import PhotoDocumentation

        data = {
            "document_type": "photo_documentation",
            "language": "en",
            "claim_number": "SD-2024-11111",
            "claimant_name": "Maria Schmidt",
            "damage_type": "storm",
            "damaged_object": "roof tiles",
            "photo_date": "2024-08-15",
            "damage_severity": "severe",
            "repair_recommendation": "Complete replacement required",
            "summary_en": "Storm damage to roof tiles documented.",
            "confidence": 0.85,
        }
        photo = PhotoDocumentation(**data)
        assert photo.damage_severity == "severe"
        assert photo.damaged_object == "roof tiles"

    def test_unknown_document_defaults(self):
        from models import UnknownDocument

        doc = UnknownDocument()
        assert doc.document_type == "unknown"
        assert doc.confidence == 0.0

    def test_confidence_out_of_range(self):
        from models import ClaimEmail
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClaimEmail(
                document_type="claim_email",
                language="en",
                summary_en="Test.",
                urgency="normal",
                confidence=1.5,  # Out of range
            )

    def test_legacy_claim_communication(self):
        """Backward compatibility: ClaimCommunication still works."""
        from models import ClaimCommunication

        data = {
            "document_type": "claim_communication",
            "language": "de",
            "summary_en": "Legacy format test.",
            "urgency": "normal",
            "confidence": 0.7,
        }
        claim = ClaimCommunication(**data)
        assert claim.confidence == 0.7


# ─── Chunking tests ──────────────────────────────────────────

class TestChunking:
    """Validate text chunking logic."""

    def test_short_document_single_chunk(self):
        from stage3_chunking import chunk_document

        doc = {
            "file_name": "short.pdf",
            "original_content": "This is a short document.",
            "status": "success",
        }
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["is_single_chunk"] is True

    def test_long_document_multiple_chunks(self):
        from stage3_chunking import chunk_text
        from config import CHUNK_SIZE, CHUNK_OVERLAP

        text = "Word " * 500  # ~2500 chars
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        assert len(chunks) > 1

    def test_chunk_text_respects_min_size(self):
        from stage3_chunking import chunk_text

        text = "Hello world."
        chunks = chunk_text(text, chunk_size=800, overlap=150)
        assert len(chunks) == 0

    def test_chunks_have_metadata(self):
        from stage3_chunking import build_chunk_record

        doc = {
            "file_name": "test.pdf",
            "file_path": "/data/pdfs/test.pdf",
            "document_type": "claim_email",
            "language": "en",
            "claim_number": "WD-2024-12345",
            "claimant_name": "Thomas Mueller",
            "damage_type": "water",
            "damaged_object": "kitchen ceiling",
            "urgency": "high",
            "confidence": 0.9,
            "summary_en": "Test summary.",
        }
        record = build_chunk_record("Some chunk text here", 0, 3, doc)
        assert record["chunk_id"] == "test.pdf_chunk_0"
        assert record["damage_type"] == "water"
        assert record["damaged_object"] == "kitchen ceiling"
        assert record["claimant_name"] == "Thomas Mueller"
        assert record["total_chunks"] == 3

    def test_empty_content_produces_no_chunks(self):
        from stage3_chunking import chunk_document

        doc = {
            "file_name": "empty.pdf",
            "original_content": "",
        }
        chunks = chunk_document(doc)
        assert len(chunks) == 0

    def test_overlap_creates_redundancy(self):
        from stage3_chunking import chunk_text

        text = "A" * 2000
        chunks = chunk_text(text, chunk_size=800, overlap=150)
        assert len(chunks) >= 2


# ─── Metadata building tests ─────────────────────────────────

class TestMetadata:
    """Validate ChromaDB metadata building."""

    def test_build_metadata_no_none_values(self):
        from stage4_embedding import build_metadata

        chunk = {
            "file_name": "test.pdf",
            "document_type": "claim_email",
            "claim_number": None,
            "claimant_name": None,
            "policy_number": None,
            "damage_type": "water",
            "damaged_object": None,
            "date": None,
            "sender": "test@email.com",
            "urgency": "high",
            "language": "en",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_single_chunk": True,
            "summary_en": "Test",
            "total_amount_eur": None,
            "vendor": None,
            "damage_severity": None,
        }
        meta = build_metadata(chunk)

        for key, value in meta.items():
            assert value is not None, f"Metadata key '{key}' has None value"

    def test_build_metadata_correct_types(self):
        from stage4_embedding import build_metadata

        chunk = {
            "file_name": "test.pdf",
            "document_type": "invoice",
            "claim_number": "WD-2024-12345",
            "claimant_name": "Thomas Mueller",
            "policy_number": "POL-100000",
            "damage_type": "water",
            "damaged_object": "kitchen ceiling",
            "date": "2024-01-01",
            "sender": "test@email.com",
            "urgency": "high",
            "language": "en",
            "chunk_index": 2,
            "total_chunks": 5,
            "is_single_chunk": False,
            "summary_en": "Test",
            "total_amount_eur": "4165.00",
            "vendor": "Rohrfix GmbH",
            "damage_severity": None,
        }
        meta = build_metadata(chunk)
        assert isinstance(meta["chunk_index"], int)
        assert isinstance(meta["total_chunks"], int)
        assert isinstance(meta["is_single_chunk"], bool)
        assert meta["damage_type"] == "water"


# ─── Extraction validation tests ─────────────────────────────

class TestExtractionValidation:
    """Validate the extraction routing logic."""

    def test_validate_claim_email(self):
        from stage2_extraction import validate_extraction

        raw = {
            "document_type": "claim_email",
            "language": "en",
            "claim_number": "WD-2024-12345",
            "claimant_name": "Thomas Mueller",
            "damage_type": "water",
            "damaged_object": "kitchen ceiling",
            "summary_en": "Water damage claim.",
            "urgency": "high",
            "confidence": 0.9,
        }
        model, doc_type = validate_extraction(raw)
        assert doc_type == "claim_email"
        assert model.damage_type == "water"

    def test_validate_unknown_fallback(self):
        from stage2_extraction import validate_extraction

        raw = {
            "document_type": "something_weird",
            "language": "en",
            "summary_en": "Cannot classify.",
            "confidence": 0.1,
        }
        model, doc_type = validate_extraction(raw)
        assert doc_type == "unknown"
        assert model.document_type == "unknown"

    def test_validate_invoice(self):
        from stage2_extraction import validate_extraction

        raw = {
            "document_type": "invoice",
            "language": "en",
            "invoice_number": "INV-10001",
            "claim_number": "WD-2024-12345",
            "total_amount_eur": 4165.00,
            "damage_type": "water",
            "summary_en": "Repair invoice.",
            "confidence": 0.85,
        }
        model, doc_type = validate_extraction(raw)
        assert doc_type == "invoice"
        assert model.total_amount_eur == 4165.00

    def test_validate_photo_documentation(self):
        from stage2_extraction import validate_extraction

        raw = {
            "document_type": "photo_documentation",
            "language": "en",
            "claim_number": "SD-2024-11111",
            "damage_type": "storm",
            "damaged_object": "roof tiles",
            "damage_severity": "severe",
            "summary_en": "Storm damage photos.",
            "confidence": 0.9,
        }
        model, doc_type = validate_extraction(raw)
        assert doc_type == "photo_documentation"
        assert model.damage_severity == "severe"


# ─── Policy check tests ──────────────────────────────────────

class TestPolicyCheck:
    """Validate the deterministic policy check logic."""

    def test_find_policy_exact_match(self):
        from stage5_retrieval import find_policy_for_claimant

        policies = [
            {"policyholder_name": "Thomas Mueller", "policy_number": "POL-100000"},
            {"policyholder_name": "Maria Schmidt", "policy_number": "POL-101111"},
        ]
        result = find_policy_for_claimant("Thomas Mueller", policies)
        assert result is not None
        assert result["policy_number"] == "POL-100000"

    def test_find_policy_case_insensitive(self):
        from stage5_retrieval import find_policy_for_claimant

        policies = [
            {"policyholder_name": "Thomas Mueller", "policy_number": "POL-100000"},
        ]
        result = find_policy_for_claimant("thomas mueller", policies)
        assert result is not None

    def test_find_policy_not_found(self):
        from stage5_retrieval import find_policy_for_claimant

        policies = [
            {"policyholder_name": "Thomas Mueller", "policy_number": "POL-100000"},
        ]
        result = find_policy_for_claimant("Unknown Person", policies)
        assert result is None

    def test_check_coverage_covered(self):
        from stage5_retrieval import check_coverage

        policy = {
            "policy_number": "POL-100000",
            "coverage_types": ["water_damage", "storm_damage", "glass_damage"],
            "covered_items": {
                "water_damage": ["kitchen ceiling", "basement flooring"],
                "storm_damage": ["roof tiles"],
                "glass_damage": ["front door glass panel"],
            },
            "coverage_limit_eur": 50000,
            "deductible_eur": 500,
        }
        result = check_coverage(policy, "water", "kitchen ceiling")
        assert result["is_covered"] is True
        assert result["coverage_limit_eur"] == 50000

    def test_check_coverage_type_covered_item_not(self):
        from stage5_retrieval import check_coverage

        policy = {
            "policy_number": "POL-100000",
            "coverage_types": ["water_damage"],
            "covered_items": {
                "water_damage": ["kitchen ceiling", "basement flooring"],
            },
            "coverage_limit_eur": 50000,
            "deductible_eur": 500,
        }
        result = check_coverage(policy, "water", "garage concrete floor")
        assert result["is_covered"] is False
        assert "NOT in the covered items" in result["reason"]

    def test_check_coverage_not_covered(self):
        from stage5_retrieval import check_coverage

        policy = {
            "policy_number": "POL-100000",
            "coverage_types": ["storm_damage"],
            "covered_items": {"storm_damage": ["roof tiles"]},
            "coverage_limit_eur": 50000,
            "deductible_eur": 500,
        }
        result = check_coverage(policy, "water")
        assert result["is_covered"] is False

    def test_check_coverage_no_policy(self):
        from stage5_retrieval import check_coverage

        result = check_coverage(None, "water")
        assert result["is_covered"] is False
        assert result["policy_number"] is None

    def test_calculate_payout_approved(self):
        from stage5_retrieval import calculate_payout

        coverage = {
            "is_covered": True,
            "coverage_limit_eur": 50000,
            "deductible_eur": 500,
        }
        result = calculate_payout(coverage, 4165.00)
        assert result["approved"] is True
        assert result["payout_amount_eur"] == 3665.00

    def test_calculate_payout_denied(self):
        from stage5_retrieval import calculate_payout

        coverage = {
            "is_covered": False,
            "reason": "Not covered",
        }
        result = calculate_payout(coverage, 4165.00)
        assert result["approved"] is False
        assert result["payout_amount_eur"] == 0

    def test_calculate_payout_capped_at_limit(self):
        from stage5_retrieval import calculate_payout

        coverage = {
            "is_covered": True,
            "coverage_limit_eur": 1000,
            "deductible_eur": 0,
        }
        result = calculate_payout(coverage, 5000.00)
        assert result["payout_amount_eur"] == 1000


# ─── Config tests ─────────────────────────────────────────────

class TestConfig:
    """Validate configuration is accessible and sensible."""

    def test_config_paths_exist(self):
        from config import PDF_FOLDER, LOG_FOLDER, OUTPUT_FOLDER, CHROMA_FOLDER
        assert isinstance(PDF_FOLDER, str)
        assert isinstance(LOG_FOLDER, str)
        assert isinstance(OUTPUT_FOLDER, str)
        assert isinstance(CHROMA_FOLDER, str)

    def test_chunk_params_sensible(self):
        from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
        assert CHUNK_SIZE > CHUNK_OVERLAP, "Chunk size must exceed overlap"
        assert MIN_CHUNK_SIZE > 0
        assert CHUNK_OVERLAP >= 0

    def test_embedding_model_configured(self):
        from config import EMBEDDING_MODEL
        assert len(EMBEDDING_MODEL) > 0

    def test_reranker_model_configured(self):
        from config import RERANKER_MODEL, RERANK_TOP_K
        assert len(RERANKER_MODEL) > 0
        assert RERANK_TOP_K > 0

    def test_policy_paths_configured(self):
        from config import POLICY_METADATA, PAYOUT_REPORT
        assert isinstance(POLICY_METADATA, str)
        assert isinstance(PAYOUT_REPORT, str)
