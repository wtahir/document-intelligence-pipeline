# knowledge_graph.py
# ═══════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH LAYER — Relational intelligence for insurance claims
# ═══════════════════════════════════════════════════════════════════
#
# Why this matters:
#   The pipeline stores everything as flat vectors in ChromaDB.
#   When someone asks "which claims from policyholder Mueller were denied?",
#   the vector search finds documents about "Mueller" and "denied" — but it
#   can't TRAVERSE the relationship: Mueller → Policy → Claims → Decisions.
#
#   A knowledge graph stores entities (claimants, policies, claims, invoices)
#   and their relationships. This enables:
#   - Multi-hop queries: "What's the total payout for all of Mueller's claims?"
#   - Relationship discovery: "Which claims share the same damage type?"
#   - Context enrichment: When retrieving a chunk about an invoice, also pull
#     related claim emails and photo documentation.
#
# Implementation:
#   We use NetworkX (in-memory graph) rather than Neo4j for two reasons:
#   1. No additional infrastructure (the pipeline is already complex enough)
#   2. The graph is small enough (hundreds of nodes, not millions) that
#      in-memory is faster than a database round-trip
#
#   The graph is built during extraction (Stage 2) and persisted as JSON.
#   At query time, we traverse the graph to find related entities and
#   use their IDs to retrieve additional chunks from ChromaDB.

import json
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from config import (
    OUTPUT_FOLDER, EXTRACTED_DATA, POLICY_METADATA,
    LOG_FOLDER, LOG_FORMAT,
)

os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "knowledge_graph.log"),
    level=logging.INFO,
    format=LOG_FORMAT,
)

GRAPH_PATH = os.path.join(OUTPUT_FOLDER, "knowledge_graph.json")


# ─── Entity Types ─────────────────────────────────────────────────

@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str                     # Unique identifier (e.g., "claimant::mueller")
    type: str                   # "claimant", "policy", "claim", "invoice", "photo", "damage_type"
    name: str                   # Human-readable name
    properties: dict = field(default_factory=dict)  # Additional attributes
    source_files: list[str] = field(default_factory=list)  # Which documents mention this entity


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source_id: str              # Entity ID
    target_id: str              # Entity ID
    relation: str               # "filed_claim", "has_policy", "covers", "submitted_invoice", etc.
    properties: dict = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph with JSON persistence."""
    entities: dict = field(default_factory=dict)       # id -> Entity
    relationships: list = field(default_factory=list)   # list of Relationship
    built_at: Optional[str] = None
    stats: dict = field(default_factory=dict)

    def add_entity(self, entity: Entity):
        """Add or merge an entity. Merges properties and source files if exists."""
        if entity.id in self.entities:
            existing = self.entities[entity.id]
            existing.properties.update(entity.properties)
            existing.source_files = list(set(existing.source_files + entity.source_files))
        else:
            self.entities[entity.id] = entity

    def add_relationship(self, rel: Relationship):
        """Add a relationship, avoiding exact duplicates."""
        for existing in self.relationships:
            if (existing.source_id == rel.source_id and
                existing.target_id == rel.target_id and
                existing.relation == rel.relation):
                return  # Already exists
        self.relationships.append(rel)

    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> list[Entity]:
        """
        BFS traversal from an entity up to max_hops away.
        Returns all reachable entities within the hop limit.

        Why max_hops=2?
        In insurance: claimant → claim → invoice is 2 hops.
        Going deeper returns too many loosely connected entities.
        """
        if entity_id not in self.entities:
            return []

        visited = set()
        queue = [(entity_id, 0)]
        result = []

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_hops:
                continue
            visited.add(current_id)

            if current_id != entity_id:  # Don't include the starting entity
                result.append(self.entities[current_id])

            if depth < max_hops:
                # Find all connected entities
                for rel in self.relationships:
                    if rel.source_id == current_id and rel.target_id not in visited:
                        queue.append((rel.target_id, depth + 1))
                    elif rel.target_id == current_id and rel.source_id not in visited:
                        queue.append((rel.source_id, depth + 1))

        return result

    def get_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> list[Entity]:
        """Fuzzy name search across entities."""
        name_lower = name.strip().lower()
        results = []
        for entity in self.entities.values():
            if entity_type and entity.type != entity_type:
                continue
            if name_lower in entity.name.lower():
                results.append(entity)
        return results

    def get_related_files(self, entity_id: str, max_hops: int = 2) -> list[str]:
        """
        Gets all source files connected to an entity within max_hops.
        Used to enrich retrieval — when we find a relevant entity,
        we pull ALL documents that mention it or its neighbors.
        """
        neighbors = self.get_neighbors(entity_id, max_hops)
        start_entity = self.entities.get(entity_id)
        files = set()
        if start_entity:
            files.update(start_entity.source_files)
        for neighbor in neighbors:
            files.update(neighbor.source_files)
        return sorted(files)

    def get_claims_for_claimant(self, claimant_name: str) -> list[dict]:
        """
        Returns all claims filed by a claimant with full context.
        This is a common insurance query pattern.
        """
        claimants = self.get_entity_by_name(claimant_name, "claimant")
        claims = []
        for claimant in claimants:
            for rel in self.relationships:
                if rel.source_id == claimant.id and rel.relation == "filed_claim":
                    claim_entity = self.entities.get(rel.target_id)
                    if claim_entity:
                        claims.append({
                            "claim_id": claim_entity.id,
                            "claim_number": claim_entity.name,
                            "properties": claim_entity.properties,
                            "source_files": claim_entity.source_files,
                        })
        return claims

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Returns all entities of a specific type."""
        return [e for e in self.entities.values() if e.type == entity_type]

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "built_at": self.built_at,
            "stats": self.stats,
            "entities": {eid: asdict(e) for eid, e in self.entities.items()},
            "relationships": [asdict(r) for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Deserialize from JSON dict."""
        graph = cls()
        graph.built_at = data.get("built_at")
        graph.stats = data.get("stats", {})

        for eid, edata in data.get("entities", {}).items():
            graph.entities[eid] = Entity(**edata)

        for rdata in data.get("relationships", []):
            graph.relationships.append(Relationship(**rdata))

        return graph


# ─── Graph Construction ───────────────────────────────────────────
#
# We build the graph from two sources:
# 1. extracted_data.json (Stage 2 output) — claims, claimants, invoices, photos
# 2. policy_metadata.json — policies, coverage types, limits
#
# This runs after Stage 2 and before Stage 4.

def _normalize_id(entity_type: str, name: str) -> str:
    """Creates a stable, lowercase entity ID."""
    clean = name.strip().lower().replace(" ", "_").replace(".", "").replace("-", "_")
    return f"{entity_type}::{clean}"


def build_graph() -> KnowledgeGraph:
    """
    Builds the knowledge graph from extracted data and policy metadata.

    Entity types created:
    - claimant: People who filed claims
    - claim: Individual claims (identified by claim number)
    - policy: Insurance policies
    - damage_type: Types of damage (water, storm, glass)
    - document: Individual documents (emails, invoices, photos)

    Relationship types:
    - filed_claim: claimant → claim
    - has_policy: claimant → policy
    - covers: policy → damage_type
    - claim_has_damage: claim → damage_type
    - documented_by: claim → document
    - invoiced_by: claim → document (specifically invoices)
    """
    graph = KnowledgeGraph()
    graph.built_at = datetime.now().isoformat()

    # Load extracted data
    if not os.path.exists(EXTRACTED_DATA):
        logging.warning("No extracted_data.json found — building empty graph")
        return graph

    with open(EXTRACTED_DATA, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load policy metadata
    policies = []
    if os.path.exists(POLICY_METADATA):
        with open(POLICY_METADATA, "r", encoding="utf-8") as f:
            policies = json.load(f)

    # --- Build entities from extracted documents ---
    for doc in documents:
        if doc.get("status") != "success":
            continue

        file_name = doc.get("file_name", "")
        claim_number = doc.get("claim_number")
        claimant_name = doc.get("claimant_name")
        damage_type = doc.get("damage_type")
        doc_type = doc.get("document_type")
        policy_number = doc.get("policy_number")

        # Create document entity
        doc_id = _normalize_id("document", file_name)
        graph.add_entity(Entity(
            id=doc_id,
            type="document",
            name=file_name,
            properties={
                "document_type": doc_type,
                "language": doc.get("language"),
                "confidence": doc.get("confidence"),
            },
            source_files=[file_name],
        ))

        # Create claim entity
        if claim_number:
            claim_id = _normalize_id("claim", claim_number)
            graph.add_entity(Entity(
                id=claim_id,
                type="claim",
                name=claim_number,
                properties={
                    "damage_type": damage_type,
                    "damaged_object": doc.get("damaged_object"),
                    "date": doc.get("date") or doc.get("damage_date"),
                    "total_amount_eur": doc.get("total_amount_eur"),
                    "damage_severity": doc.get("damage_severity"),
                },
                source_files=[file_name],
            ))

            # Link document to claim
            graph.add_relationship(Relationship(
                source_id=claim_id,
                target_id=doc_id,
                relation="documented_by" if doc_type != "invoice" else "invoiced_by",
            ))

        # Create claimant entity
        if claimant_name:
            claimant_id = _normalize_id("claimant", claimant_name)
            graph.add_entity(Entity(
                id=claimant_id,
                type="claimant",
                name=claimant_name,
                properties={},
                source_files=[file_name],
            ))

            # Link claimant to claim
            if claim_number:
                graph.add_relationship(Relationship(
                    source_id=claimant_id,
                    target_id=claim_id,
                    relation="filed_claim",
                ))

            # Link claimant to policy
            if policy_number:
                policy_id = _normalize_id("policy", policy_number)
                graph.add_relationship(Relationship(
                    source_id=claimant_id,
                    target_id=policy_id,
                    relation="has_policy",
                ))

        # Create damage type entity
        if damage_type:
            dt_id = _normalize_id("damage_type", damage_type)
            graph.add_entity(Entity(
                id=dt_id,
                type="damage_type",
                name=damage_type,
                source_files=[file_name],
            ))

            if claim_number:
                graph.add_relationship(Relationship(
                    source_id=claim_id,
                    target_id=dt_id,
                    relation="claim_has_damage",
                ))

    # --- Build entities from policy metadata ---
    for policy in policies:
        policy_number = policy.get("policy_number")
        if not policy_number:
            continue

        policy_id = _normalize_id("policy", policy_number)
        graph.add_entity(Entity(
            id=policy_id,
            type="policy",
            name=policy_number,
            properties={
                "policyholder_name": policy.get("policyholder_name"),
                "coverage_limit_eur": policy.get("coverage_limit_eur"),
                "deductible_eur": policy.get("deductible_eur"),
                "coverage_types": policy.get("coverage_types", []),
            },
        ))

        # Link policy to damage types it covers
        for coverage in policy.get("coverage_types", []):
            dt_id = _normalize_id("damage_type", coverage.replace("_damage", ""))
            graph.add_entity(Entity(
                id=dt_id,
                type="damage_type",
                name=coverage.replace("_damage", ""),
            ))
            graph.add_relationship(Relationship(
                source_id=policy_id,
                target_id=dt_id,
                relation="covers",
            ))

        # Link policyholder to policy
        policyholder = policy.get("policyholder_name")
        if policyholder:
            claimant_id = _normalize_id("claimant", policyholder)
            graph.add_entity(Entity(
                id=claimant_id,
                type="claimant",
                name=policyholder,
            ))
            graph.add_relationship(Relationship(
                source_id=claimant_id,
                target_id=policy_id,
                relation="has_policy",
            ))

    # Compute stats
    entity_types = {}
    for e in graph.entities.values():
        entity_types[e.type] = entity_types.get(e.type, 0) + 1

    relation_types = {}
    for r in graph.relationships:
        relation_types[r.relation] = relation_types.get(r.relation, 0) + 1

    graph.stats = {
        "total_entities": len(graph.entities),
        "total_relationships": len(graph.relationships),
        "entity_types": entity_types,
        "relation_types": relation_types,
    }

    logging.info(
        f"Knowledge graph built: {graph.stats['total_entities']} entities, "
        f"{graph.stats['total_relationships']} relationships"
    )

    return graph


def save_graph(graph: KnowledgeGraph):
    """Persist the knowledge graph to JSON."""
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)
    logging.info(f"Knowledge graph saved to {GRAPH_PATH}")


def load_graph() -> Optional[KnowledgeGraph]:
    """Load the knowledge graph from JSON."""
    if not os.path.exists(GRAPH_PATH):
        return None
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return KnowledgeGraph.from_dict(data)


# ─── Graph-Enhanced Retrieval ─────────────────────────────────────
#
# These functions are called by the agentic retrieval layer to enrich
# vector search results with graph context.

def get_graph_context(query_entities: dict, graph: KnowledgeGraph) -> dict:
    """
    Given entities extracted from a query, traverses the knowledge graph
    to find related entities and their source files.

    Returns a dict with:
    - related_entities: entities connected to the query entities
    - related_files: source files that should be included in retrieval
    - graph_facts: structured facts that can be injected into the prompt
    """
    related_entities = []
    related_files = set()
    graph_facts = []

    # Search by claim numbers
    for claim_num in query_entities.get("claim_numbers", []):
        claim_id = _normalize_id("claim", claim_num)
        if claim_id in graph.entities:
            claim = graph.entities[claim_id]
            related_entities.append(claim)
            related_files.update(graph.get_related_files(claim_id))
            graph_facts.append(
                f"Claim {claim.name}: damage_type={claim.properties.get('damage_type')}, "
                f"amount={claim.properties.get('total_amount_eur')}, "
                f"severity={claim.properties.get('damage_severity')}"
            )

    # Search by claimant names
    for name in query_entities.get("names", []):
        matches = graph.get_entity_by_name(name, "claimant")
        for claimant in matches:
            related_entities.append(claimant)
            related_files.update(graph.get_related_files(claimant.id))
            claims = graph.get_claims_for_claimant(name)
            for claim in claims:
                graph_facts.append(
                    f"Claimant {claimant.name} filed claim {claim['claim_number']} "
                    f"({claim['properties'].get('damage_type')} damage)"
                )

    # Search by damage type
    for dt in query_entities.get("damage_types", []):
        dt_id = _normalize_id("damage_type", dt)
        if dt_id in graph.entities:
            # Find all claims with this damage type
            for rel in graph.relationships:
                if rel.target_id == dt_id and rel.relation == "claim_has_damage":
                    claim = graph.entities.get(rel.source_id)
                    if claim:
                        related_files.update(claim.source_files)

    return {
        "related_entities": [asdict(e) for e in related_entities],
        "related_files": sorted(related_files),
        "graph_facts": graph_facts,
    }


def build_and_save_graph():
    """CLI entry point — build and persist the knowledge graph."""
    print("Building knowledge graph...")
    graph = build_graph()
    save_graph(graph)
    print(f"\nKnowledge Graph Stats:")
    print(f"  Entities: {graph.stats.get('total_entities', 0)}")
    print(f"  Relationships: {graph.stats.get('total_relationships', 0)}")
    print(f"  Entity types: {graph.stats.get('entity_types', {})}")
    print(f"  Relation types: {graph.stats.get('relation_types', {})}")
    print(f"  Saved to: {GRAPH_PATH}")
    return graph


if __name__ == "__main__":
    build_and_save_graph()
