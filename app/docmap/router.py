"""Decide how to answer a question: specific lookup, whole-document, or entity.

Simple keyword rules first (PRD: only add an LLM classifier if the eval set shows rules are weak).
"""

import re
from dataclasses import dataclass, field

from app.ingest.sections import parse_number

SPECIFIC = "specific"
WHOLE_DOCUMENT = "whole_document"
ENTITY = "entity"

WHOLE_RE = re.compile(
    r"\b(summar(y|ise|ize|ies|ising|izing)|overview|overall|gist|tl;?dr|in a nutshell|outline"
    r"|main (themes?|ideas?|points?|arguments?|message|plot)|key (points|takeaways|themes|ideas)"
    r"|themes?|plot|storyline"
    r"|what (is|was) (this|the) (book|document|story|novel|text|paper|chapter|agreement|contract) about"
    r"|what happens in)\b",
    re.IGNORECASE,
)
ENTITY_LIST_RE = re.compile(
    r"\b(characters?|who are the (main |key )?(people|parties|characters|persons|players)"
    r"|list (of |all )?(the )?(people|characters|parties|persons|names|organi[sz]ations|companies)"
    r"|(main|key) (people|persons|parties|figures))\b",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"\b(?:chapter|ch\.?|section|part)\s+([0-9]{1,3}|[ivxlcdm]{1,8}|[a-z]+)\b", re.IGNORECASE)


@dataclass
class Route:
    kind: str
    entity_ids: list[int] = field(default_factory=list)  # matched entities (empty = all, for list questions)
    section_number: int | None = None


def find_section_number(question: str) -> int | None:
    for match in SECTION_RE.finditer(question):
        number = parse_number(match.group(1))
        if number:
            return number
    return None


def match_entities(question: str, entity_names: dict[int, set[str]]) -> list[int]:
    """Entity ids whose name or alias appears in the question as whole words (case-insensitive)."""
    found = []
    for entity_id, names in entity_names.items():
        for name in names:
            if len(name) >= 3 and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.IGNORECASE):
                found.append(entity_id)
                break
    return found


def route_question(question: str, entity_names: dict[int, set[str]] | None = None) -> Route:
    section_number = find_section_number(question)
    matched = match_entities(question, entity_names or {})
    if matched:
        return Route(ENTITY, entity_ids=matched, section_number=section_number)
    if WHOLE_RE.search(question):
        return Route(WHOLE_DOCUMENT, section_number=section_number)
    if ENTITY_LIST_RE.search(question):
        return Route(ENTITY, section_number=section_number)
    return Route(SPECIFIC, section_number=section_number)
