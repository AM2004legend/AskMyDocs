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
    r"|what happens in|how does (it|the (story|book|novel)) (end|finish)|ending)\b",
    re.IGNORECASE,
)
# A question is ABOUT a person/organisation (not just mentioning one) when it asks who they are,
# what they are like, or how they relate to / feel about someone.
ENTITY_INTENT_RE = re.compile(
    r"\b(who (is|was|are|were)|tell me about|describe|relationship|relation between|role of|character of"
    r"|what (kind|sort) of (person|man|woman)|opinion of|feel(s|ings)? (about|for|towards)|think of"
    r"|how does .{1,40} change)\b",
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
    # Only questions ABOUT someone use the entity index. Measured on the eval set: sending every
    # question that merely names a character ("How much money does Mr. Bingley have?") to the entity
    # route dropped specific retrieval from 20% to 0%, because in a novel almost every question has a name.
    if matched and ENTITY_INTENT_RE.search(question):
        return Route(ENTITY, entity_ids=matched, section_number=section_number)
    if WHOLE_RE.search(question):
        return Route(WHOLE_DOCUMENT, section_number=section_number)
    if ENTITY_LIST_RE.search(question):
        return Route(ENTITY, section_number=section_number)
    return Route(SPECIFIC, section_number=section_number)
