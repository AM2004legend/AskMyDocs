"""Merge per-section entity extractions into one entity index for the whole document.

Each section's LLM output lists entities with a name and aliases. The same person shows up
many times ("Elizabeth Bennet" in ch.1, "Lizzy" in ch.2, "Elizabeth" in ch.3). Rules:
1. Same normalised name -> same entity.
2. A name/alias claimed by exactly two groups links them ("Lizzy" is an alias of
   "Elizabeth Bennet" and also the name of another group -> merge).
   A key claimed by 3+ groups is ambiguous ("Bennet" for Mr, Mrs and Jane) and is ignored.
3. Optionally an LLM pass groups the remaining duplicates (see summaries.py).
"""

import re
from collections import Counter
from dataclasses import dataclass, field

KIND_ALIASES = {
    "person": "person",
    "character": "person",
    "people": "person",
    "human": "person",
    "organisation": "organisation",
    "organization": "organisation",
    "company": "organisation",
    "org": "organisation",
    "group": "organisation",
    "institution": "organisation",
    "place": "place",
    "location": "place",
    "city": "place",
    "country": "place",
}


NON_NAME_PLACEHOLDER_RE = re.compile(r"\[?(EMAIL|UPI|CARD|AADHAAR|ACCOUNT|PHONE|PAN|IFSC|DOB)_Q?\d+\]?")


@dataclass
class Entity:
    name: str
    kind: str
    description: str = ""
    aliases: set[str] = field(default_factory=set)
    section_ids: set[int] = field(default_factory=set)
    name_counts: Counter = field(default_factory=Counter)

    def all_names(self) -> set[str]:
        return {self.name} | self.aliases


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", name.lower())).strip()


def clean_raw(item: dict) -> dict | None:
    """Validate one entity dict from the LLM; return None if it's junk."""
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    if not name or len(name) > 120 or not norm(name):
        return None
    if NON_NAME_PLACEHOLDER_RE.fullmatch(name):  # sensitive docs: "[ACCOUNT_1]" is not a character
        return None
    aliases = item.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    aliases = [str(a).strip() for a in aliases if str(a).strip() and len(str(a)) <= 120]
    kind = KIND_ALIASES.get(str(item.get("kind") or "").lower().strip(), "other")
    description = str(item.get("description") or "").strip()[:300]
    return {"name": name, "aliases": aliases, "kind": kind, "description": description}


def _combine(entities: list[Entity]) -> Entity:
    counts: Counter = Counter()
    for e in entities:
        counts.update(e.name_counts)
    name = max(counts, key=lambda n: (counts[n], len(n)))
    names = set().union(*(e.all_names() for e in entities))
    kind = Counter(e.kind for e in entities).most_common(1)[0][0]
    description = max((e.description for e in entities), key=len, default="")
    return Entity(
        name=name,
        kind=kind,
        description=description,
        aliases={n for n in names if norm(n) != norm(name)},
        section_ids=set().union(*(e.section_ids for e in entities)),
        name_counts=counts,
    )


def merge_entities(raw: list[tuple[int, dict]]) -> list[Entity]:
    """raw: (section ordinal, entity dict) pairs. Returns merged entities, most frequent first."""
    groups: dict[str, Entity] = {}
    for section_id, item in raw:
        item = clean_raw(item)
        if item is None:
            continue
        key = norm(item["name"])
        entity = groups.setdefault(key, Entity(name=item["name"], kind=item["kind"]))
        entity.name_counts[item["name"]] += 1
        entity.aliases.update(item["aliases"])
        entity.section_ids.add(section_id)
        if len(item["description"]) > len(entity.description):
            entity.description = item["description"]

    # union-find over group keys, linking groups that share an unambiguous name/alias
    parent = {k: k for k in groups}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    claims: dict[str, set[str]] = {}
    for key, entity in groups.items():
        for n in entity.all_names():
            if norm(n):
                claims.setdefault(norm(n), set()).add(key)
    for owners in claims.values():
        if len(owners) == 2:
            a, b = (find(k) for k in owners)
            parent[a] = b

    clusters: dict[str, list[Entity]] = {}
    for key, entity in groups.items():
        clusters.setdefault(find(key), []).append(entity)
    merged = [_combine(members) for members in clusters.values()]
    for e in merged:
        e.kind = e.kind if e.kind in {"person", "organisation", "place"} else "other"
    return sorted(merged, key=lambda e: (-len(e.section_ids), e.name))


def apply_groups(entities: list[Entity], groups: list[list[int]]) -> list[Entity]:
    """Merge entities by index groups (1-based, as shown to the LLM). Bad indices are ignored."""
    used: set[int] = set()
    result: list[Entity] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        idx = sorted({i - 1 for i in group if isinstance(i, int) and 0 < i <= len(entities)} - used)
        if len(idx) < 2:
            continue
        used.update(idx)
        result.append(_combine([entities[i] for i in idx]))
    result.extend(e for i, e in enumerate(entities) if i not in used)
    return sorted(result, key=lambda e: (-len(e.section_ids), e.name))
