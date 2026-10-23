from app.docmap.entities import apply_groups, merge_entities


def e(name, aliases=(), kind="person", description=""):
    return {"name": name, "aliases": list(aliases), "kind": kind, "description": description}


def test_same_name_merges_and_collects_sections():
    merged = merge_entities([(1, e("Elizabeth Bennet")), (3, e("elizabeth bennet", description="heroine"))])
    assert len(merged) == 1
    assert merged[0].section_ids == {1, 3}
    assert merged[0].description == "heroine"


def test_alias_links_two_groups():
    raw = [(1, e("Elizabeth Bennet", ["Lizzy"])), (2, e("Lizzy")), (2, e("Lizzy")), (5, e("Elizabeth Bennet"))]
    merged = merge_entities(raw)
    assert len(merged) == 1
    assert merged[0].name in {"Elizabeth Bennet", "Lizzy"}
    assert merged[0].section_ids == {1, 2, 5}
    assert {"Elizabeth Bennet", "Lizzy"} == merged[0].all_names()


def test_ambiguous_alias_does_not_merge_a_family():
    raw = [
        (1, e("Mr. Bennet", ["Bennet"])),
        (1, e("Mrs. Bennet", ["Bennet"])),
        (1, e("Jane Bennet", ["Bennet"])),
    ]
    assert len(merge_entities(raw)) == 3


def test_junk_is_dropped_and_kinds_normalised():
    raw = [(1, {"name": ""}), (1, "not a dict"), (1, e("Acme Corp", kind="Company")), (1, e("Paris", kind="city"))]
    merged = {m.name: m.kind for m in merge_entities(raw)}
    assert merged == {"Acme Corp": "organisation", "Paris": "place"}


def test_apply_groups_from_llm():
    entities = merge_entities([(1, e("Elizabeth Bennet")), (2, e("Eliza")), (3, e("Mr. Darcy"))])
    names = [x.name for x in entities]
    groups = [[names.index("Elizabeth Bennet") + 1, names.index("Eliza") + 1], [99, -1], "junk"]
    result = apply_groups(entities, groups)
    assert len(result) == 2
    merged = next(x for x in result if "Eliza" in x.all_names())
    assert merged.section_ids == {1, 2}
