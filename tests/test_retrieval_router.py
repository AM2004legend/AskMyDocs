from app.docmap.router import ENTITY, SPECIFIC, WHOLE_DOCUMENT, route_question
from app.retrieval.hybrid import rrf_merge
from app.retrieval.keyword import to_or_query


def test_rrf_rewards_agreement_between_lists():
    vector = [1, 2, 3]
    keyword = [3, 4, 1]
    merged = rrf_merge([vector, keyword], k=60)
    ids = [i for i, _ in merged]
    assert ids[:2] == [1, 3]  # in both lists -> top
    assert set(ids) == {1, 2, 3, 4}
    assert merged[0][1] == 1 / 61 + 1 / 63


def test_rrf_single_list_keeps_order():
    assert [i for i, _ in rrf_merge([[5, 6, 7], []])] == [5, 6, 7]


def test_or_query_strips_syntax():
    query = to_or_query("What's the notice-period? (clause 14 & 15)!")
    assert query == "what | s | the | notice | period | clause | 14 | 15"
    assert to_or_query("?!") is None


ENTITIES = {1: {"Elizabeth Bennet", "Lizzy"}, 2: {"Mr. Darcy", "Fitzwilliam Darcy"}, 3: {"Al"}}


def test_router_whole_document():
    assert route_question("Summarise the story").kind == WHOLE_DOCUMENT
    assert route_question("What are the main themes?").kind == WHOLE_DOCUMENT
    assert route_question("What is this agreement about?").kind == WHOLE_DOCUMENT


def test_router_entities_by_name_and_alias():
    r = route_question("How does Lizzy's opinion of Mr. Darcy change?", ENTITIES)
    assert r.kind == ENTITY and sorted(r.entity_ids) == [1, 2]


def test_router_entity_list_questions():
    r = route_question("List the main characters", ENTITIES)
    assert r.kind == ENTITY and r.entity_ids == []


def test_router_ignores_tiny_names_and_partial_words():
    assert route_question("What is the total salary?", ENTITIES).kind == SPECIFIC  # "Al" ignored, "total" != "Al"


def test_router_specific_with_chapter():
    r = route_question("Explain the Krebs cycle from chapter IV")
    assert r.kind == SPECIFIC and r.section_number == 4
    assert route_question("Summarise chapter 3").section_number == 3
