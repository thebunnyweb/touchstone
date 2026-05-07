import pytest
from touchstone.rubric import Rubric, FACTUALITY, HELPFULNESS


def test_rubric_chain():
    r = Rubric().add("accuracy").add("clarity", weight=2.0)
    assert len(r.criteria) == 2
    assert r.criteria[1].weight == 2.0


def test_rubric_from_list():
    r = Rubric.from_list(["a", "b", "c"])
    assert len(r.criteria) == 3
    assert r.criteria[0].name == "a"


def test_rubric_prompt_section():
    r = Rubric().add("factual accuracy", weight=1.0)
    section = r.to_prompt_section()
    assert "factual accuracy" in section
    assert "100%" in section


def test_builtin_factuality():
    assert len(FACTUALITY.criteria) == 1


def test_builtin_helpfulness():
    assert len(HELPFULNESS.criteria) == 3
    total_weight = sum(c.weight for c in HELPFULNESS.criteria)
    assert abs(total_weight - 1.0) < 0.01
