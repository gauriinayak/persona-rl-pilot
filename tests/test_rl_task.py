import pytest

from rl_task import (
    completion_text, correct_reward, extract_final, format_example, format_reward, same_number,
)


@pytest.mark.parametrize("text,expected", [
    ("so 3+4=7\n#### 7", "7"),
    ("#### 1,234", "1234"),
    ("#### $18", "18"),
    ("#### -5", "-5"),
    ("#### 3.50", "3.50"),
    ("first #### 4 then revised #### 9", "9"),   # last marker wins
    ("the answer is 7", None),                     # no marker
    ("####", None),
    ("", None),
    (None, None),
])
def test_extract_final(text, expected):
    assert extract_final(text) == expected


def test_same_number():
    assert same_number("18", "18")
    assert same_number("18.0", "18")
    assert not same_number("18", "19")
    assert not same_number(None, "18")
    assert not same_number("abc", "18")


def test_rewards_conversational_and_plain():
    comps = [
        [{"role": "assistant", "content": "work...\n#### 12"}],
        [{"role": "assistant", "content": "work...\n#### 13"}],
        [{"role": "assistant", "content": "no marker, 12"}],
    ]
    assert correct_reward(comps, ["12", "12", "12"]) == [1.0, 0.0, 0.0]
    assert format_reward(comps) == [1.0, 1.0, 0.0]
    assert correct_reward(["#### 12"], ["12"]) == [1.0]
    assert completion_text("plain") == "plain"


def test_format_example_strips_gold_formatting():
    row = {"question": "Q?", "answer": "reasoning <<1+1=2>>\n#### 1,200"}
    out = format_example(row)
    assert out["answer"] == "1200"
    assert out["prompt"][0]["role"] == "system"
    assert out["prompt"][1] == {"role": "user", "content": "Q?"}
