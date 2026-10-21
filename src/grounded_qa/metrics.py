import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, golds: list[str]) -> bool:
    return any(normalize(pred) == normalize(g) for g in golds)


def f1(pred: str, golds: list[str]) -> float:
    def single(p: str, g: str) -> float:
        p_tokens = normalize(p).split()
        g_tokens = normalize(g).split()
        same = sum((Counter(p_tokens) & Counter(g_tokens)).values())
        if same == 0:
            return 0.0
        precision = same / len(p_tokens)
        recall = same / len(g_tokens)
        return 2 * precision * recall / (precision + recall)
    return max(single(pred, g) for g in golds)


if __name__ == "__main__":
    assert normalize("The Normans!") == "normans"
    assert exact_match("The Normans", ["Normans", "the Normans"])
    assert not exact_match("France, Europe", ["France"])
    print(f"F1('in France', ['France']) = {f1('in France', ['France']):.2f}")
    print("Minden teszt átment.")