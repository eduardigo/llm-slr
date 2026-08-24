from dataclasses import dataclass

import yaml

from llm_slr.config import CRITERIA_DIR


@dataclass(frozen=True)
class ReviewCriteria:

    theme: str
    status: str
    topic: str
    research_question: str
    inclusion_criteria: tuple

    def __post_init__(self):
        if not self.inclusion_criteria:
            raise ValueError(f"tema '{self.theme}' sem critérios de inclusão")


def load_criteria(theme):
    path = CRITERIA_DIR / f"{theme}.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ReviewCriteria(
        theme=raw["theme"],
        status=raw["status"],
        topic=raw["topic"].strip(),
        research_question=raw["research_question"].strip(),
        inclusion_criteria=tuple(c.strip() for c in raw["inclusion_criteria"]),
    )
