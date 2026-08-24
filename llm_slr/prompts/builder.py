LIKERT_SCALE = (
    "1 = strongly disagree, 2 = disagree, 3 = somewhat disagree, "
    "4 = neither agree nor disagree, 5 = somewhat agree, 6 = agree, "
    "7 = strongly agree"
)

SYSTEM_PROMPT = (
    "You are an expert research assistant supporting the study selection "
    "stage of a systematic literature review (SLR) in software engineering. "
    "You judge candidate studies strictly by their title and abstract, "
    "against the review's inclusion criteria. "
    "Always respond with a single JSON object and nothing else."
)


def _criteria_block(criteria):
    lines = [f"{i}. {c}" for i, c in enumerate(criteria.inclusion_criteria, 1)]
    return "\n".join(lines)


MAX_EXAMPLE_ABSTRACT_CHARS = 700


def _example_block(example):
    decision = "included" if example.label == 1 else "not included"
    abstract = example.abstract
    if len(abstract) > MAX_EXAMPLE_ABSTRACT_CHARS:
        abstract = abstract[:MAX_EXAMPLE_ABSTRACT_CHARS].rsplit(" ", 1)[0] + " [...]"
    return (
        f"Title: {example.title}\n"
        f"Abstract: {abstract}\n"
        f"Decision by the original reviewers: {decision}"
    )


def build_user_prompt(article, criteria, examples=()):
    parts = [
        f"Review topic: {criteria.topic}",
        f"Research question: {criteria.research_question}",
        "Inclusion criteria — a study is relevant to this review when it "
        "matches the scope described by the criteria below:\n"
        + _criteria_block(criteria),
    ]

    if examples:
        shown = "\n\n".join(_example_block(e) for e in examples)
        parts.append(
            "Examples of decisions made by the human reviewers in the "
            "original review:\n\n" + shown
        )

    parts.append(
        "Now rate your agreement that the following candidate study should "
        "be INCLUDED in the review, using a 7-point Likert scale "
        f"({LIKERT_SCALE}).\n\n"
        f"Title: {article.title}\n"
        f"Abstract: {article.abstract}"
    )
    parts.append(
        'Respond with a JSON object exactly in this format: '
        '{"score": <integer 1-7>, "justification": "<one short sentence>"}'
    )
    return "\n\n".join(parts)
