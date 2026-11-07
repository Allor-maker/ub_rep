"""A5 concrete implementations match the existing public Protocols."""

from nsu_budget_agent.contracts.ports import (
    AcceptancePolicy,
    AttemptEngine,
    Publisher,
    SubmissionLayout,
)
from nsu_budget_agent.orchestration.engine import JiuwenAdapter
from nsu_budget_agent.submission import (
    AnswerOnlyPublisher,
    RootSubmissionLayout,
    V1AcceptancePolicy,
)


def consumers(engine: JiuwenAdapter, publisher: AnswerOnlyPublisher) -> None:
    attempt_engine: AttemptEngine = engine
    submission: Publisher = publisher
    policy: AcceptancePolicy = V1AcceptancePolicy()
    layout: SubmissionLayout = RootSubmissionLayout()
    _ = attempt_engine, submission, policy, layout
