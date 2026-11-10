"""The interview must not ask the stakeholder the same question twice.

A local model asked, as Q1 and Q2 of a real LM Studio run:

    Q1: Will the portal replace email for claim submission and tracking only, or also replace
        manual review and payment steps?
    Q2: For scope, does the portal replace email for submission/tracking only, or also replace
        manual review and payment execution?

The "do not repeat" instruction in the prompt did not prevent it. The guard has to be
deterministic, and it has to compare *subject-matter words* rather than characters — the two
questions above differ in framing while sharing 8 of 12 content words.
"""

from __future__ import annotations

from ba_agent.schemas import question_is_duplicate
from ba_agent.workflow import WorkflowEngine

Q1 = (
    "Will the portal replace email for claim submission and tracking only, or also replace manual "
    "review and payment steps?"
)
Q2 = (
    "For scope, does the portal replace email for submission/tracking only, or also replace manual "
    "review and payment execution?"
)

BROAD_AND_NARROW = (
    "Which operational reports should the portal provide: claim status, SLA breaches, duplicate "
    "alerts, payment reconciliation, or audit trails?",
    "Do you need operational reports for SLA breaches, duplicate alerts, and payment reconciliation exceptions?",
)
EMAIL_CUTOVER = (
    "During the cutover from email to the portal, should email remain available as a fallback "
    "submission channel, and for how long?",
    "After the portal rollout, will email remain an accepted fallback channel for claim submission "
    "or is email fully retired?",
)
POLICY_FAILURE = (
    "When the portal cannot retrieve policy data from the existing policy system over REST, should "
    "it block the claim, queue it for retry, or allow manual entry?",
    "If the policy REST call fails or returns no policy data, should the portal block submission, "
    "allow a manual policy reference, or retry automatically?",
)
#: Same project, different subject: the guard must not collapse unrelated questions.
ROLES_PAIR = (
    "Which roles should have portal access for claim submission, tracking, review, and payment processing?",
    "What training will adjusters need to use the portal, and who delivers it?",
)

ROLES = (
    "Which roles should have portal access for claim submission, tracking, review, and payment processing?",
    "Do any of these roles, or another role, have portal access for payment processing approvals?",
)


#: A realistic session, so the guard's inverse-document-frequency weighting has something to work
#: with. Every pair below is verbatim from captured interviews.
SESSION = [
    "When a broker submits a claim in the portal, should we immediately accept it and assign a "
    "'Submitted' status for your team to review later, or block submission until all required "
    "details and documents are complete?",
    "What is the expected peak number of claim submissions per day in winter?",
    "Which roles should have portal access for claim submission, tracking, review, and payment processing?",
    "What uptime or availability target is required for the portal?",
    "How will internal adjusters and managers authenticate and be authorized in the portal?",
    "What audit trail must the portal keep for every claim status change?",
    "Which claim-status reporting outputs should be available at launch: a real-time broker "
    "dashboard, a daily summary, a monthly performance report, or an adjuster queue view?",
    "What retention period is required for claim records and attachments in the portal?",
    "How many brokers are expected to submit claims through the portal each month?",
    "What training will adjusters need to use the portal, and who delivers it?",
]


def test_a_rephrased_question_is_detected_as_a_duplicate():
    assert question_is_duplicate(Q1, Q2, SESSION) is True
    assert question_is_duplicate(Q2, Q1, SESSION) is True


def test_an_identical_question_is_detected():
    assert question_is_duplicate(Q1, Q1, SESSION) is True


def test_related_but_distinct_questions_are_not_suppressed():
    """Over-sensitive dedup is worse than none: it silently closes subjects the stakeholder never answered."""
    for a, b in (POLICY_FAILURE, ROLES_PAIR):
        corpus = [*SESSION, a]
        assert question_is_duplicate(a, b, corpus) is False, f"false positive: {a[:60]!r} vs {b[:60]!r}"


def test_a_short_question_reasked_at_length_is_still_a_repeat():
    """The ratio floor must not rescue a re-ask just because the second wording is longer.

    Real runs produce this: the model asks a short question, gets a partial answer, then asks the
    same thing again with four extra clauses bolted on. Character ratio is low because of the length
    difference, not because the question changed.
    """
    short = "Which of these should be out of scope for the broker portal at launch?"
    long = (
        "What are the explicit MVP launch scope boundaries, including which claim types, broker "
        "accounts, historical email claims, dashboards, notifications, and legacy email intake "
        "processes are in scope, out of scope, deferred, or subject to cutover criteria?"
    )
    assert question_is_duplicate(short, long, [*SESSION, short, long]) is True


def test_a_narrower_followup_on_the_same_subject_survives_the_subset_branch():
    """'On-duty notifications' vs 'how do we identify the on-duty adjuster' is a new question, not a restatement."""
    a = (
        "How should adjuster follow-ups and on-duty notifications be defined: what constitutes a "
        "follow-up, who can see or act on it, are deadlines or statuses required, and should email "
        "be used for on-duty alerts?"
    )
    b = "How should the system identify the on-duty adjuster for new claim email alerts?"
    assert question_is_duplicate(a, b, [*SESSION, a, b]) is False


def test_email_fallback_rephrased_as_a_retirement_question_is_a_repeat():
    """Both questions ask the same thing: does email stay usable for intake, and for how long."""
    a, b = EMAIL_CUTOVER
    assert question_is_duplicate(a, b, [*SESSION, a]) is True


def test_questions_about_different_topics_are_not_duplicates():
    assert question_is_duplicate(Q1, ROLES[0], SESSION) is False
    assert question_is_duplicate("", Q1, SESSION) is False
    assert question_is_duplicate("?", "!", SESSION) is False


#: Re-asks captured in real interviews. Each pair asks the stakeholder the same thing; a
#: character-similarity check alone let all of them through.
LIVE_RE_ASKS = [
    (
        "What portal uptime, recovery, and SLA targets are required for Phase 1?",
        "What uptime, recovery, and SLA target must the portal meet?",
    ),
    (
        "What compliance thresholds for claim opening and payment processing must the portal enforce or report?",
        "What are the exact compliance thresholds and enforcement rules for claim opening and payment processing?",
    ),
    (
        "Is the winter peak of approximately 1,200 claims per month, per day, or another time period?",
        "Is the winter peak of approximately 1,200 claims per month or per day, and over what time window must it be sustained?",
    ),
    (
        "How will internal adjusters and managers authenticate and be authorized in the portal: existing "
        "policy system credentials, SSO, or separate portal accounts?",
        "How will internal claims adjusters and managers authenticate and be authorized in the portal?",
    ),
    (
        "Which KPIs should be mandatory for phase 1 portal reporting: broker adoption, claim cycle time, "
        "duplicate reduction, or portal reliability?",
        "Are broker adoption, duplicate reduction, and portal reliability mandatory for phase 1 portal reporting, "
        "or only the monthly claims status, average handling time, and rejection reasons metrics?",
    ),
    (
        "During the cutover from email to the portal, how should claims already submitted by email but not yet "
        "resolved be handled?",
        "How should claims already submitted by email before cutover but not yet resolved be migrated into the "
        "portal and tracked going forward?",
    ),
    (
        "Which operational reports should the portal provide: claim status, SLA breaches, duplicate alerts, "
        "payment reconciliation, or audit trails?",
        "Do you need operational reports for SLA breaches, duplicate alerts, and payment reconciliation exceptions?",
    ),
    (BROAD_AND_NARROW[0], BROAD_AND_NARROW[1]),
]

#: Adjacent questions on the same theme that ask genuinely different things.
LIVE_DISTINCT = [
    (
        "Which KPIs should be mandatory for phase 1 portal reporting: broker adoption, claim cycle time, "
        "duplicate reduction, or portal reliability?",
        "What numeric thresholds for broker adoption, duplicate reduction, and portal reliability define Phase 1 success?",
    ),
    (
        "What is the expected peak number of claim submissions per day in winter?",
        "Over what period must the portal sustain the winter peak of approximately 1,200 claims?",
    ),
    (
        "How will internal adjusters and managers authenticate and be authorized in the portal?",
        "What audit trail must the portal keep for every claim status change?",
    ),
    (
        "What training will adjusters need to use the portal, and who delivers it?",
        "How will the portal handle duplicate claim submissions from the same broker for the same policy number?",
    ),
]


def test_real_re_asks_from_live_runs_are_caught():
    for a, b in LIVE_RE_ASKS:
        corpus = [*SESSION, a]
        assert question_is_duplicate(a, b, corpus) is True, f"missed re-ask: {a[:50]!r} vs {b[:50]!r}"


def test_adjacent_but_distinct_questions_survive():
    for a, b in LIVE_DISTINCT:
        corpus = [*SESSION, a]
        assert question_is_duplicate(a, b, corpus) is False, f"false positive: {a[:50]!r} vs {b[:50]!r}"


def test_the_guard_is_conservative_without_a_session_corpus():
    """With no corpus, "claim" carrying meaning is indistinguishable from "claim" as noise.

    The engine always supplies one; the fallback tightens so an early question is more likely to
    be asked than wrongly suppressed.
    """
    assert question_is_duplicate(Q1, Q2) is True  # near-verbatim, survives unweighted
    assert (
        question_is_duplicate(
            "What compliance thresholds for claim opening and payment processing must the portal enforce or report?",
            "Which roles should have portal access for claim submission, tracking, review, and payment processing?",
        )
        is False
    )


def test_a_duplicate_is_recorded_once_in_the_session(services):
    """The engine must not queue a second copy of a question already asked."""
    engine = WorkflowEngine(services)
    state = services.store.create_session(brief="claims portal for 300 brokers")

    first = engine._queue_question(state, subject="Scope", text=Q1)
    second = engine._queue_question(state, subject="Scope", text=Q2)
    assert first is not None
    assert second is None, "a rephrased question was queued as a new question"
    assert len(state.questions) == 1

    # A distinct question on the same subject still gets through.
    third = engine._queue_question(state, subject="Scope", text=ROLES[0])
    assert third is not None
    assert len(state.questions) == 2
