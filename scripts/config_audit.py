"""Audit the configuration surface: nothing declared may be unread, and config.yaml may not drift.

Two failure modes this catches, both of which have bitten this project:

1. **Dead config** — a field declared on AppConfig that nothing ever reads. It looks like a knob the
   user can tune; in fact editing it does nothing. (``subject_coverage_threshold`` shipped this way.)
2. **Drift** — config.yaml carries a value that differs from the AppConfig default, or omits a field
   the code defaults elsewhere. Whichever file a reader trusts, they get the wrong number.

Exit code 1 on any finding, so CI can gate on it::

    uv run python scripts/config_audit.py
"""

from ba_agent.config_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
