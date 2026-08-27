"""
engine.py
---------
The orchestration layer. Ties profiling.py and insight_rules.py together:

    1. Profile the DataFrame (profiling.py)
    2. Run every rule in INSIGHT_RULES against (df, profile)
    3. Collect all findings into one flat list
    4. Sort by severity: critical -> warning -> info
    5. Return the sorted list to the UI (app.py)

Each rule is wrapped in its own try/except so that one buggy or unlucky rule
(e.g. a correlation check on a weird dataset) can never crash the whole app —
it just gets skipped, and the rest of the insights still show up.
"""

from profiling import profile_columns
from insight_rules import INSIGHT_RULES

# Lower number = higher priority when sorting
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def generate_insights(df) -> list:
    """
    Run the full profiling + rule-engine pipeline on a DataFrame.

    Returns:
        A list of finding dicts (column, message, severity), sorted so that
        critical findings appear first, then warnings, then info.
    """
    profile = profile_columns(df)

    all_findings = []
    for rule in INSIGHT_RULES:
        rule_name = rule["name"]
        rule_function = rule["function"]
        try:
            findings = rule_function(df, profile)
            if findings:
                all_findings.extend(findings)
        except Exception as e:
            # A single failing rule should never take down the whole app.
            # We silently skip it here; app.py could optionally surface
            # this in a debug panel if desired.
            print(f"[DataLens] Rule '{rule_name}' failed and was skipped: {e}")
            continue

    all_findings.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 3))
    return all_findings
