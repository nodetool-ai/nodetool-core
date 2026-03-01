# TS Runner Rollback Runbook

## Trigger Conditions
- Protocol drift affecting clients.
- Elevated output drift on critical workflows.
- Error/completion regression beyond SLO threshold.
- Runtime instability or sustained incident.

## Immediate Actions
1. Set `runner.engine=python`.
2. Keep `runner.shadow=on` for diagnosis.
3. Confirm recovery of completion/error metrics.
4. Open incident with failing workflow cohorts and drift samples.

## Triage
- Classify failures: protocol, ordering, output, timing-only.
- Identify first bad build and impacted package.
- Add regression tests before re-enable.

## Re-enable Criteria
- Root cause fixed with tests.
- Shadow drift back under threshold.
- Canary cohort passes for at least 24 hours.
