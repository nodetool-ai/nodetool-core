# TS Runner Cutover Runbook

## Preconditions
- TS workspace build/test green.
- Shadow comparison enabled for selected workflows.
- Drift rates within threshold for 7 consecutive days.

## Feature Flags
- `runner.engine=python|ts`
- `runner.shadow=on|off`
- `runner.control_legacy=on|off`

## Rollout Steps
1. Enable shadow mode for internal workflows only.
2. Compare Python vs TS streams with parity harness reports.
3. Enable TS for low-risk workflow category cohort.
4. Expand by user cohort while monitoring SLOs.
5. Switch default to TS when error/completion/parity gates pass.

## Rollout Gates
- Error rate: no statistically significant regression.
- Completion rate: no statistically significant regression.
- Runtime p50/p95: within agreed margin.
- Output parity score: above agreed threshold.

## Observability
- Persist shadow diffs by run ID and category.
- Alert on protocol/output drift spikes.
- Track drift trend by workflow type.
