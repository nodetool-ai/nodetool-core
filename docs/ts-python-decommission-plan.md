# Python Runner Decommission Plan

## Preconditions
- TS runner is default for all traffic.
- Rollback confidence established.
- No unresolved parity blockers.

## Decommission Steps
1. Remove Python runtime selection from default codepath.
2. Keep Python runner behind emergency-only flag for one release cycle.
3. Archive parity fixtures and shadow reports.
4. Remove Python workflow runner shims and dead integration code.
5. Update docs and on-call playbooks to TS-only runtime.

## Safety Checks
- Contract test suite remains green.
- No references to deprecated Python runner entry points in CI.
- Incident drills confirm rollback to previous TS release only.
