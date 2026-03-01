export {
  diffMessageStreams,
  type Drift,
  type DriftCategory,
  type DiffOptions,
} from "./diff.js";

export {
  runShadowComparison,
  type CommandSpec,
  type ShadowRunResult,
  type ShadowComparison,
} from "./shadow.js";

export {
  evaluateCanaryGates,
  DEFAULT_THRESHOLDS,
  type CanaryMetrics,
  type CanaryThresholds,
  type CanaryGateResult,
} from "./gates.js";
