export interface CanaryMetrics {
  baselineErrorRate: number;
  candidateErrorRate: number;
  baselineCompletionRate: number;
  candidateCompletionRate: number;
  baselineP95Ms: number;
  candidateP95Ms: number;
  outputParityScore: number;
}

export interface CanaryThresholds {
  maxErrorRateDelta: number;
  maxCompletionRateDrop: number;
  maxP95RegressionRatio: number;
  minOutputParityScore: number;
}

export interface CanaryGateResult {
  pass: boolean;
  reasons: string[];
}

export const DEFAULT_THRESHOLDS: CanaryThresholds = {
  maxErrorRateDelta: 0.005,
  maxCompletionRateDrop: 0.01,
  maxP95RegressionRatio: 1.2,
  minOutputParityScore: 0.99,
};

export function evaluateCanaryGates(
  metrics: CanaryMetrics,
  thresholds: CanaryThresholds = DEFAULT_THRESHOLDS
): CanaryGateResult {
  const reasons: string[] = [];

  const errorDelta = metrics.candidateErrorRate - metrics.baselineErrorRate;
  if (errorDelta > thresholds.maxErrorRateDelta) {
    reasons.push(
      `Error-rate regression ${errorDelta.toFixed(4)} exceeds ${thresholds.maxErrorRateDelta.toFixed(4)}`
    );
  }

  const completionDrop = metrics.baselineCompletionRate - metrics.candidateCompletionRate;
  if (completionDrop > thresholds.maxCompletionRateDrop) {
    reasons.push(
      `Completion-rate drop ${completionDrop.toFixed(4)} exceeds ${thresholds.maxCompletionRateDrop.toFixed(4)}`
    );
  }

  const p95Ratio = metrics.baselineP95Ms === 0 ? 1 : metrics.candidateP95Ms / metrics.baselineP95Ms;
  if (p95Ratio > thresholds.maxP95RegressionRatio) {
    reasons.push(
      `P95 regression ratio ${p95Ratio.toFixed(3)} exceeds ${thresholds.maxP95RegressionRatio.toFixed(3)}`
    );
  }

  if (metrics.outputParityScore < thresholds.minOutputParityScore) {
    reasons.push(
      `Output parity ${metrics.outputParityScore.toFixed(4)} below ${thresholds.minOutputParityScore.toFixed(4)}`
    );
  }

  return {
    pass: reasons.length === 0,
    reasons,
  };
}
