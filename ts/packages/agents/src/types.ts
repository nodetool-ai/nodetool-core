/**
 * Core type definitions for the agent system.
 */

export interface Step {
  id: string;
  instructions: string;
  completed: boolean;
  startTime?: number;
  endTime?: number;
  dependsOn: string[];
  tools?: string[];
  outputSchema?: string;
  logs: string[];
}

export interface Task {
  id: string;
  title: string;
  description?: string;
  steps: Step[];
}
