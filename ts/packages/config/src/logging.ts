/**
 * Logging configuration — T-CFG-3.
 *
 * Simple logging configuration that reads level from environment
 * and provides a global log level getter.
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

const VALID_LEVELS: LogLevel[] = ["debug", "info", "warn", "error"];

let currentLevel: LogLevel = "info";

export interface LoggingOptions {
  level?: LogLevel;
}

/**
 * Configure the global log level.
 *
 * Priority: explicit `level` option > `NODETOOL_LOG_LEVEL` env > `LOG_LEVEL` env > "info"
 */
export function configureLogging(opts: LoggingOptions = {}): void {
  if (opts.level) {
    currentLevel = opts.level;
    return;
  }

  const envLevel = (
    process.env.NODETOOL_LOG_LEVEL ??
    process.env.LOG_LEVEL ??
    "info"
  ).toLowerCase() as LogLevel;

  currentLevel = VALID_LEVELS.includes(envLevel) ? envLevel : "info";
}

/**
 * Get the current global log level.
 */
export function getLogLevel(): LogLevel {
  return currentLevel;
}
