/**
 * Master key management.
 *
 * This module manages the master encryption key for secrets.
 * Currently supports environment variable-based key retrieval
 * with auto-generation for development use.
 *
 * Key sources (in order of precedence):
 * 1. SECRETS_MASTER_KEY environment variable
 * 2. Auto-generated key (development only, not persisted across restarts)
 */

import { generateMasterKey } from "./crypto.js";

let cachedMasterKey: string | null = null;

/**
 * Get the master encryption key.
 *
 * Checks sources in order:
 * 1. Cached key (if previously retrieved)
 * 2. SECRETS_MASTER_KEY environment variable
 * 3. Auto-generates a new key (for development; will not persist across restarts)
 *
 * @returns The master key as a base64-encoded string.
 */
export function getMasterKey(): string {
  if (cachedMasterKey !== null) {
    return cachedMasterKey;
  }

  // 1. Check environment variable
  const envKey = process.env["SECRETS_MASTER_KEY"];
  if (envKey) {
    cachedMasterKey = envKey;
    return envKey;
  }

  // 2. Auto-generate for development (not persisted)
  const newKey = generateMasterKey();
  cachedMasterKey = newKey;
  return newKey;
}

/**
 * Clear the cached master key.
 *
 * Forces the next call to getMasterKey() to re-read from
 * environment or generate a new key.
 */
export function clearMasterKeyCache(): void {
  cachedMasterKey = null;
}

/**
 * Set a specific master key (useful for testing).
 *
 * @param masterKey - The master key to use.
 */
export function setMasterKey(masterKey: string): void {
  cachedMasterKey = masterKey;
}

/**
 * Check if the master key is being sourced from an environment variable.
 *
 * @returns True if SECRETS_MASTER_KEY environment variable is set.
 */
export function isUsingEnvKey(): boolean {
  return process.env["SECRETS_MASTER_KEY"] !== undefined;
}
