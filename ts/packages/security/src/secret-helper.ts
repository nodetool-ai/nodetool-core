/**
 * Helper functions for retrieving secrets at runtime.
 *
 * This module provides utilities to get secret values from environment
 * variables, with hooks for future encrypted database integration.
 */

/** Cache for resolved secrets: "userId:key" -> value */
const secretCache = new Map<string, string | null>();

/** Keys that should always prioritize environment variables */
const FORCE_ENV_PRIORITY = new Set([
  "SUPABASE_URL",
  "SUPABASE_KEY",
  "SUPABASE_SERVICE_ROLE_KEY",
  "SERVER_AUTH_TOKEN",
]);

/**
 * Clear a specific secret from the local cache.
 *
 * @param userId - The user ID.
 * @param key - The secret key.
 */
export function clearSecretCache(userId: string, key: string): void {
  secretCache.delete(`${userId}:${key}`);
}

/**
 * Clear all cached secrets.
 */
export function clearAllSecretCache(): void {
  secretCache.clear();
}

/**
 * Get a secret value for a user.
 *
 * Resolution order:
 * 1. Forced env priority keys -> environment variable
 * 2. Local cache
 * 3. Environment variable
 * 4. Default value
 *
 * @param key - The secret key (e.g., "OPENAI_API_KEY").
 * @param userId - The user ID (optional, used for cache scoping).
 * @param defaultValue - Default value if not found.
 * @returns The secret value, or null if not found.
 */
export async function getSecret(
  key: string,
  userId?: string,
  defaultValue?: string
): Promise<string | null> {
  const resolvedUserId = userId ?? "default";

  // 1. Check forced environment priority
  if (FORCE_ENV_PRIORITY.has(key)) {
    const envVal = process.env[key];
    if (envVal !== undefined) {
      return envVal;
    }
  }

  // 2. Check cache
  const cacheKey = `${resolvedUserId}:${key}`;
  if (secretCache.has(cacheKey)) {
    return secretCache.get(cacheKey) ?? null;
  }

  // 3. Check environment variable
  const envValue = process.env[key];
  if (envValue !== undefined) {
    secretCache.set(cacheKey, envValue);
    return envValue;
  }

  // 4. Return default
  if (defaultValue !== undefined) {
    return defaultValue;
  }

  return null;
}

/**
 * Get a required secret value for a user.
 *
 * Same as getSecret() but throws if the secret is not found.
 *
 * @param key - The secret key.
 * @param userId - The user ID (optional).
 * @returns The secret value.
 * @throws {Error} If the secret is not found.
 */
export async function getSecretRequired(
  key: string,
  userId?: string
): Promise<string> {
  const value = await getSecret(key, userId);
  if (value === null) {
    throw new Error(
      `Required secret '${key}' not found, please set it in the settings menu.`
    );
  }
  return value;
}

/**
 * Check if a secret exists for a user.
 *
 * @param key - The secret key.
 * @param userId - The user ID (optional).
 * @returns True if the secret exists (in env or cache), false otherwise.
 */
export async function hasSecret(
  key: string,
  userId?: string
): Promise<boolean> {
  const resolvedUserId = userId ?? "default";

  // Check environment
  if (process.env[key] !== undefined) {
    return true;
  }

  // Check cache
  const cacheKey = `${resolvedUserId}:${key}`;
  if (secretCache.has(cacheKey)) {
    return secretCache.get(cacheKey) !== null;
  }

  return false;
}

/**
 * Get a secret value synchronously from environment variables only.
 *
 * @param key - The secret key.
 * @param defaultValue - Default value if not found.
 * @returns The secret value, or null/default if not found.
 */
export function getSecretSync(
  key: string,
  defaultValue?: string
): string | null {
  // Check forced env priority
  if (FORCE_ENV_PRIORITY.has(key)) {
    const envVal = process.env[key];
    if (envVal !== undefined) {
      return envVal;
    }
  }

  // Check environment variable
  const envValue = process.env[key];
  if (envValue !== undefined) {
    return envValue;
  }

  // Return default
  if (defaultValue !== undefined) {
    return defaultValue;
  }

  return null;
}
