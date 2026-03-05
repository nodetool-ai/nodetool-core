/**
 * Cryptographic utilities for secret encryption and decryption.
 *
 * This module provides utilities for encrypting and decrypting secrets using
 * AES-256-GCM symmetric encryption with PBKDF2 key derivation.
 *
 * The master key is combined with user_id (as salt) to derive user-specific
 * encryption keys via PBKDF2-SHA256 with 100,000 iterations.
 */

import { randomBytes, pbkdf2Sync, createCipheriv, createDecipheriv } from "node:crypto";

const PBKDF2_ITERATIONS = 100_000;
const KEY_LENGTH = 32; // 256 bits for AES-256
const IV_LENGTH = 12; // 96 bits for GCM
const AUTH_TAG_LENGTH = 16; // 128 bits

/**
 * Generate a new random master key.
 *
 * @returns A base64-encoded 32-byte master key string.
 */
export function generateMasterKey(): string {
  return randomBytes(KEY_LENGTH).toString("base64");
}

/**
 * Derive an encryption key from the master key using user_id as salt.
 *
 * This ensures each user's secrets are encrypted with a unique derived key,
 * providing isolation between users even if the master key is compromised.
 *
 * @param masterKey - The master key (base64-encoded or raw string).
 * @param userId - The user ID to use as salt for key derivation.
 * @returns A 32-byte derived key as a Buffer.
 */
export function deriveKey(masterKey: string, userId: string): Buffer {
  const salt = Buffer.from(userId, "utf-8");
  return pbkdf2Sync(masterKey, salt, PBKDF2_ITERATIONS, KEY_LENGTH, "sha256");
}

/**
 * Encrypt a plaintext value using AES-256-GCM.
 *
 * The output format is: base64(iv || ciphertext || authTag)
 *
 * @param masterKey - The master key to use for encryption.
 * @param userId - The user ID to use as salt for key derivation.
 * @param plaintext - The plaintext string to encrypt.
 * @returns The encrypted value as a base64-encoded string.
 */
export function encrypt(masterKey: string, userId: string, plaintext: string): string {
  const key = deriveKey(masterKey, userId);
  const iv = randomBytes(IV_LENGTH);
  const cipher = createCipheriv("aes-256-gcm", key, iv);

  const encrypted = Buffer.concat([
    cipher.update(plaintext, "utf-8"),
    cipher.final(),
  ]);
  const authTag = cipher.getAuthTag();

  // Pack: iv (12) + ciphertext (variable) + authTag (16)
  const packed = Buffer.concat([iv, encrypted, authTag]);
  return packed.toString("base64");
}

/**
 * Decrypt an encrypted value using AES-256-GCM.
 *
 * @param masterKey - The master key to use for decryption.
 * @param userId - The user ID to use as salt for key derivation.
 * @param encryptedValue - The encrypted value as a base64-encoded string.
 * @returns The decrypted plaintext string.
 * @throws {Error} If the master key is incorrect or the data is corrupted.
 */
export function decrypt(masterKey: string, userId: string, encryptedValue: string): string {
  const key = deriveKey(masterKey, userId);
  const packed = Buffer.from(encryptedValue, "base64");

  if (packed.length < IV_LENGTH + AUTH_TAG_LENGTH) {
    throw new Error("Failed to decrypt secret: data too short");
  }

  const iv = packed.subarray(0, IV_LENGTH);
  const authTag = packed.subarray(packed.length - AUTH_TAG_LENGTH);
  const ciphertext = packed.subarray(IV_LENGTH, packed.length - AUTH_TAG_LENGTH);

  try {
    const decipher = createDecipheriv("aes-256-gcm", key, iv);
    decipher.setAuthTag(authTag);

    const decrypted = Buffer.concat([
      decipher.update(ciphertext),
      decipher.final(),
    ]);
    return decrypted.toString("utf-8");
  } catch {
    throw new Error("Failed to decrypt secret");
  }
}

/**
 * Check if a master key is valid by attempting to decrypt a test value.
 *
 * @param masterKey - The master key to validate.
 * @param testEncryptedValue - An encrypted test value to decrypt.
 * @param userId - The user ID used as salt for the test value.
 * @returns True if the master key is valid, false otherwise.
 */
export function isValidMasterKey(
  masterKey: string,
  testEncryptedValue: string,
  userId: string
): boolean {
  try {
    decrypt(masterKey, userId, testEncryptedValue);
    return true;
  } catch {
    return false;
  }
}
