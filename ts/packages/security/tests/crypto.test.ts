import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  generateMasterKey,
  deriveKey,
  encrypt,
  decrypt,
  isValidMasterKey,
} from "../src/crypto.js";
import {
  getMasterKey,
  clearMasterKeyCache,
  setMasterKey,
  isUsingEnvKey,
} from "../src/master-key.js";
import {
  getSecret,
  getSecretRequired,
  hasSecret,
  getSecretSync,
  clearSecretCache,
  clearAllSecretCache,
} from "../src/secret-helper.js";

describe("crypto", () => {
  describe("generateMasterKey", () => {
    it("should return a valid base64 string", () => {
      const key = generateMasterKey();
      expect(typeof key).toBe("string");
      // Should be valid base64
      const decoded = Buffer.from(key, "base64");
      expect(decoded.toString("base64")).toBe(key);
    });

    it("should return a key of correct length (32 bytes)", () => {
      const key = generateMasterKey();
      const decoded = Buffer.from(key, "base64");
      expect(decoded.length).toBe(32);
    });

    it("should generate unique keys", () => {
      const key1 = generateMasterKey();
      const key2 = generateMasterKey();
      expect(key1).not.toBe(key2);
    });
  });

  describe("deriveKey", () => {
    it("should return a 32-byte buffer", () => {
      const masterKey = generateMasterKey();
      const derived = deriveKey(masterKey, "user-1");
      expect(derived.length).toBe(32);
    });

    it("should be deterministic for same inputs", () => {
      const masterKey = generateMasterKey();
      const derived1 = deriveKey(masterKey, "user-1");
      const derived2 = deriveKey(masterKey, "user-1");
      expect(derived1.equals(derived2)).toBe(true);
    });

    it("should produce different keys for different users", () => {
      const masterKey = generateMasterKey();
      const derived1 = deriveKey(masterKey, "user-1");
      const derived2 = deriveKey(masterKey, "user-2");
      expect(derived1.equals(derived2)).toBe(false);
    });

    it("should produce different keys for different master keys", () => {
      const masterKey1 = generateMasterKey();
      const masterKey2 = generateMasterKey();
      const derived1 = deriveKey(masterKey1, "user-1");
      const derived2 = deriveKey(masterKey2, "user-1");
      expect(derived1.equals(derived2)).toBe(false);
    });
  });

  describe("encrypt/decrypt", () => {
    it("should round-trip encrypt and decrypt", () => {
      const masterKey = generateMasterKey();
      const userId = "user-1";
      const plaintext = "my-secret-api-key-12345";

      const encrypted = encrypt(masterKey, userId, plaintext);
      const decrypted = decrypt(masterKey, userId, encrypted);

      expect(decrypted).toBe(plaintext);
    });

    it("should handle empty strings", () => {
      const masterKey = generateMasterKey();
      const userId = "user-1";

      const encrypted = encrypt(masterKey, userId, "");
      const decrypted = decrypt(masterKey, userId, encrypted);

      expect(decrypted).toBe("");
    });

    it("should handle unicode content", () => {
      const masterKey = generateMasterKey();
      const userId = "user-1";
      const plaintext = "secret with unicode: \u00e9\u00e0\u00fc\u00f1 \ud83d\udd10";

      const encrypted = encrypt(masterKey, userId, plaintext);
      const decrypted = decrypt(masterKey, userId, encrypted);

      expect(decrypted).toBe(plaintext);
    });

    it("should produce different ciphertexts for same plaintext (due to random IV)", () => {
      const masterKey = generateMasterKey();
      const userId = "user-1";
      const plaintext = "same-secret";

      const encrypted1 = encrypt(masterKey, userId, plaintext);
      const encrypted2 = encrypt(masterKey, userId, plaintext);

      expect(encrypted1).not.toBe(encrypted2);
      // But both decrypt to the same value
      expect(decrypt(masterKey, userId, encrypted1)).toBe(plaintext);
      expect(decrypt(masterKey, userId, encrypted2)).toBe(plaintext);
    });

    it("should produce different ciphertexts for different users", () => {
      const masterKey = generateMasterKey();
      const plaintext = "shared-secret";

      const encrypted1 = encrypt(masterKey, "user-1", plaintext);
      const encrypted2 = encrypt(masterKey, "user-2", plaintext);

      expect(encrypted1).not.toBe(encrypted2);
    });

    it("should fail to decrypt with wrong master key", () => {
      const masterKey1 = generateMasterKey();
      const masterKey2 = generateMasterKey();
      const userId = "user-1";

      const encrypted = encrypt(masterKey1, userId, "secret");

      expect(() => decrypt(masterKey2, userId, encrypted)).toThrow(
        "Failed to decrypt secret"
      );
    });

    it("should fail to decrypt with wrong userId", () => {
      const masterKey = generateMasterKey();

      const encrypted = encrypt(masterKey, "user-1", "secret");

      expect(() => decrypt(masterKey, "user-2", encrypted)).toThrow(
        "Failed to decrypt secret"
      );
    });

    it("should fail to decrypt corrupted data", () => {
      const masterKey = generateMasterKey();
      const userId = "user-1";

      expect(() => decrypt(masterKey, userId, "not-valid-base64!!!")).toThrow();
    });

    it("should fail to decrypt truncated data", () => {
      const masterKey = generateMasterKey();
      const userId = "user-1";

      // Too short to contain IV + authTag
      const shortData = Buffer.alloc(10).toString("base64");
      expect(() => decrypt(masterKey, userId, shortData)).toThrow(
        "Failed to decrypt secret: data too short"
      );
    });
  });

  describe("isValidMasterKey", () => {
    it("should return true for valid key", () => {
      const masterKey = generateMasterKey();
      const userId = "user-1";
      const encrypted = encrypt(masterKey, userId, "test-value");

      expect(isValidMasterKey(masterKey, encrypted, userId)).toBe(true);
    });

    it("should return false for wrong key", () => {
      const masterKey1 = generateMasterKey();
      const masterKey2 = generateMasterKey();
      const userId = "user-1";
      const encrypted = encrypt(masterKey1, userId, "test-value");

      expect(isValidMasterKey(masterKey2, encrypted, userId)).toBe(false);
    });
  });
});

describe("master-key", () => {
  const originalEnv = process.env["SECRETS_MASTER_KEY"];

  beforeEach(() => {
    clearMasterKeyCache();
    delete process.env["SECRETS_MASTER_KEY"];
  });

  afterEach(() => {
    clearMasterKeyCache();
    if (originalEnv !== undefined) {
      process.env["SECRETS_MASTER_KEY"] = originalEnv;
    } else {
      delete process.env["SECRETS_MASTER_KEY"];
    }
  });

  it("should return env var when set", () => {
    process.env["SECRETS_MASTER_KEY"] = "test-master-key-from-env";
    const key = getMasterKey();
    expect(key).toBe("test-master-key-from-env");
  });

  it("should auto-generate a key when no env var", () => {
    const key = getMasterKey();
    expect(typeof key).toBe("string");
    expect(key.length).toBeGreaterThan(0);
  });

  it("should cache the key across calls", () => {
    const key1 = getMasterKey();
    const key2 = getMasterKey();
    expect(key1).toBe(key2);
  });

  it("should clear cache", () => {
    const key1 = getMasterKey();
    clearMasterKeyCache();
    // Without env var, a new key will be generated
    const key2 = getMasterKey();
    // They could be different since new key is generated
    expect(typeof key2).toBe("string");
  });

  it("should allow setting a custom key", () => {
    setMasterKey("custom-key");
    expect(getMasterKey()).toBe("custom-key");
  });

  it("should report env key status", () => {
    expect(isUsingEnvKey()).toBe(false);
    process.env["SECRETS_MASTER_KEY"] = "test";
    expect(isUsingEnvKey()).toBe(true);
  });
});

describe("secret-helper", () => {
  const savedEnv: Record<string, string | undefined> = {};

  beforeEach(() => {
    clearAllSecretCache();
    // Save and clear test env vars
    for (const key of ["TEST_SECRET", "OPENAI_API_KEY", "SUPABASE_URL"]) {
      savedEnv[key] = process.env[key];
      delete process.env[key];
    }
  });

  afterEach(() => {
    clearAllSecretCache();
    // Restore env vars
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value !== undefined) {
        process.env[key] = value;
      } else {
        delete process.env[key];
      }
    }
  });

  describe("getSecret", () => {
    it("should return env var value", async () => {
      process.env["TEST_SECRET"] = "env-value";
      const value = await getSecret("TEST_SECRET", "user-1");
      expect(value).toBe("env-value");
    });

    it("should return null when not found", async () => {
      const value = await getSecret("NONEXISTENT_KEY", "user-1");
      expect(value).toBeNull();
    });

    it("should return default when not found", async () => {
      const value = await getSecret("NONEXISTENT_KEY", "user-1", "default-val");
      expect(value).toBe("default-val");
    });

    it("should prioritize forced env keys", async () => {
      process.env["SUPABASE_URL"] = "https://forced.supabase.co";
      const value = await getSecret("SUPABASE_URL", "user-1");
      expect(value).toBe("https://forced.supabase.co");
    });
  });

  describe("getSecretRequired", () => {
    it("should return value when found", async () => {
      process.env["TEST_SECRET"] = "required-value";
      const value = await getSecretRequired("TEST_SECRET", "user-1");
      expect(value).toBe("required-value");
    });

    it("should throw when not found", async () => {
      await expect(
        getSecretRequired("NONEXISTENT_KEY", "user-1")
      ).rejects.toThrow("Required secret 'NONEXISTENT_KEY' not found");
    });
  });

  describe("hasSecret", () => {
    it("should return true when env var exists", async () => {
      process.env["TEST_SECRET"] = "exists";
      expect(await hasSecret("TEST_SECRET", "user-1")).toBe(true);
    });

    it("should return false when not found", async () => {
      expect(await hasSecret("NONEXISTENT_KEY", "user-1")).toBe(false);
    });
  });

  describe("getSecretSync", () => {
    it("should return env var value", () => {
      process.env["TEST_SECRET"] = "sync-value";
      expect(getSecretSync("TEST_SECRET")).toBe("sync-value");
    });

    it("should return null when not found", () => {
      expect(getSecretSync("NONEXISTENT_KEY")).toBeNull();
    });

    it("should return default when not found", () => {
      expect(getSecretSync("NONEXISTENT_KEY", "default")).toBe("default");
    });

    it("should prioritize forced env keys (SUPABASE_URL)", () => {
      process.env["SUPABASE_URL"] = "https://sync.supabase.co";
      expect(getSecretSync("SUPABASE_URL")).toBe("https://sync.supabase.co");
    });
  });

  describe("clearSecretCache", () => {
    it("should clear a specific cached secret", async () => {
      // Populate cache by retrieving a secret
      process.env["TEST_SECRET"] = "cached-value";
      await getSecret("TEST_SECRET", "user-1");

      // Remove from env
      delete process.env["TEST_SECRET"];

      // Should still return cached value
      const cached = await getSecret("TEST_SECRET", "user-1");
      expect(cached).toBe("cached-value");

      // Clear the specific cache entry
      clearSecretCache("user-1", "TEST_SECRET");

      // Now should return null since env is also gone
      const afterClear = await getSecret("TEST_SECRET", "user-1");
      expect(afterClear).toBeNull();
    });
  });

  describe("getSecret caching", () => {
    it("should use default userId when not provided", async () => {
      process.env["TEST_SECRET"] = "default-user-val";
      const value = await getSecret("TEST_SECRET");
      expect(value).toBe("default-user-val");
    });

    it("should return cached value on second call", async () => {
      process.env["TEST_SECRET"] = "first-val";
      await getSecret("TEST_SECRET", "user-1");

      // Change env but cache should prevail
      process.env["TEST_SECRET"] = "second-val";
      const cached = await getSecret("TEST_SECRET", "user-1");
      expect(cached).toBe("first-val");
    });

    it("should return null from cache when cached as null", async () => {
      // First call with no env value sets null in cache path (actually returns null)
      const val = await getSecret("NONEXISTENT_CACHED", "user-1");
      expect(val).toBeNull();
    });
  });

  describe("hasSecret caching", () => {
    it("should detect cached secrets", async () => {
      process.env["TEST_SECRET"] = "exists-for-cache";
      await getSecret("TEST_SECRET", "user-1");
      delete process.env["TEST_SECRET"];

      // Should find it in cache
      const found = await hasSecret("TEST_SECRET", "user-1");
      expect(found).toBe(true);
    });

    it("should use default userId when not provided", async () => {
      const found = await hasSecret("NONEXISTENT_KEY");
      expect(found).toBe(false);
    });
  });
});
