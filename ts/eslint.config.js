// @ts-check
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  // Base JS recommended rules
  eslint.configs.recommended,

  // TypeScript-aware rules for all TS source files
  ...tseslint.configs.recommended,

  // Global ignores
  {
    ignores: [
      "**/dist/**",
      "**/node_modules/**",
      "**/coverage/**",
      "**/*.js",   // compiled output and plain JS config files
      "**/*.mjs",
      "**/*.cjs",
    ],
  },

  // Source files — full type-aware linting
  {
    files: ["packages/*/src/**/*.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "@typescript-eslint/no-non-null-assertion": "warn",
      "@typescript-eslint/no-floating-promises": "error",
      "no-console": "warn",
    },
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },

  // Test files — relaxed rules, no type-aware analysis (tests are excluded from tsconfig)
  {
    files: ["packages/*/tests/**/*.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-floating-promises": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-require-imports": "off",
      "no-console": "off",
      "no-useless-assignment": "off",
      "preserve-caught-error": "off",
      "require-yield": "off",
      "no-empty": "off",
      "no-useless-escape": "off",
    },
    languageOptions: {
      parserOptions: {
        // No projectService — avoids tsconfig resolution errors for test files
        project: false,
      },
    },
  }
);
