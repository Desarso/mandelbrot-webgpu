import { defineConfig } from "vitest/config";

// The numerical engine is deliberately independent of the UI, so its tests run
// in plain Node with no DOM and no Solid transform.
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
