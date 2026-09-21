/**
 * Every setting the UI writes through `updateSetting` needs an entry in
 * `settingUpdaters`, or the change is applied to the store, never sent to the
 * backend, and silently lost on the next reload. The store only logs a warning
 * for that, which is easy to miss — this check turns it into a failure.
 *
 * Run: `bun scripts/check-setting-updaters.ts`
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const STORE = "src/stores/settingsStore.ts";
/** Keys the store handles outside the `settingUpdaters` table. */
const HANDLED_ELSEWHERE = new Set(["bindings", "selected_model"]);

/**
 * Keys whose component calls a dedicated command itself and then writes to the
 * store only to keep it in sync. Legitimate, but each one has to be named here
 * so a genuinely unpersisted setting cannot hide among them.
 */
const PERSISTED_BY_COMPONENT: Record<string, string> = {
  model_unload_timeout: "commands.setModelUnloadTimeout",
};

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.(ts|tsx)$/.test(path) && path !== STORE ? [path] : [];
  });
}

const store = readFileSync(STORE, "utf8");
const table = store.slice(
  store.indexOf("const settingUpdaters"),
  store.indexOf("export const useSettingsStore"),
);
const declared = new Set(
  [...table.matchAll(/^\s{2}([a-z0-9_]+):/gm)].map((m) => m[1]),
);

const missing = new Map<string, string[]>();
for (const file of sourceFiles("src")) {
  const text = readFileSync(file, "utf8");
  for (const match of text.matchAll(/updateSetting\(\s*"([a-z0-9_]+)"/g)) {
    const key = match[1];
    if (declared.has(key) || HANDLED_ELSEWHERE.has(key)) continue;
    const command = PERSISTED_BY_COMPONENT[key];
    if (command && text.includes(command)) continue;
    missing.set(key, [...(missing.get(key) ?? []), file]);
  }
}

if (missing.size > 0) {
  console.error("Settings written by the UI but never persisted:\n");
  for (const [key, files] of missing) {
    console.error(`  ${key}  —  used in ${[...new Set(files)].join(", ")}`);
  }
  console.error(`\nAdd a handler to settingUpdaters in ${STORE}.`);
  process.exit(1);
}

console.log(
  `✓ every setting written by the UI has a handler (${declared.size} declared)`,
);
