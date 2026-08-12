import { log } from "./logger.js";

export async function withRetry(
  fn,
  { attempts = 4, baseDelay = 2000, label = "op" } = {},
) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (err?.status && err.status < 500 && err.status !== 429) throw err;
      const isRateLimit = err?.status === 429 || err?.message?.includes("rate");
      const delay = baseDelay * Math.pow(2, i) + Math.random() * 500;
      if (i < attempts - 1) {
        log.warn(
          `${label} failed (attempt ${i + 1}/${attempts}): ${err.message} — retrying in ${Math.round(delay / 1000)}s`,
        );
        if (isRateLimit) await sleep(delay * 2);
        else await sleep(delay);
      }
    }
  }
  throw lastErr;
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
