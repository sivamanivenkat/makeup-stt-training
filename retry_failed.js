import path from 'path';
import * as checkpoint from './utils/checkpoint.js';
import { log } from './utils/logger.js';

const CKPT_DIR = process.env.CHECKPOINT_DIR || './checkpoints';
const FAILED_PATH = path.join(CKPT_DIR, 'failed.jsonl');

const PERMANENT_PATTERNS = [
  'Private video',
  'unavailable',
  'removed',
  'no longer available',
  'account associated',
  'terminated',
];

function isPermanent(reason) {
  return PERMANENT_PATTERNS.some(p => reason.includes(p));
}

function main() {
  const all = checkpoint.read(FAILED_PATH);

  const cleanupEntries = all.filter(e => e.reason?.startsWith('cleanup:'));
  const downloadEntries = all.filter(e => !e.reason?.startsWith('cleanup:'));

  const permanent = downloadEntries.filter(e => isPermanent(e.reason || ''));
  const retryable = downloadEntries.filter(e => !isPermanent(e.reason || ''));

  // Keep cleanup failures (harmless, cleanup phase retries them itself)
  // and permanent download failures (won't succeed on retry: real
  // private/removed/unavailable videos). Drop everything else — bot-check
  // failures (now fixable via --cookies-from-browser) and transient
  // errors (disk-full, ffmpeg missing, network hiccups, all now fixed) —
  // so runExtract() picks them up again.
  const kept = [...cleanupEntries, ...permanent];
  checkpoint.write(FAILED_PATH, kept);

  log.stat('Cleanup failures kept', cleanupEntries.length);
  log.stat('Permanent download failures kept', permanent.length);
  log.stat('Retryable download failures freed', retryable.length);
  log.stat('Next step', 'node orchestrator.js download');
}

main();
