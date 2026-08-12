import fs from 'fs';
import path from 'path';
import * as checkpoint from './utils/checkpoint.js';
import { log } from './utils/logger.js';

const CKPT_DIR  = process.env.CHECKPOINT_DIR || './checkpoints';
const AUDIO_DIR = process.env.AUDIO_DIR       || './temp/audio';
const DSET      = process.env.DATASET_DIR     || './dataset';
const MIN_VALID_BYTES = 8000;

function findCorruptSegmentIds() {
  const segDir = path.join(AUDIO_DIR, 'segments');
  const bad = new Set();
  for (const fn of fs.readdirSync(segDir)) {
    if (!fn.endsWith('.wav')) continue;
    const p = path.join(segDir, fn);
    if (fs.statSync(p).size < MIN_VALID_BYTES) {
      bad.add(fn.slice(0, -4));
      fs.unlinkSync(p);
    }
  }
  return bad;
}

function filterCheckpoint(filePath, idField, badIds) {
  const rows = checkpoint.read(filePath);
  if (!rows.length) return 0;
  const kept = rows.filter(r => !badIds.has(r[idField]));
  const removed = rows.length - kept.length;
  if (removed > 0) checkpoint.write(filePath, kept);
  return removed;
}

function purgeDataset(badIds) {
  let removed = 0;
  for (const split of ['train', 'val', 'test']) {
    const metaPath = path.join(DSET, split, 'metadata.jsonl');
    if (!fs.existsSync(metaPath)) continue;
    const rows = checkpoint.read(metaPath);
    const kept = [];
    for (const row of rows) {
      if (badIds.has(row.segment_id)) {
        const wavPath = path.join(DSET, split, row.file_name);
        try { fs.unlinkSync(wavPath); } catch {}
        removed++;
      } else {
        kept.push(row);
      }
    }
    if (kept.length !== rows.length) {
      fs.writeFileSync(metaPath, kept.map(o => JSON.stringify(o)).join('\n') + '\n');
    }
  }
  return removed;
}

function main() {
  const badIds = findCorruptSegmentIds();
  log.stat('Corrupt WAV files found & deleted', badIds.size);
  if (!badIds.size) { log.success('No corruption found'); return; }
  for (const id of badIds) log.warn(`  ${id}`);

  const dl = filterCheckpoint(path.join(CKPT_DIR, 'downloaded.jsonl'), 'segmentId', badIds);
  const tr = filterCheckpoint(path.join(CKPT_DIR, 'transcribed.jsonl'), 'segment_id', badIds);
  const cl = filterCheckpoint(path.join(CKPT_DIR, 'cleaned.jsonl'), 'segment_id', badIds);
  const ds = purgeDataset(badIds);

  log.stat('Removed from downloaded.jsonl', dl);
  log.stat('Removed from transcribed.jsonl', tr);
  log.stat('Removed from cleaned.jsonl', cl);
  log.stat('Removed from dataset/*/metadata.jsonl', ds);
  log.stat('Next step', 'node orchestrator.js download   (re-extracts these cleanly)');
}

main();
