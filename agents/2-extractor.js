import { execSync, spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import pLimit from 'p-limit';
import { log } from '../utils/logger.js';
import { sleep } from '../utils/retry.js';
import * as checkpoint from '../utils/checkpoint.js';

const AUDIO_DIR    = process.env.AUDIO_DIR    || './temp/audio';
const CKPT_DIR     = process.env.CHECKPOINT_DIR || './checkpoints';
const WORKERS      = parseInt(process.env.WORKERS || '2', 10);
const SLEEP_MS     = 2500;

// yt-dlp now requires a JS runtime to solve YouTube's signature/n
// challenges; without one, downloads silently degrade or fail format
// selection. deno.exe ships alongside the pipeline (winget install
// DenoLand.Deno) since it isn't reliably on PATH on every machine.
const DENO_PATH = process.env.DENO_PATH || './deno.exe';

// Whisper's feature extractor silently truncates/pads audio to a fixed
// 30s window regardless of true length. Segments longer than that get
// their tail dropped at train/inference time while the full-length label
// stays in metadata, corrupting supervision. Cap extraction duration here
// so raw_text/clean_text downstream always describe only the audible span.
const MAX_SEGMENT_SECONDS = 29.8;

// A crash, disk-full event, or killed process can leave a 0-byte or
// header-only stub at outPath. Without this check, existsSync() alone
// would trust that stub forever on every future run. 8000 bytes is well
// under the smallest real segment (~1s of 16kHz mono s16 = ~32KB) but
// well above an empty/near-empty WAV header (~44-78 bytes).
const MIN_VALID_BYTES = 8000;

// 16kHz mono s16 = 32000 bytes/sec. Segment files predating the
// MAX_SEGMENT_SECONDS cap (or written before it existed) can be far longer
// than 30s on disk — existsSync()-only checks would trust those forever.
const BYTES_PER_SEC = 32000;
const MAX_SEGMENT_BYTES = Math.ceil(MAX_SEGMENT_SECONDS * BYTES_PER_SEC) + 100;

function isValidAudioFile(p) {
  try {
    return fs.statSync(p).size >= MIN_VALID_BYTES;
  } catch {
    return false;
  }
}

function isValidSegmentFile(p) {
  try {
    const size = fs.statSync(p).size;
    return size >= MIN_VALID_BYTES && size <= MAX_SEGMENT_BYTES;
  } catch {
    return false;
  }
}

function ytUrl(videoId) {
  return `https://www.youtube.com/watch?v=${videoId}`;
}

function rawAudioPath(videoId) {
  return path.join(AUDIO_DIR, 'raw', `${videoId}.wav`);
}

function segmentPath(segmentId) {
  return path.join(AUDIO_DIR, 'segments', `${segmentId}.wav`);
}

function downloadVideo(videoId) {
  const outPath = rawAudioPath(videoId);
  if (fs.existsSync(outPath)) {
    if (isValidAudioFile(outPath)) return outPath;
    fs.unlinkSync(outPath);
  }

  fs.mkdirSync(path.dirname(outPath), { recursive: true });

  const result = spawnSync('yt-dlp', [
    '--js-runtimes', `deno:${DENO_PATH}`,
    '--remote-components', 'ejs:github',
    '--cookies-from-browser', 'firefox',
    '-x', '--audio-format', 'wav',
    '--audio-quality', '0',
    '--no-playlist',
    '-o', outPath.replace('.wav', '.%(ext)s'),
    ytUrl(videoId),
  ], { timeout: 120_000, encoding: 'utf8' });

  if (result.status !== 0) {
    try { fs.unlinkSync(outPath); } catch {}
    try { fs.unlinkSync(outPath.replace('.wav', '.m4a')); } catch {}
    try { fs.unlinkSync(outPath.replace('.wav', '.webm')); } catch {}
    const err = result.stderr || result.stdout || '';
    if (err.includes('removed') || err.includes('unavailable') || err.includes('private')) {
      throw new Error('VIDEO_UNAVAILABLE');
    }
    throw new Error(`yt-dlp exited ${result.status}: ${err.slice(0, 200)}`);
  }

  const possible = [outPath, outPath.replace('.wav', '.m4a'), outPath.replace('.wav', '.webm')];
  const found = possible.find(fs.existsSync);
  if (!found) throw new Error('yt-dlp produced no output file');

  if (found !== outPath) {
    spawnSync('ffmpeg', ['-y', '-i', found, '-ar', '16000', '-ac', '1', outPath], { timeout: 120_000 });
    fs.unlinkSync(found);
  }

  if (!isValidAudioFile(outPath)) {
    try { fs.unlinkSync(outPath); } catch {}
    throw new Error('yt-dlp produced an empty/truncated output file');
  }

  return outPath;
}

function trimSegment(rawPath, { segmentId, start, end }) {
  const outPath = segmentPath(segmentId);
  if (fs.existsSync(outPath)) {
    if (isValidSegmentFile(outPath)) return outPath;
    fs.unlinkSync(outPath); // stale/oversized — predates MAX_SEGMENT_SECONDS cap
  }

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const duration = Math.min(end - start, MAX_SEGMENT_SECONDS);
  if (duration < 1) throw new Error('Segment too short (<1s)');

  const result = spawnSync('ffmpeg', [
    '-y', '-i', rawPath,
    '-ss', String(start),
    '-t',  String(duration),
    '-ar', '16000',
    '-ac', '1',
    '-sample_fmt', 's16',
    outPath,
  ], { timeout: 60_000 });

  if (result.status !== 0) {
    try { fs.unlinkSync(outPath); } catch {}
    const errText = (result.stderr || result.stdout || '').toString().trim();
    throw new Error(`ffmpeg failed: ${errText.slice(-300)}`);
  }

  if (!isValidAudioFile(outPath)) {
    try { fs.unlinkSync(outPath); } catch {}
    throw new Error('ffmpeg produced an empty/truncated output file');
  }

  return outPath;
}

async function processVideo(videoId, segments) {
  let rawPath;
  try {
    rawPath = downloadVideo(videoId);
    await sleep(SLEEP_MS + Math.random() * 1000);
  } catch (err) {
    const reason = err.message === 'VIDEO_UNAVAILABLE' ? 'unavailable' : err.message;
    for (const seg of segments) {
      checkpoint.append(path.join(CKPT_DIR, 'failed.jsonl'), { segmentId: seg.segmentId, videoId, reason });
    }
    log.warn(`[download] ${videoId} — ${reason}`);
    return { ok: 0, failed: segments.length };
  }

  let ok = 0, failed = 0;
  for (const seg of segments) {
    try {
      const wavPath = trimSegment(rawPath, seg);
      checkpoint.append(path.join(CKPT_DIR, 'downloaded.jsonl'), { ...seg, wavPath });
      ok++;
    } catch (err) {
      checkpoint.append(path.join(CKPT_DIR, 'failed.jsonl'), { segmentId: seg.segmentId, videoId, reason: err.message });
      log.warn(`[trim] ${seg.segmentId} — ${err.message}`);
      failed++;
    }
  }

  try { fs.unlinkSync(rawPath); } catch { /* best effort */ }
  return { ok, failed };
}

export async function runExtract(segments) {
  const doneIds = checkpoint.readSet(path.join(CKPT_DIR, 'downloaded.jsonl'), 'segmentId');
  const failedIds = checkpoint.readSet(path.join(CKPT_DIR, 'failed.jsonl'), 'segmentId');

  // Checkpoint says "done", but the on-disk file may predate the duration
  // cap — re-verify rather than trusting the checkpoint entry blindly.
  const reclaimedIds = new Set();
  for (const id of doneIds) {
    if (!isValidSegmentFile(segmentPath(id))) {
      doneIds.delete(id);
      reclaimedIds.add(id);
    }
  }
  if (reclaimedIds.size) log.warn(`${reclaimedIds.size} segments marked done but oversized on disk — re-queuing`);

  // failed.jsonl is shared with later phases (e.g. cleanup writes its own
  // failures here under the same segmentId key) — a reclaimed segment already
  // succeeded extraction once, so an entry there is not an extraction failure
  // and must not block re-extraction.
  const todo = segments.filter(s =>
    !doneIds.has(s.segmentId) &&
    (reclaimedIds.has(s.segmentId) || !failedIds.has(s.segmentId))
  );
  log.info(`Extracting ${todo.length} segments (${doneIds.size} already done, ${failedIds.size} permanently failed)`);
  if (!todo.length) { log.success('All segments already extracted'); return; }

  const byVideo = new Map();
  for (const seg of todo) {
    if (!byVideo.has(seg.videoId)) byVideo.set(seg.videoId, []);
    byVideo.get(seg.videoId).push(seg);
  }

  const limit = pLimit(WORKERS);
  let totalOk = 0, totalFailed = 0;
  const videoList = [...byVideo.entries()];

  await Promise.all(
    videoList.map(([videoId, segs], i) =>
      limit(async () => {
        const { ok, failed } = await processVideo(videoId, segs);
        totalOk += ok;
        totalFailed += failed;
        if ((i + 1) % 10 === 0 || i === videoList.length - 1) {
          log.info(`Progress: ${i + 1}/${videoList.length} videos — ${totalOk} segments OK, ${totalFailed} failed`);
        }
      })
    )
  );

  log.stat('Extraction complete', `${totalOk} OK / ${totalFailed} failed`);
}
