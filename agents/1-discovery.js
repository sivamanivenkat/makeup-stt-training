import fs from 'fs';
import path from 'path';
import { log } from '../utils/logger.js';

export function seededShuffle(arr, seed) {
  let s = seed >>> 0;
  const rand = () => {
    s = Math.imul(1664525, s) + 1013904223 >>> 0;
    return s / 0x100000000;
  };
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function parseTime(t) {
  if (t == null) return 0;
  if (typeof t === 'number') return t;
  const str = String(t).trim();
  const parts = str.split(':').map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return parseFloat(str) || 0;
}

export function parseVideoEntry(line) {
  const obj = JSON.parse(line);
  const videoId = obj.video_id;
  const steps = [];

  const raw = obj.step || obj.steps || {};
  const entries = Array.isArray(raw)
    ? raw.map((s, i) => [String(i + 1), s])
    : Object.entries(raw);

  for (const [stepNum, data] of entries) {
    const start = parseTime(data.startime ?? data.start_time ?? data.start);
    const end   = parseTime(data.endtime   ?? data.end_time   ?? data.end);
    const area  = Array.isArray(data.area)   ? data.area   : [data.area   || 'face'];
    const caption = data.caption || data.description || '';
    if (end > start && caption) {
      steps.push({ stepNum, start, end, area, caption });
    }
  }

  return { videoId, steps };
}

export function loadAllVideos(dataDir) {
  const sources = [
    path.join(dataDir, 'train', 'train_steps.json'),
    path.join(dataDir, 'val',   'val_steps.json'),
    path.join(dataDir, 'test',  'test_steps.json'),
  ].filter(fs.existsSync);

  if (!sources.length) throw new Error(`No YouMakeup JSON files found in ${dataDir}`);

  const allVideos = [];
  for (const src of sources) {
    const lines = fs.readFileSync(src, 'utf8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      try {
        const v = parseVideoEntry(line);
        if (v.steps.length) allVideos.push(v);
      } catch { /* skip malformed lines */ }
    }
  }

  log.info(`Loaded ${allVideos.length} valid videos from ${sources.length} source files`);
  return allVideos;
}

export function discover({
  dataDir       = './data',
  targetVideos  = 1000,
  seed          = 42,
  ratios        = [0.70, 0.15, 0.15],
} = {}) {
  const allVideos = loadAllVideos(dataDir);
  const shuffled = seededShuffle(allVideos, seed).slice(0, targetVideos);
  const [r0, r1] = ratios;
  const t = Math.floor(shuffled.length * r0);
  const v = Math.floor(shuffled.length * (r0 + r1));

  const splitMap = new Map();
  shuffled.slice(0, t).forEach(({ videoId }) => splitMap.set(videoId, 'train'));
  shuffled.slice(t, v).forEach(({ videoId }) => splitMap.set(videoId, 'val'));
  shuffled.slice(v).forEach(({ videoId })    => splitMap.set(videoId, 'test'));

  const segments = [];
  for (const { videoId, steps } of shuffled) {
    const split = splitMap.get(videoId);
    for (const step of steps) {
      segments.push({
        segmentId: `${videoId}_step${step.stepNum}`,
        videoId,
        stepNum: step.stepNum,
        split,
        start:   step.start,
        end:     step.end,
        area:    step.area,
        caption: step.caption,
      });
    }
  }

  const counts = { train: 0, val: 0, test: 0 };
  const videoCounts = { train: t, val: v - t, test: shuffled.length - v };
  for (const s of segments) counts[s.split]++;

  log.stat('Videos sampled', shuffled.length);
  log.stat('Split (videos)', `train ${videoCounts.train} / val ${videoCounts.val} / test ${videoCounts.test}`);
  log.stat('Total segments', segments.length);
  log.stat('Split (segments)', `train ${counts.train} / val ${counts.val} / test ${counts.test}`);

  return segments;
}
