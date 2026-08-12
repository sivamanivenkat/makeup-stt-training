import OpenAI from 'openai';
import pLimit from 'p-limit';
import path from 'path';
import { log } from '../utils/logger.js';
import { withRetry } from '../utils/retry.js';
import * as checkpoint from '../utils/checkpoint.js';

const client = new OpenAI();
const CKPT_DIR = process.env.CHECKPOINT_DIR || './checkpoints';
const CONCURRENCY = 5;
const MODEL = process.env.OPENAI_CLEANUP_MODEL || 'gpt-4o-mini';

const SYSTEM_PROMPT = `You are a makeup tutorial speech-to-text correction assistant.
You receive a raw ASR transcript from a makeup tutorial video segment and return the corrected version.

Fix these types of errors:
- Makeup product names: foundation, concealer, contour, highlight, blush, bronzer, eyeshadow, eyeliner, mascara, lipstick, primer, setting powder, setting spray, toner, serum, BB cream, CC cream
- Tool names: brush, sponge, beauty blender, blending brush, flat brush, fan brush, kabuki brush, spoolie, liner brush
- Techniques: blending, buffing, contouring, strobing, baking, cut crease, halo eye, smoky eye, stippling, dabbing, patting, feathering
- Facial areas: lid, crease, brow bone, waterline, lash line, cupid's bow, philtrum, temple, apple of the cheeks, bridge of nose, T-zone, jawline, inner corner, outer corner
- Brand names: MAC, Fenty, NARS, Urban Decay, Charlotte Tilbury, NYX, Maybelline, L'Oreal, Too Faced, ABH (Anastasia Beverly Hills)

Common ASR errors to fix:
- "eye lid" → "eyelid"  |  "high lighter" → "highlighter"  |  "con tour" → "contour"
- "eye liner" → "eyeliner"  |  "water line" → "waterline"  |  "beauty blender" → "beauty blender"
- "brow bone" → "brow bone"  |  "eye shadow" → "eyeshadow"  |  "lip stick" → "lipstick"
- "under eye" → "under-eye"  |  "lash line" → "lash line"  |  "setting spray" → "setting spray"

Rules:
1. Preserve the original sentence structure and meaning exactly
2. Only fix clear ASR errors — do not rephrase or add information
3. If the transcript is empty or unintelligible, return an empty string
4. Return ONLY the corrected transcript text with no commentary or explanation`;

async function cleanOne(entry) {
  if (!entry.raw_text?.trim()) return '';

  const userPrompt = `Step context: ${entry.caption}
Facial area: ${(entry.area || []).join(', ')}

Raw ASR transcript:
${entry.raw_text}`;

  const msg = await client.chat.completions.create({
    model: MODEL,
    max_tokens: 512,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userPrompt },
    ],
  });

  return msg.choices[0]?.message?.content?.trim() || '';
}

export async function runCleanup(transcribed) {
  const doneIds = checkpoint.readSet(path.join(CKPT_DIR, 'cleaned.jsonl'), 'segment_id');
  const todo = transcribed.filter(e => !doneIds.has(e.segment_id));

  log.info(`Cleaning ${todo.length} segments (${doneIds.size} already done)`);
  if (!todo.length) { log.success('All segments already cleaned'); return; }

  const limit = pLimit(CONCURRENCY);
  let ok = 0, failed = 0;

  await Promise.all(
    todo.map((entry, i) =>
      limit(async () => {
        try {
          const clean_text = await withRetry(
            () => cleanOne(entry),
            { attempts: 4, baseDelay: 3000, label: entry.segment_id }
          );

          checkpoint.append(path.join(CKPT_DIR, 'cleaned.jsonl'), {
            segment_id: entry.segment_id,
            video_id:   entry.video_id,
            step_num:   entry.step_num,
            split:      entry.split,
            wav_path:   entry.wav_path,
            area:       entry.area,
            caption:    entry.caption,
            raw_text:   entry.raw_text,
            clean_text,
          });
          ok++;
        } catch (err) {
          log.error(`[cleanup] ${entry.segment_id}: ${err.message}`);
          checkpoint.append(path.join(CKPT_DIR, 'failed.jsonl'), {
            segmentId: entry.segment_id, reason: `cleanup: ${err.message}`
          });
          failed++;
        }

        if ((i + 1) % 100 === 0 || i === todo.length - 1) {
          log.info(`Cleanup progress: ${i + 1}/${todo.length} — ${ok} OK / ${failed} failed`);
        }
      })
    )
  );

  log.stat('Cleanup complete', `${ok} cleaned / ${failed} failed`);
}
