import "dotenv/config";
import path from "path";
import { loadAllVideos, seededShuffle } from "./agents/1-discovery.js";
import * as checkpoint from "./utils/checkpoint.js";
import { log } from "./utils/logger.js";

const CKPT = process.env.CHECKPOINT_DIR || "./checkpoints";
const DATA = process.env.DATA_DIR || "./data";
const SEED = parseInt(process.env.SEED || "42", 10);
const ADD = parseInt(process.argv[2] || process.env.EXPAND_VIDEOS || "800", 10);

function main() {
  const segPath = path.join(CKPT, "segments.jsonl");
  const existing = checkpoint.read(segPath);
  const seenIds = new Set(existing.map((s) => s.videoId));

  const allVideos = loadAllVideos(DATA);
  const shuffled = seededShuffle(allVideos, SEED);
  const candidates = shuffled.filter((v) => !seenIds.has(v.videoId));

  if (!candidates.length) {
    log.error("No unsampled videos remain");
    process.exit(1);
  }

  const picked = candidates.slice(0, ADD);
  if (picked.length < ADD) {
    log.warn(
      `Only ${picked.length} unsampled videos available (requested ${ADD})`,
    );
  }

  const newSegments = [];
  for (const { videoId, steps } of picked) {
    for (const step of steps) {
      newSegments.push({
        segmentId: `${videoId}_step${step.stepNum}`,
        videoId,
        stepNum: step.stepNum,
        split: "train",
        start: step.start,
        end: step.end,
        area: step.area,
        caption: step.caption,
      });
    }
  }

  checkpoint.appendBatch(segPath, newSegments);

  log.stat("Already sampled videos", seenIds.size);
  log.stat("New videos added", picked.length);
  log.stat("New segments added", newSegments.length);
  log.stat("All added as split", "train (val/test left untouched)");
  log.stat("Next step", "node orchestrator.js download");
}

main();
