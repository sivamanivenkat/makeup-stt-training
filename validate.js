import "dotenv/config";
import fs from "fs";
import path from "path";
import { log } from "./utils/logger.js";

const DSET = process.env.DATASET_DIR || "./dataset";
const CKPT = process.env.CHECKPOINT_DIR || "./checkpoints";

function checkSplit(split) {
  const metaPath = path.join(DSET, split, "metadata.jsonl");
  if (!fs.existsSync(metaPath)) {
    log.error(`Missing: ${metaPath}`);
    return { total: 0, ok: 0, missingAudio: 0, emptyText: 0 };
  }

  const lines = fs
    .readFileSync(metaPath, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean);
  let ok = 0,
    missingAudio = 0,
    emptyText = 0;

  for (const line of lines) {
    const entry = JSON.parse(line);
    const audioPath = path.join(DSET, split, entry.file_name);
    const hasAudio = fs.existsSync(audioPath);
    const hasText = (entry.transcription || "").trim().length > 0;

    if (!hasAudio) {
      missingAudio++;
      log.warn(`Missing audio: ${audioPath}`);
    }
    if (!hasText) {
      emptyText++;
      log.warn(`Empty transcript: ${entry.segment_id}`);
    }
    if (hasAudio && hasText) ok++;
  }

  return { total: lines.length, ok, missingAudio, emptyText };
}

log.phase("Dataset Validation");

let grandTotal = 0,
  grandOk = 0;
for (const split of ["train", "val", "test"]) {
  const r = checkSplit(split);
  grandTotal += r.total;
  grandOk += r.ok;
  log.stat(
    split,
    `${r.ok}/${r.total} OK — ${r.missingAudio} missing audio, ${r.emptyText} empty text`,
  );
}

log.stat("Total", `${grandOk}/${grandTotal} segments ready for training`);

const failedPath = path.join(CKPT, "failed.jsonl");
if (fs.existsSync(failedPath)) {
  const failed = fs
    .readFileSync(failedPath, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean).length;
  log.stat("Permanently failed", `${failed} segments (see ${failedPath})`);
}

if (grandOk === grandTotal) {
  log.success("All segments valid — ready to run: python train.py");
} else {
  log.warn(
    `${grandTotal - grandOk} segments have issues — check warnings above`,
  );
}
