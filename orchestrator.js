import "dotenv/config";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { discover } from "./agents/1-discovery.js";
import { runExtract } from "./agents/2-extractor.js";
import { runCleanup } from "./agents/4-cleanup.js";
import * as checkpoint from "./utils/checkpoint.js";
import { log } from "./utils/logger.js";

const CKPT = process.env.CHECKPOINT_DIR || "./checkpoints";
const DATA = process.env.DATA_DIR || "./data";
const DSET = process.env.DATASET_DIR || "./dataset";
const MODEL = process.env.WHISPER_MODEL || "large-v3";
const DEV = process.env.DEVICE || "cuda";
const N = parseInt(process.env.TARGET_VIDEOS || "1000", 10);
const SEED = parseInt(process.env.SEED || "42", 10);

function runPythonStream(scriptPath, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn("python", [scriptPath, ...args], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    proc.stdout.on("data", (d) => process.stdout.write(d));
    proc.stderr.on("data", (d) => process.stderr.write(d));
    proc.on("close", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`Python exited with code ${code}`)),
    );
  });
}

async function phaseDiscover() {
  log.phase("Phase 1 — Discover");
  const ckptPath = path.join(CKPT, "segments.jsonl");
  if (checkpoint.count(ckptPath) > 0) {
    const segments = checkpoint.read(ckptPath);
    log.success(`Loaded ${segments.length} segments from checkpoint`);
    return segments;
  }
  const segments = discover({ dataDir: DATA, targetVideos: N, seed: SEED });
  checkpoint.write(ckptPath, segments);
  log.success(`Saved ${segments.length} segments to checkpoint`);
  return segments;
}

async function phaseDownload(segments) {
  log.phase("Phase 2 — Download & Extract");
  await runExtract(segments);
}

async function phaseTranscribe() {
  log.phase("Phase 3 — Transcribe (WhisperX)");
  const manifestPath = path.join(CKPT, "downloaded.jsonl");
  const outputPath = path.join(CKPT, "transcribed.jsonl");

  if (!fs.existsSync(manifestPath)) {
    log.error("No downloaded.jsonl found — run download phase first");
    process.exit(1);
  }

  await runPythonStream("agents/3-transcriber.py", [
    "--manifest",
    manifestPath,
    "--output",
    outputPath,
    "--model",
    MODEL,
    "--device",
    DEV,
    "--batch-size",
    "16",
  ]);
}

async function phaseCleanup() {
  log.phase("Phase 4 — Cleanup (Claude)");
  const transcribed = checkpoint.read(path.join(CKPT, "transcribed.jsonl"));
  if (!transcribed.length) {
    log.error("No transcribed.jsonl found — run transcribe phase first");
    process.exit(1);
  }
  await runCleanup(transcribed);
}

async function phasePackage() {
  log.phase("Phase 5 — Package Dataset");
  const cleanedRaw = checkpoint.read(path.join(CKPT, "cleaned.jsonl"));
  if (!cleanedRaw.length) {
    log.error("No cleaned.jsonl — run cleanup phase first");
    process.exit(1);
  }

  const bySegmentId = new Map();
  for (const entry of cleanedRaw) {
    bySegmentId.set(entry.segment_id, entry); // last write wins, dedupes append-only duplicates
  }
  const cleaned = [...bySegmentId.values()];
  log.info(
    `${cleanedRaw.length} raw cleaned entries, ${cleaned.length} unique segments`,
  );

  const splits = { train: [], val: [], test: [] };
  for (const entry of cleaned) {
    const split = entry.split || "train";
    if (splits[split]) splits[split].push(entry);
  }

  for (const [split, entries] of Object.entries(splits)) {
    const splitDir = path.join(DSET, split);
    const audioDir = path.join(splitDir, "audio");
    const metaPath = path.join(splitDir, "metadata.jsonl");
    fs.mkdirSync(audioDir, { recursive: true });

    const metaLines = [];
    for (const entry of entries) {
      const transcription = entry.clean_text || entry.raw_text || "";
      if (!transcription.trim()) continue; // VAD/ASR misfire, not a genuinely silent clip — don't train on it

      const destName = `${entry.segment_id}.wav`;
      const destPath = path.join(audioDir, destName);

      if (!fs.existsSync(destPath)) {
        // Falls back to temp/audio/segments/ only when the dataset doesn't
        // already have this file (e.g. a from-scratch download run) --
        // re-transcription workflows read audio straight from dataset/
        // and never touch temp/audio/segments/, so destPath already
        // existing is the common case here.
        const srcPath = path.join(
          process.env.AUDIO_DIR || "./temp/audio",
          "segments",
          destName,
        );
        if (!fs.existsSync(srcPath)) continue;
        fs.copyFileSync(srcPath, destPath);
      }

      const area = Array.isArray(entry.area)
        ? entry.area.join(", ")
        : entry.area || "";

      metaLines.push(
        JSON.stringify({
          file_name: `audio/${destName}`,
          transcription: entry.clean_text || entry.raw_text || "",
          segment_id: entry.segment_id,
          video_id: entry.video_id,
          area,
          caption: entry.caption || "",
        }),
      );
    }

    fs.writeFileSync(metaPath, metaLines.join("\n") + "\n");
    log.success(`${split}: ${metaLines.length} segments → ${splitDir}`);
  }

  log.stat("Dataset ready", DSET);
  log.stat("Fine-tune with", "python train.py");
}

async function main() {
  const phase = process.argv[2] || "all";
  const validPhases = ["all", "download", "transcribe", "cleanup", "package"];

  if (!validPhases.includes(phase)) {
    console.error(`Usage: node orchestrator.js [${validPhases.join("|")}]`);
    process.exit(1);
  }

  log.phase(`MakeUP AI STT Pipeline — ${phase.toUpperCase()}`);
  fs.mkdirSync(CKPT, { recursive: true });

  let segments;

  if (phase === "all" || phase === "download") {
    segments = await phaseDiscover();
    await phaseDownload(segments);
  }

  if (phase === "all" || phase === "transcribe") {
    await phaseTranscribe();
  }

  if (phase === "all" || phase === "cleanup") {
    await phaseCleanup();
  }

  if (phase === "all" || phase === "package") {
    await phasePackage();
  }

  log.phase("Pipeline complete");
}

main().catch((err) => {
  log.error(err.message);
  process.exit(1);
});
