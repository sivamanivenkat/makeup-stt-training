# Operations & Environment Notes

Things that broke, why, and how they were fixed — not covered by the design docs
(`ingestion.md`, `curation.md`, `training.md`, `future_adaptation.md`), which describe how
the pipeline is _supposed_ to work but not what it actually takes to run it on real
infrastructure.

---

## 1. yt-dlp environment requirements

### JS runtime (deno) — required, not optional

yt-dlp now requires a JS runtime to solve YouTube's signature/n-challenge. Without one,
downloads silently degrade or fail format selection with `HTTP Error 403: Forbidden`, even
for videos with no bot-check/login issue at all.

- Install: `winget install --id DenoLand.Deno` (or ship `deno.exe` alongside the project —
  `agents/2-extractor.js` looks for it at `DENO_PATH`, default `./deno.exe`).
- Wired into `agents/2-extractor.js` via:
  ```
  --js-runtimes deno:<DENO_PATH>
  --remote-components ejs:github
  ```
- Windows gotcha: `winget install` updates the registry PATH but a long-lived shell won't
  pick it up without a fresh process. Use an absolute path to `deno.exe` if `deno` isn't
  resolving.

### YouTube authentication — a real login is required

`--cookies-from-browser` only helps if the browser actually has an authenticated Google
session. An anonymous cookie export (23 cookies, zero `.google.com` domain entries, zero
`SID`/`SAPISID`/`LOGIN_INFO`) looks superficially fine to yt-dlp but produces
`Sign in to confirm you're not a bot` / `LOGIN_REQUIRED` on any player client
(`android_vr`, `web_safari`, `web`, `tv` all tried, all failed identically).

- **Fix:** log into a real Google account on youtube.com in the browser yt-dlp reads
  cookies from. No client-side flag fixes this — it was never a bot-detection or
  PO-token problem.
- **Verify a browser actually has a real session before troubleshooting anything else:**
  ```
  yt-dlp --cookies-from-browser firefox --cookies /tmp/cookies.txt --skip-download <url>
  grep -iE 'SID|SSID|SAPISID|LOGIN_INFO|APISID|HSID' /tmp/cookies.txt
  ```
  Zero matches = not actually logged in, regardless of what the browser UI shows.
- **Firefox is the working cookie source on this machine.** Chrome cookies fail with
  `Failed to decrypt with DPAPI` (Chrome's App-Bound Encryption, Chrome 127+) even with
  `pycryptodomex` installed and even outside any tool sandbox — fixing it requires disabling
  the `ApplicationBoundEncryptionEnabled` policy under
  `HKCU:\Software\Policies\Google\Chrome`, which needs admin rights not available in this
  environment. If Chrome cookies are ever needed again, that registry change has to be run
  by hand, as Administrator, followed by one Chrome relaunch+close to let it re-encrypt
  under legacy DPAPI.

### PO-Token provider (bgutil-ytdlp-pot-provider)

Installed as a defense-in-depth measure but turned out not to be the actual fix for the
bot-check wall (real login was). Still useful to have wired up:

- Cloned to yt-dlp's expected default path: `C:\Users\user\bgutil-ytdlp-pot-provider`
- Uses `script-deno` provider mode — runs `server/src/generate_once.ts` directly via deno,
  **no build step** (there is no `npm run build` script, despite what you'd expect —
  `server/package.json` only has `lint`/`lint-fix`/`format`).
- First invocation is slow (deno downloads npm-imported type packages) and can exceed
  yt-dlp's internal 15s timeout on a cold cache. Pre-warm once manually:
  ```
  deno run <flags> server/src/generate_once.ts --version
  ```
  before relying on it inside a yt-dlp run.

---

## 2. Transcription — Colab, not local

`curation.md` describes WhisperX running on `cuda` locally. In practice, local hardware
can't do this:

- GPU: NVIDIA MX130, 2GB VRAM. `large-v3` needs ~10GB.
- Installed torch is a CPU-only build (`2.8.0+cpu`, `torch.version.cuda is None`) even
  though the GPU driver itself is fine (`nvidia-smi` reports CUDA 13.0 driver support).

**Actual workflow:**

1. Finish local download/extraction (`agents/2-extractor.js`) — CPU-only work, fine locally.
2. Upload `temp/audio/segments/` to Google Drive.
3. Run `agents/3-transcriber.py` in Colab against the uploaded audio (same script, same
   `checkpoints/transcribed.jsonl` output format — drops straight back into this pipeline
   with no format changes needed).
4. Pull `transcribed.jsonl` back down, resume locally with `node orchestrator.js cleanup`
   then `node orchestrator.js package` (neither needs GPU — cleanup calls OpenAI's API,
   package is file I/O).

---

## 3. Known bugs / open issues

- **`checkpoints/failed.jsonl` mixes download-phase and cleanup-phase failures** under one
  file, deduped by `segmentId` with no reason-prefix distinction in some code paths.
  `retry_failed.js` correctly separates `cleanup:`-prefixed entries from download failures,
  but `runExtract()`'s own `doneIds`/`failedIds` filtering in `agents/2-extractor.js` reads
  all of `failed.jsonl` regardless of reason — worth checking whether this ever incorrectly
  blocks a download retry for a segment that only ever failed at the cleanup stage.
- **Duplicate `segmentId` entries** in checkpoint files: raw `wc -l` line counts diverge
  significantly from deduped-by-segmentId counts in both `downloaded.jsonl` and
  `failed.jsonl` (e.g. 9489 raw vs 7998 unique seen in one snapshot) — likely from repeated
  pipeline runs re-logging the same segment across attempts. Not root-caused.
- **`HTTP Error 416: Requested range not satisfiable`** — a recurring yt-dlp failure mode
  during the 2026-07-16 full-corpus download run, distinct from both the bot-check and
  JS-runtime issues. Not yet investigated; minority of failures so far (~7% of video-level
  failures in the first 300-video sample).
- **`[WinError 32] The process cannot access the file because it is being used by another
process`** — intermittent file-lock error on `temp/audio/raw/*.wav`, likely a race between
  ffmpeg finishing and the next step trying to read/delete the same raw file. Not yet fixed.
- **`TARGET_VIDEOS=1000` in `.env` is stale.** `checkpoints/segments.jsonl` already contains
  the full YouMakeup corpus (1960 videos / 20910 segments) from an earlier discovery run —
  the env var no longer reflects what's actually queued and re-running discovery with it
  as-is would _shrink_ the dataset. Don't rerun `agents/1-discovery.js` without first raising
  or removing this cap.

---

## 4. Tool versions in use

Pin/record these — this stack breaks on upstream changes fast (yt-dlp added a hard
JS-runtime requirement between sessions with no prior warning):

| Tool          | Version                                                           |
| ------------- | ----------------------------------------------------------------- |
| yt-dlp        | 2026.07.04                                                        |
| torch         | 2.8.0+cpu (no CUDA build installed locally)                       |
| deno          | 1.3.1                                                             |
| pywin32       | 312                                                               |
| pycryptodomex | 3.23.0                                                            |
| GPU driver    | NVIDIA 581.83 (CUDA 13.0 capable, but see local-GPU limits above) |

Re-check this table when anything YouTube-download-related breaks again — a yt-dlp/deno
version bump is the first thing to suspect.

---

## 5. Data licensing / ToS — needs review

This pipeline downloads audio from ~1960 YouTube videos at scale via the YouMakeup dataset's
video ID annotations, for the purpose of training data. Two things worth explicitly
verifying before this dataset or a resulting model is shared or published outside personal/
research use:

- **YouMakeup dataset's own license terms** — what it actually grants (research-only?
  attribution required? redistribution of derived audio permitted?).
- **YouTube's Terms of Service** re: bulk downloading via yt-dlp — generally against ToS for
  anything beyond personal/fair-use research contexts; relevant if this ever moves toward
  a distributed or commercial dataset/model.

Not blocking for local experimentation, but flag before any external release.

---

## 6. Cost tracking

Cleanup phase (`agents/4-cleanup.js`) calls OpenAI's API (`gpt-4o-mini` by default) once per
segment. At full-corpus scale (~20910 segments, realistically ~14-17k after download
failures), this is a non-trivial number of API calls. No running cost tally exists yet —
worth checking OpenAI usage dashboard after the first full `cleanup` phase run and logging
the actual $ figure here for future budget planning.

| Date                          | Segments cleaned | Model | Approx cost |
| ----------------------------- | ---------------- | ----- | ----------- |
| _(not yet run at full scale)_ |                  |       |             |
