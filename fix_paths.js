const fs = require("fs");
const lines = fs
  .readFileSync("checkpoints/downloaded.jsonl.bak", "utf8")
  .split("\n")
  .filter(Boolean);
const out =
  lines
    .map((l) => {
      const obj = JSON.parse(l);
      if (obj.wavPath) obj.wavPath = obj.wavPath.split("\\").join("/");
      return JSON.stringify(obj);
    })
    .join("\n") + "\n";
fs.writeFileSync("checkpoints/downloaded.jsonl", out);
console.log("done, lines:", lines.length);
