import fs from 'fs';
import path from 'path';

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

export function read(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return fs.readFileSync(filePath, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line));
}

export function readSet(filePath, keyField) {
  return new Set(read(filePath).map(e => e[keyField]));
}

export function append(filePath, obj) {
  ensureDir(filePath);
  fs.appendFileSync(filePath, JSON.stringify(obj) + '\n');
}

export function appendBatch(filePath, objs) {
  if (!objs.length) return;
  ensureDir(filePath);
  fs.appendFileSync(filePath, objs.map(o => JSON.stringify(o)).join('\n') + '\n');
}

export function write(filePath, objs) {
  ensureDir(filePath);
  fs.writeFileSync(filePath, objs.map(o => JSON.stringify(o)).join('\n') + '\n');
}

export function count(filePath) {
  if (!fs.existsSync(filePath)) return 0;
  return fs.readFileSync(filePath, 'utf8').split('\n').filter(Boolean).length;
}
