import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const preferredPort = Number(process.env.PORT || 4173);
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.png': 'image/png' };
const MAX_BODY_BYTES = 1_000_000;
// Only these files are served; .env, server.mjs and trace-live.json stay private.
const isPublicPath = pathname => pathname === '/index.html' || /^\/figures\/[\w-]+\.png$/.test(pathname);

// Load a local .env file without adding a runtime dependency.
try {
  const envText = await readFile(join(root, '.env'), 'utf8');
  for (const line of envText.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (match && !process.env[match[1]]) process.env[match[1]] = match[2].replace(/^['"]|['"]$/g, '');
  }
} catch {}

class HttpError extends Error { constructor(status, message) { super(message); this.status = status; } }
function readBody(req) {
  return new Promise((resolve, reject) => {
    if (Number(req.headers['content-length']) > MAX_BODY_BYTES) { req.resume(); return reject(new HttpError(413, 'Request body too large')); }
    const chunks = []; let size = 0;
    req.on('data', chunk => {
      size += chunk.length;
      if (size <= MAX_BODY_BYTES) return chunks.push(chunk);
      req.removeAllListeners('data'); req.resume(); reject(new HttpError(413, 'Request body too large'));
    });
    req.on('end', () => { if (size > MAX_BODY_BYTES) return; try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); } catch { reject(new HttpError(400, 'Invalid JSON body')); } });
    req.on('error', reject);
  });
}
// Block cross-site form/fetch POSTs: require JSON and a same-origin, loopback request.
function checkPost(req) {
  const host = req.headers.host || '';
  if (!/^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(host)) throw new HttpError(403, 'Forbidden host');
  if (req.headers.origin && req.headers.origin !== `http://${host}`) throw new HttpError(403, 'Cross-origin request rejected');
  const type = (req.headers['content-type'] || '').split(';')[0].trim().toLowerCase();
  if (type !== 'application/json') throw new HttpError(415, 'Content-Type must be application/json');
}
async function callJev(payload, res) {
  if (!process.env.TYPESAFE_API_KEY) return send(res, 503, { error: 'TYPESAFE_API_KEY is not configured on the server.' });
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const upstream = await fetch('https://api.typesafe.ai/v1/systemone', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.TYPESAFE_API_KEY}` }, body: JSON.stringify(payload) });
      const text = await upstream.text(); res.writeHead(upstream.status, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(text); return;
    } catch { if (attempt === 1) return send(res, 502, { error: 'TypeSafe request failed.' }); await new Promise(resolve => setTimeout(resolve, 250)); }
  }
}
async function readState(req) {
  const state = await readBody(req);
  if (!state || typeof state !== 'object' || Array.isArray(state)) throw new HttpError(400, 'Expected a JSON object');
  return state;
}
async function coach(req, res) {
  const state = await readState(req);
  const payload = { state, model: 'jev-latest', questions: { pace: { type: 'choice', instructions: 'Given this current Snake game state, what pace advice should the player follow next?', criteria: { calm: 'The player is early in the game or the board position needs safer space. Recommend a careful pace.', steady: 'The player has a stable position and should maintain a balanced pace with two-step planning.', sprint: 'The player has enough length or momentum to safely move faster toward the food.' } } } };
  return callJev(payload, res);
}
async function move(req, res) {
  const state = await readState(req);
  if (state.legalMoves !== undefined && (!Array.isArray(state.legalMoves) || !state.legalMoves.every(item => typeof item === 'string'))) throw new HttpError(400, 'legalMoves must be an array of strings');
  const legal = Array.isArray(state.legalMoves) && state.legalMoves.length ? state.legalMoves : ['up', 'down', 'left', 'right'];
  const planned = state.plannedDirection || legal[0];
  const criteria = Object.fromEntries(legal.map(move => [move, move === planned ? `Move ${move}. This is the safest known route toward the food.` : `Move ${move} only if it clearly avoids a loop or collision.`]));
  const loopNote = state.loopDetected ? ` The recent path is looping. Break the loop now; prefer plannedDirection (${planned}) unless it causes immediate danger.` : '';
  const payload = { state, model: 'jev-latest', questions: { move: { type: 'choice', instructions: `Choose the single next direction for the snake. Only choose a direction from legalMoves. Preserve survival, avoid repeating recentHeads, and move toward the food.${loopNote}`, criteria }, trap: { type: 'noul', instructions: 'Is the snake currently in a dangerous local situation where the obvious food-seeking move is likely to create a collision or repeated loop?', criteria: { true: 'The current position is risky or repetitive and needs a safer route.', false: 'The current position has a clear safe continuation.' } } } };
  return callJev(payload, res);
}
function send(res, status, data) { res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(data)); }
const server = createServer(async (req, res) => {
  try {
    if (req.method === 'POST') checkPost(req);
    if (req.method === 'POST' && req.url === '/api/coach') return await coach(req, res);
    if (req.method === 'POST' && req.url === '/api/move') return await move(req, res);
    if (req.method === 'POST' && req.url === '/api/trace') { const body = await readBody(req); await writeFile(join(root, 'trace-live.json'), JSON.stringify(body)); return send(res, 200, { saved: true }); }
    if (req.method === 'GET' && req.url === '/api/status') return process.env.TYPESAFE_API_KEY ? send(res, 200, { configured: true }) : send(res, 503, { configured: false });
    if (req.method !== 'GET') return send(res, 405, { error: 'Method not allowed' });
    const pathname = req.url === '/' ? '/index.html' : req.url.split('?')[0];
    if (!isPublicPath(pathname)) return send(res, 404, { error: 'Not found' });
    const file = join(root, pathname);
    const content = await readFile(file); res.writeHead(200, { 'Content-Type': mime[extname(file)] || 'application/octet-stream' }); res.end(content);
  } catch (error) {
    if (res.headersSent) return res.end();
    if (error instanceof HttpError) return send(res, error.status, { error: error.message });
    send(res, 404, { error: 'Not found' });
  }
});

function listenOn(port) {
  server.once('error', error => {
    if (error.code === 'EADDRINUSE') {
      const nextPort = port + 1;
      console.warn(`Port ${port} is already in use. Trying ${nextPort}...`);
      return listenOn(nextPort);
    }
    throw error;
  });
  server.listen(port, '127.0.0.1', () => console.log(`Gridloop running at http://localhost:${port}`));
}

listenOn(preferredPort);
