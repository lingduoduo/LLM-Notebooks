import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { request } from 'node:http';
import { cp, mkdtemp, rm, writeFile, access } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

// These tests copy each game into a temp dir with a dummy .env, so no real key is
// read and every asserted request is rejected before any TypeSafe call.
const gamesRoot = fileURLToPath(new URL('../games/', import.meta.url));
const DUMMY_KEY = 'dummy-test-key-should-never-leak';
const games = [
  { name: 'gridloop', postPath: '/api/move' },
  { name: 'minesweeper', postPath: '/api/move' },
  { name: 'werewolf', postPath: '/api/game/new' }
];

async function startServer(name) {
  const dir = await mkdtemp(join(tmpdir(), `jev-${name}-`));
  await cp(join(gamesRoot, name), dir, { recursive: true, filter: source => !basename(source).startsWith('.env') && !source.includes(`${name}/trace/`) });
  await writeFile(join(dir, '.env'), `TYPESAFE_API_KEY=${DUMMY_KEY}\n`);
  const env = { ...process.env, PORT: String(20000 + Math.floor(Math.random() * 30000)) };
  delete env.TYPESAFE_API_KEY;
  delete env.DEEPSEEK_API_KEY;
  const child = spawn(process.execPath, ['server.mjs'], { cwd: dir, env, stdio: ['ignore', 'pipe', 'pipe'] });
  const port = await new Promise((resolve, reject) => {
    let output = '';
    const timer = setTimeout(() => reject(new Error(`${name} did not start: ${output}`)), 10_000);
    const onData = chunk => {
      output += chunk;
      const match = output.match(/running at http:\/\/localhost:(\d+)/);
      if (match) { clearTimeout(timer); resolve(Number(match[1])); }
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', chunk => { output += chunk; });
    child.on('exit', code => { clearTimeout(timer); reject(new Error(`${name} exited with ${code}: ${output}`)); });
  });
  return { dir, port, child, async stop() { child.kill(); await rm(dir, { recursive: true, force: true }); } };
}

function send(port, { method = 'GET', path = '/', headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const req = request({ host: '127.0.0.1', port, method, path, headers: { Host: `localhost:${port}`, ...headers } }, res => {
      let text = '';
      res.setEncoding('utf8');
      res.on('data', chunk => { text += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body: text }));
    });
    // The server may answer (413) and stop reading before an oversized upload finishes.
    req.on('error', error => (['EPIPE', 'ECONNRESET'].includes(error.code) ? null : reject(error)));
    req.end(body);
  });
}

const json = { 'Content-Type': 'application/json' };

for (const { name, postPath } of games) {
  test(`${name} server rejects unsafe requests offline`, async t => {
    const server = await startServer(name);
    t.after(() => server.stop());
    const { port } = server;

    const index = await send(port, { path: '/' });
    assert.equal(index.status, 200);

    for (const path of ['/.env', '/server.mjs', '/trace-live.json', '/figures/../.env', '/%2e%2e/.env', '/.env.example']) {
      const response = await send(port, { path });
      assert.equal(response.status, 404, path);
      assert.ok(!response.body.includes(DUMMY_KEY), path);
    }

    const invalid = await send(port, { method: 'POST', path: postPath, headers: json, body: 'notjson' });
    assert.equal(invalid.status, 400);
    const status = await send(port, { path: '/api/status' });
    assert.equal(status.status, 200, 'server still alive after bad JSON');

    const oversized = await send(port, { method: 'POST', path: postPath, headers: json, body: JSON.stringify({ pad: 'x'.repeat(2_000_000) }) });
    assert.equal(oversized.status, 413);

    const chunked = await send(port, { method: 'POST', path: postPath, headers: { ...json, 'Transfer-Encoding': 'chunked' }, body: JSON.stringify({ pad: 'x'.repeat(2_000_000) }) });
    assert.equal(chunked.status, 413);

    const crossOrigin = await send(port, { method: 'POST', path: postPath, headers: { ...json, Origin: 'http://evil.example' }, body: '{}' });
    assert.equal(crossOrigin.status, 403);

    const rebound = await send(port, { method: 'POST', path: postPath, headers: { ...json, Host: 'evil.example', Origin: 'http://evil.example' }, body: '{}' });
    assert.equal(rebound.status, 403);

    const textPlain = await send(port, { method: 'POST', path: postPath, headers: { 'Content-Type': 'text/plain' }, body: '{}' });
    assert.equal(textPlain.status, 415);

    const traceAttempt = await send(port, { method: 'POST', path: '/api/trace', headers: { 'Content-Type': 'text/plain', Origin: 'http://evil.example' }, body: '[]' });
    assert.equal(traceAttempt.status, 403);
    await assert.rejects(access(join(server.dir, 'trace-live.json')));

    if (name === 'minesweeper') {
      const badActions = await send(port, { method: 'POST', path: postPath, headers: json, body: '{"actions":[1]}' });
      assert.equal(badActions.status, 400);
    }
    if (name === 'gridloop') {
      const badMoves = await send(port, { method: 'POST', path: postPath, headers: json, body: '{"legalMoves":[1]}' });
      assert.equal(badMoves.status, 400);
      const sameOrigin = await send(port, { method: 'POST', path: '/api/trace', headers: { ...json, Origin: `http://localhost:${port}` }, body: '[]' });
      assert.equal(sameOrigin.status, 200);
    }

    const alive = await send(port, { path: '/api/status' });
    assert.equal(alive.status, 200);
    assert.equal(server.child.exitCode, null);
  });
}
