import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import register from '../pi_jev_demo/pi_jev_demo_extension.ts';

function harness() {
  const events = new Map(), commands = new Map(), notices = [];
  register({on: (event, handler) => events.set(event, handler), registerCommand: (name, command) => commands.set(name, command)});
  return {events, commands, notices, ctx: {ui: {notify: (message, level) => notices.push({payload: JSON.parse(message.slice('[pi-jev-demo] '.length)), level})}}};
}

async function live(answers, fn) {
  const original = globalThis.fetch, oldKey = process.env.TYPESAFE_API_KEY;
  let options;
  process.env.TYPESAFE_API_KEY = 'unit-test-key';
  globalThis.fetch = async (_url, opts) => { options = opts; return {ok: true, json: async () => ({model: 'test-model', answers})}; };
  try { await fn(() => options); }
  finally { globalThis.fetch = original; if (oldKey === undefined) delete process.env.TYPESAFE_API_KEY; else process.env.TYPESAFE_API_KEY = oldKey; }
}

for (const value of [.2, .2001, .5, .7999, undefined, '0.99', NaN, Infinity, -1, 2]) {
  test(`tool hook blocks uncertain or malformed probability ${value}`, async () => {
    await live({allowed: {type: 'noul', noul: value}}, async () => {
      const h = harness();
      const result = await h.events.get('tool_call')({toolName: 'bash', input: {command: 'git status'}}, h.ctx);
      assert.equal(result?.block, true);
    });
  });
}

test('hook allows validated .8 and API calls have timeout and redirect protection', async () => {
  await live({allowed: {type: 'noul', noul: .8}}, async getOptions => {
    const h = harness();
    assert.equal(await h.events.get('tool_call')({toolName: 'bash', input: {command: 'git status'}}, h.ctx), undefined);
    assert.ok(getOptions().signal instanceof AbortSignal);
    assert.equal(getOptions().redirect, 'error');
  });
});

test('skill outside declared choices is rejected before any file load', async () => {
  const directory = mkdtempSync(`${tmpdir()}/pi-jev-untrusted-`);
  writeFileSync(`${directory}/SKILL.md`, 'Untrusted skill outside the allowlist');
  const choice = relative(fileURLToPath(new URL('../pi_jev_demo/skills/', import.meta.url)), directory);
  try {
    await live({skill: {type: 'choice', choice, confidence: .99, probabilities: {[choice]: 1}}}, async () => {
      const h = harness();
      await h.commands.get('jev-demo-skill').handler('documentation', h.ctx);
      assert.ok(h.notices[0].payload.error);
      assert.equal(await h.events.get('before_agent_start')({}), undefined);
    });
  } finally { rmSync(directory, {recursive: true}); }
});

test('invalid Choice confidence or distribution is not injected', async () => {
  for (const answer of [
    {choice: 'documentation', confidence: .99, probabilities: {documentation: .1, accessibility: .8, ui_implementation: .1}},
    {choice: 'documentation', confidence: 9, probabilities: {documentation: 1, accessibility: 0, ui_implementation: 0}},
    {choice: 'documentation', confidence: .99, probabilities: {documentation: 0, accessibility: 0, ui_implementation: 0}},
  ]) {
    await live({skill: {type: 'choice', ...answer}}, async () => {
      const h = harness();
      await h.commands.get('jev-demo-skill').handler('documentation', h.ctx);
      assert.ok(h.notices[0].payload.error);
      assert.equal(await h.events.get('before_agent_start')({}), undefined);
    });
  }
});

test('offline tool hook sees real tabs in command arguments', async () => {
  const oldKey = process.env.TYPESAFE_API_KEY;
  delete process.env.TYPESAFE_API_KEY;
  try {
    const h = harness();
    const result = await h.events.get('tool_call')({toolName: 'bash', input: {command: 'rm\t-rf ./build'}}, h.ctx);
    assert.equal(result?.block, true);
  } finally { if (oldKey !== undefined) process.env.TYPESAFE_API_KEY = oldKey; }
});

test('valid choice loads exactly the declared skill', async () => {
  await live({skill: {type: 'choice', choice: 'documentation', confidence: .9,
      probabilities: {documentation: .9, accessibility: .05, ui_implementation: .05}}}, async () => {
    const h = harness();
    await h.commands.get('jev-demo-skill').handler('write docs', h.ctx);
    assert.equal(h.notices[0].payload.jev.choice, 'documentation');
    const result = await h.events.get('before_agent_start')({});
    assert.ok(result.message.content.includes('<selected-skill name="documentation">'));
  });
});
