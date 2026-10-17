import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createGame, appendTranscript, buildPublicState } from './game-engine.mjs';
import { decideForAgent, speechFromDecision } from './jev-agent.mjs';
import { renderSpeech } from './dialogue.mjs';

test('OpenAI writes actual public dialogue; voting receives its private analysis; no Jev call for speech', async t => {
  const env = { ...process.env };
  t.after(() => { process.env = env; });
  process.env.OPENAI_API_KEY = 'test-openai';
  process.env.TYPESAFE_API_KEY = 'test-jev';
  const game = createGame({ seed: 2, mode: 'auto' });
  const calls = [];
  const prose = '真人玩家说要看投票变化，但目前还没投过票。我想先听大家各自怀疑谁，以及依据是什么，再决定站哪边。';
  t.mock.method(globalThis, 'fetch', async (url, request) => {
    const body = JSON.parse(request.body);
    calls.push({ url, body });
    if (String(url).includes('typesafe')) return Response.json({ answers: { voteTarget: { choice: 'human' }, confidence: { confidence: 0.7 } } });
    if (body.temperature === 0.8) return Response.json({ choices: [{ message: { content: JSON.stringify({ text: prose }) } }] });
    return Response.json({ choices: [{ message: { content: JSON.stringify({ summary: 'PRIVATE_ANALYSIS', hypotheses: [], actionAdvice: 'Compare claims', memorySummary: [], speechPlan: { intent: 'question_claim', target: 'human', claim: 'none', evidence: 'claim_conflict', shouldSpeak: true } }) } }] });
  });
  appendTranscript(game, { speaker: 'human', text: '我想看投票变化。' });
  const decision = await decideForAgent(game, 'jev_2', 'day_speech');
  const speech = speechFromDecision(game, 'jev_2', decision);
  assert.equal(speech.text, prose);
  assert.equal(speech.source, 'openai');
  assert.equal(calls.length, 2);
  for (const call of calls) assert.equal(String(call.url), 'https://api.openai.com/v1/chat/completions');
  assert.equal(calls[0].body.model, 'gpt-4o-mini');
  const writer = JSON.parse(calls[1].body.messages[1].content);
  assert.equal(writer.privateState, undefined);
  assert.equal(JSON.stringify(writer).includes('PRIVATE_ANALYSIS'), false);
  assert.equal(writer.publicPlan.evidence, 'low_information');
  appendTranscript(game, { speaker: 'jev_2', ...speech });
  assert.equal(JSON.stringify(buildPublicState(game)).includes('PRIVATE_ANALYSIS'), false);
  await decideForAgent(game, 'jev_2', 'vote');
  assert.equal(calls.length, 4);
  assert.equal(calls[3].body.state.cognition.summary, 'PRIVATE_ANALYSIS');
  assert.equal(JSON.parse(calls[2].body.messages[1].content.split('Current player state:\n')[1]).previousAnalysis.summary, 'PRIVATE_ANALYSIS');
});

test('template fallback cannot turn missing evidence back into a false conflict', () => {
  const speech = renderSpeech({ intent: 'question_claim', target: 'human', evidence: 'low_information' });
  assert.doesNotMatch(speech.text, /冲突/);
});

test('public speech timeout or malformed text uses explicitly labelled fallback', async t => {
  const env = { ...process.env };
  t.after(() => { process.env = env; });
  process.env.OPENAI_API_KEY = 'test-openai';
  t.mock.method(globalThis, 'fetch', async () => Response.json({ choices: [{ message: { content: '{}' } }] }));
  const { writePublicSpeech } = await import('./cognition.mjs');
  const speech = await writePublicSpeech(createGame({ seed: 2 }), 'jev_1', { intent: 'question_claim', target: 'human', evidence: 'claim_conflict', claim: 'none' });
  assert.equal(speech.source, 'template_fallback');
  assert.equal(speech.fallbackReason, 'Invalid public speech');
  assert.doesNotMatch(speech.text, /冲突/);
});

test('OpenAI is the only hosted speech backend and its config is read from OPENAI_* env', async t => {
  const env = { ...process.env };
  t.after(() => { process.env = env; });
  process.env.OPENAI_API_KEY = 'test-openai';
  process.env.OPENAI_MODEL = 'gpt-test-model';
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (url, request) => {
    seen.push({ url: String(url), headers: request.headers, body: JSON.parse(request.body) });
    return Response.json({ choices: [{ message: { content: JSON.stringify({ text: '我先听听真人玩家的理由。' }) } }] });
  });
  const { writePublicSpeech, OPENAI_CHAT_URL } = await import('./cognition.mjs');
  const speech = await writePublicSpeech(createGame({ seed: 3 }), 'jev_1', { intent: 'accuse_player', target: 'human', evidence: 'low_information', claim: 'none' });
  assert.equal(speech.source, 'openai');
  assert.equal(OPENAI_CHAT_URL, 'https://api.openai.com/v1/chat/completions');
  assert.equal(seen.length, 1);
  assert.equal(seen[0].url, OPENAI_CHAT_URL);
  assert.equal(seen[0].headers.Authorization, 'Bearer test-openai');
  assert.equal(seen[0].body.model, 'gpt-test-model');
  const source = await readFile(new URL('./cognition.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /deepseek/i);
});

test('without OPENAI_API_KEY speech uses templates and never calls the network', async t => {
  const env = { ...process.env };
  t.after(() => { process.env = env; });
  delete process.env.OPENAI_API_KEY;
  const fetchMock = t.mock.method(globalThis, 'fetch', async () => { throw new Error('network must not be used'); });
  const { writePublicSpeech, summarizeForAgent } = await import('./cognition.mjs');
  const game = createGame({ seed: 4, mode: 'auto' });
  const speech = await writePublicSpeech(game, 'jev_2', { intent: 'accuse_player', target: 'human', evidence: 'low_information', claim: 'none' });
  assert.equal(speech.source, 'template_fallback');
  assert.equal(speech.fallbackReason, 'no_openai_key');
  assert.ok(speech.text.length > 0);
  const cognition = await summarizeForAgent(game, 'jev_2', 'day_speech');
  assert.equal(cognition.mode, 'heuristic');
  assert.equal(cognition.fallbackReason, 'no_openai_key');
  assert.equal(fetchMock.mock.callCount(), 0);
});
