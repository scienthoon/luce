// scripts/jev_eval.mjs
//
// jevlocal JSONL 을 Vercel AI Gateway 의 Jev(typesafe-ai/jev-latest) 에 던지고,
// jevlocal.eval / eval_logprob 의 --dump 와 같은 형식으로 예측 덤프를 쓴다.
// 그 덤프는 jevlocal.eval_dump 로 정확도·NLL·ECE·reliability 를 계산한다.
//
// 준비:
//   npm init -y && npm i ai@latest
//   export AI_GATEWAY_API_KEY=...        (Vercel AI Gateway 키)
//
// 실행:
//   node scripts/jev_eval.mjs --in data/ho_csqa/val.jsonl --out logs/jev_csqa.jsonl
//   node scripts/jev_eval.mjs --in data/ho_hs/val.jsonl   --out logs/jev_hs.jsonl --concurrency 8
//   node scripts/jev_eval.mjs --in data/real/val.jsonl    --out logs/jev_real.jsonl --limit 500
//
// 입력 레코드 (jevlocal 스키마):
//   {"state", "type": "choice"|"score"|"noul", "question", "options"|"levels", "label", "label_probs"?, "source"?}
// 출력 레코드:
//   {"type", "option_keys", "probs", "target", "pred", "gold", "confidence", "source", "usage"}

import fs from 'node:fs';
import readline from 'node:readline';
import { experimental_evaluate as evaluate } from 'ai';

// ---------------------------------------------------------------------------
// 인자
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const args = {
    in: null,
    out: null,
    model: 'typesafe-ai/jev-latest',
    concurrency: 4,
    limit: null,
    zdr: true,
    delayMs: 0,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === '--in') { args.in = value; i += 1; }
    else if (key === '--out') { args.out = value; i += 1; }
    else if (key === '--model') { args.model = value; i += 1; }
    else if (key === '--concurrency') { args.concurrency = Number(value); i += 1; }
    else if (key === '--limit') { args.limit = Number(value); i += 1; }
    else if (key === '--no-zdr') { args.zdr = false; }
    else if (key === '--delay-ms') { args.delayMs = Number(value); i += 1; }
    else { throw new Error(`unknown argument: ${key}`); }
  }
  if (!args.in || !args.out) {
    throw new Error('usage: node scripts/jev_eval.mjs --in <jsonl> --out <jsonl> [--model id] [--concurrency n] [--limit n]');
  }
  return args;
}

// ---------------------------------------------------------------------------
// 레코드 -> (state, question, 키 순서, 타겟)
// ---------------------------------------------------------------------------

const NOUL_TRUE_DESCRIPTION = 'Yes, the statement is true.';
const NOUL_FALSE_DESCRIPTION = 'No, the statement is false.';

function normalize(values) {
  const total = values.reduce((a, b) => a + b, 0);
  if (!(total > 0)) throw new Error('target distribution must have positive mass');
  return values.map((v) => v / total);
}

function buildQuestion(record) {
  const type = String(record.type || '').toLowerCase();
  const prompt = record.question;
  if (typeof prompt !== 'string' || !prompt.trim()) throw new Error("record needs a non-empty 'question'");

  if (type === 'choice') {
    const options = record.options;
    if (!options || typeof options !== 'object' || Object.keys(options).length < 2) {
      throw new Error("choice record needs 'options' with >= 2 entries");
    }
    const keys = Object.keys(options);
    let target;
    if (record.label_probs != null) {
      target = normalize(keys.map((k) => Number(record.label_probs[k] ?? 0)));
    } else {
      if (!(record.label in options)) throw new Error(`choice label ${record.label} not in options`);
      target = keys.map((k) => (k === record.label ? 1 : 0));
    }
    return {
      question: { type: 'choice', instructions: prompt, criteria: { ...options } },
      keys,
      target,
      qtype: 'choice',
    };
  }

  if (type === 'score') {
    const levels = record.levels;
    if (!Array.isArray(levels) || levels.length < 2) throw new Error("score record needs 'levels' with >= 2 entries");
    const keys = levels.map((_, i) => String(i));
    let target;
    if (record.label_probs != null) {
      target = normalize(record.label_probs.map(Number));
    } else {
      const index = Number(record.label);
      if (!Number.isInteger(index) || index < 0 || index >= levels.length) throw new Error('score label out of range');
      target = levels.map((_, i) => (i === index ? 1 : 0));
    }
    return {
      question: { type: 'score', instructions: prompt, criteria: levels.map(String) },
      keys,
      target,
      qtype: 'score',
    };
  }

  if (type === 'noul') {
    let pTrue;
    if (record.label_probs != null) pTrue = Number(record.label_probs);
    else if (typeof record.label === 'boolean') pTrue = record.label ? 1 : 0;
    else if (record.label === 0 || record.label === 1) pTrue = Number(record.label);
    else throw new Error('noul label must be a bool');
    return {
      question: {
        type: 'boolean',
        instructions: prompt,
        criteria: { true: NOUL_TRUE_DESCRIPTION, false: NOUL_FALSE_DESCRIPTION },
      },
      keys: ['yes', 'no'],
      target: [pTrue, 1 - pTrue],
      qtype: 'noul',
    };
  }

  throw new Error(`unknown record type: ${type}`);
}

// ---------------------------------------------------------------------------
// 답 -> 키 순서 확률
// ---------------------------------------------------------------------------

function probsInKeyOrder(answer, keys, qtype) {
  if (qtype === 'noul') {
    const p = Number(answer.probability);
    if (!Number.isFinite(p)) throw new Error('boolean answer has no finite probability');
    return [p, 1 - p];
  }
  const dist = answer.probabilities;
  if (!dist || typeof dist !== 'object') {
    throw new Error(`${qtype} answer has no probability distribution (provider ${answer.type})`);
  }
  return keys.map((k) => {
    const v = Number(dist[k]);
    if (!Number.isFinite(v)) throw new Error(`missing probability for key ${k}`);
    return v;
  });
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
  return best;
}

// ---------------------------------------------------------------------------
// 실행
// ---------------------------------------------------------------------------

async function readRecords(path, limit) {
  const records = [];
  const rl = readline.createInterface({ input: fs.createReadStream(path, 'utf8'), crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    records.push(JSON.parse(trimmed));
    if (limit != null && records.length >= limit) break;
  }
  return records;
}

async function evaluateOne(model, record, zdr) {
  const built = buildQuestion(record);
  const providerOptions = zdr ? { gateway: { zeroDataRetention: true } } : undefined;
  const result = await evaluate({
    model,
    state: record.state,
    questions: { q: built.question },
    providerOptions,
  });
  const answer = result.answers.q;
  const probs = probsInKeyOrder(answer, built.keys, built.qtype);
  const confidence = result.providerMetadata?.typesafe?.confidence?.q ?? null;
  const usage = result.usage ?? null;
  return {
    type: built.qtype,
    option_keys: built.keys,
    probs,
    target: built.target,
    pred: built.keys[argmax(probs)],
    gold: built.keys[argmax(built.target)],
    confidence,
    source: record.source ?? null,
    usage: usage ? { inputTokens: usage.promptTokens ?? usage.inputTokens ?? null, outputTokens: usage.completionTokens ?? usage.outputTokens ?? null } : null,
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const records = await readRecords(args.in, args.limit);
  console.error(`loaded ${records.length} records from ${args.in}`);

  const out = fs.createWriteStream(args.out, { encoding: 'utf8' });
  const results = new Array(records.length);
  let next = 0;
  let done = 0;
  let failed = 0;
  let inputTokens = 0;
  const started = Date.now();

  async function worker() {
    while (true) {
      const index = next;
      next += 1;
      if (index >= records.length) return;
      if (args.delayMs > 0 && index > 0) await new Promise((r) => setTimeout(r, args.delayMs));
      try {
        const row = await evaluateOne(args.model, records[index], args.zdr);
        results[index] = row;
        if (row.usage?.inputTokens) inputTokens += row.usage.inputTokens;
      } catch (error) {
        failed += 1;
        results[index] = { error: String(error?.message ?? error), source: records[index].source ?? null };
        console.error(`[${index}] error: ${String(error?.message ?? error).slice(0, 200)}`);
      }
      done += 1;
      if (done % 100 === 0 || done === records.length) {
        const seconds = (Date.now() - started) / 1000;
        console.error(`  ${done}/${records.length} (${seconds.toFixed(0)}s, ${failed} failed, ${inputTokens} input tokens)`);
      }
    }
  }

  const workers = [];
  for (let i = 0; i < Math.max(1, args.concurrency); i += 1) workers.push(worker());
  await Promise.all(workers);

  for (const row of results) out.write(`${JSON.stringify(row)}\n`);
  out.end();

  const seconds = (Date.now() - started) / 1000;
  console.error(`wrote ${results.length} rows to ${args.out} in ${seconds.toFixed(0)}s (${failed} failed)`);
  if (inputTokens > 0) {
    console.error(`input tokens: ${inputTokens} (~$${((inputTokens / 1e6) * 0.042).toFixed(4)} at $0.042/M list price)`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
