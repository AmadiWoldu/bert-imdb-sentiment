import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

const MODEL_ID = 'UnrestrictedTemperament/bert-imdb-sentiment-onnx';
const LABELS = { LABEL_0: 'Negative', LABEL_1: 'Positive' };

const $ = (id) => document.getElementById(id);
const text = $('text');
const btn = $('btn');
const count = $('count');
const loader = $('loader');
const loaderText = $('loader-text');
const loaderFill = $('loader-fill');
const result = $('result');
const verdict = $('verdict');

text.addEventListener('input', () => {
  count.textContent = `${text.value.length} / 2000`;
});

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    text.value = chip.dataset.text;
    text.dispatchEvent(new Event('input'));
    if (!btn.disabled) analyze();
  });
});

const clf = await pipeline('text-classification', MODEL_ID, {
  dtype: 'q8', // use 'fp32' if you did not quantize
  progress_callback: (p) => {
    if (p.status === 'progress' && typeof p.progress === 'number') {
      loaderFill.style.width = `${p.progress.toFixed(0)}%`;
      loaderText.textContent = `Downloading model... ${p.progress.toFixed(0)}%`;
    }
  },
});

loader.classList.add('done');
btn.disabled = false;
btn.textContent = 'Analyze sentiment';

async function analyze() {
  const value = text.value.trim();
  if (!value) {
    text.focus();
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Analyzing...';

  try {
    const out = await clf(value, { top_k: 2 });
    const scores = { Positive: 0, Negative: 0 };
    for (const item of out) {
      scores[LABELS[item.label] ?? item.label] = item.score;
    }

    const isPositive = scores.Positive >= scores.Negative;
    verdict.textContent = isPositive ? '😊 Positive review' : '😞 Negative review';
    verdict.className = `verdict ${isPositive ? 'pos' : 'neg'}`;

    $('bar-pos').style.width = `${(scores.Positive * 100).toFixed(1)}%`;
    $('bar-neg').style.width = `${(scores.Negative * 100).toFixed(1)}%`;
    $('pct-pos').textContent = `${(scores.Positive * 100).toFixed(1)}%`;
    $('pct-neg').textContent = `${(scores.Negative * 100).toFixed(1)}%`;

    result.classList.remove('hidden');
  } catch (err) {
    verdict.textContent = 'Something went wrong. Please try again.';
    verdict.className = 'verdict neg';
    result.classList.remove('hidden');
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze sentiment';
  }
}

btn.addEventListener('click', analyze);
text.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
});