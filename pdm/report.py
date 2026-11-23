"""Render a standalone local result explorer for a completed experiment."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote


def _materialize_clip(source: Path, directory: Path) -> str:
    if source.is_relative_to(directory):
        return quote(source.relative_to(directory).as_posix(), safe="/")

    source_key = hashlib.sha256(os.fsencode(source)).hexdigest()
    destination = directory / "media" / source_key / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete files, even if the report is regenerated across filesystems.
    with TemporaryDirectory(dir=destination.parent) as staging:
        staged = Path(staging) / source.name
        try:
            os.link(source, staged)
        except OSError:
            shutil.copyfile(source, staged)
        staged.replace(destination)
    return quote(destination.relative_to(directory).as_posix(), safe="/")


def write_report(out: Path, records: list[dict], predictions: list[dict],
                 metrics: dict, sensor: dict | None = None, sampler: dict | None = None) -> Path:
    """Write report.html in the run directory (or to an explicit HTML path)."""
    target = Path(out)
    if target.suffix.lower() != ".html" or target.is_dir():
        target = target / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    directory = target.parent.resolve()
    media = {}

    def media_url(path: str | Path) -> str:
        source = Path(path).resolve(strict=True)
        if source not in media:
            media[source] = _materialize_clip(source, directory)
        return media[source]

    evaluated = {p["sample_id"] for p in predictions}
    fields = ("sample_id", "clip_id", "stage", "domain", "question", "choices",
              "answer_index", "evidence_ms")
    questions = [
        {**{key: record[key] for key in fields if key in record},
         "clip_path": media_url(record["clip_path"])}
        for record in records if record["sample_id"] in evaluated
    ]
    if sensor is not None:
        sensor = {**sensor, "clip_path": media_url(sensor["clip_path"])}
    data = {"records": questions, "predictions": predictions, "metrics": metrics,
            "sensor": sensor, "sampler": sampler}
    # JSON script elements use HTML raw-text parsing, even with a non-JavaScript MIME type.
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).replace("<", "\\u003c")
    target.write_text(_PAGE.replace("<!--RESULT_DATA-->", payload), encoding="utf-8")
    return target


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Continual VideoQA · PDM Framework</title>
<style>
:root{color-scheme:light;--ink:#172e38;--muted:#49646b;--accent:#006b5d;--line:#ccdad7;
--paper:#fff;--wash:#f2f6f4;--mint:#e2f4ed}
*{box-sizing:border-box}
body{margin:0;background:var(--wash);color:var(--ink);font:16px/1.5 "Inter",system-ui,sans-serif}
main,header>div{width:min(1180px,94%);margin:auto}
header{border-bottom:1px solid var(--line);background:var(--paper)}
header>div{display:flex;justify-content:space-between;align-items:center;min-height:64px;gap:12px;flex-wrap:wrap}
header strong{font-size:1.04rem;letter-spacing:-.03em}header nav{display:flex;gap:20px}
a{color:var(--accent);text-underline-offset:3px}
main{padding:24px 0 64px}h1,h2,h3,p{margin:0}
h1{font-size:clamp(2rem,5vw,3.05rem);letter-spacing:-.055em;line-height:1.12}
h2{font-size:1.5rem;letter-spacing:-.035em}h3{font-size:1.1rem;letter-spacing:-.02em}
#summary,.muted,caption{color:var(--muted)}#summary{margin-top:6px}
section{margin-top:32px;scroll-margin-top:14px}
.panel{margin-top:12px;padding:clamp(14px,2.5vw,26px);background:var(--paper);
border:1px solid var(--line);border-radius:12px}
.controls{display:grid;grid-template-columns:1fr 1.8fr 1.1fr .7fr;gap:12px;margin-bottom:22px}
label{display:grid;gap:5px;color:var(--muted);font-size:.83rem;font-weight:650}
select,button{font:inherit;color:var(--ink);border-radius:6px;border:1px solid #9badab}
select{min-width:0;width:100%;background:var(--paper);padding:8px;font-size:.94rem}
button{cursor:pointer}button.primary{padding:9px 14px;border:1px solid var(--accent);
background:var(--accent);color:white;font-weight:650}button:disabled{opacity:.55;cursor:default}
:focus-visible{outline:3px solid #b05600;outline-offset:3px}
.inspection{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:clamp(18px,3vw,36px)}
video{display:block;width:100%;aspect-ratio:16 / 10;max-height:370px;object-fit:contain;
background:#18292e;border-radius:8px}
.playback{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-top:12px}
small,.detail{color:var(--muted);font-size:.85rem}
#question-title{font-size:clamp(1.22rem,2.6vw,1.78rem);margin:3px 0 12px;line-height:1.28}
#choices{margin:0;padding:0;list-style:none}.choice{display:flex;align-items:center;gap:10px;
padding:8px 0;border-top:1px solid var(--line)}
.letter{color:var(--muted);font:600 .85rem/1 monospace}
.choice-text{flex:1;overflow-wrap:anywhere}
.tag{font-size:.74rem;padding:2px 7px;border-radius:4px;font-weight:650;background:var(--mint);color:#075c4e}
.tag.label{color:#364e71;background:#e9eef6}
.table-wrap{overflow:auto;margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:.91rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{background:var(--wash);color:var(--muted);font-size:.8rem}
th[scope=row]{font-weight:600}caption{text-align:left;margin-bottom:8px;font-weight:550}
.table-title{margin-top:24px}details{margin-top:14px}summary{cursor:pointer;color:var(--accent);font-weight:600}
.metric{display:block;min-width:4.7rem}
.meter{height:4px;background:#dce8e4;border-radius:4px;margin:3px 0;width:4.5rem;display:block}
.meter i{background:#29897d;height:100%;display:block;border-radius:inherit}
.pair-grid{display:grid;grid-template-columns:minmax(190px,.7fr) minmax(0,1.65fr);gap:18px}
.pair-grid video{max-height:240px}.pair-grid .table-wrap{margin-top:0}
.text-button{background:white;color:var(--accent);text-decoration:underline;padding:0;border:0;text-align:left}
[hidden]{display:none!important}noscript{display:block;margin:1rem 0}
@media(max-width:780px){.controls{grid-template-columns:1fr 1fr}.inspection,.pair-grid{grid-template-columns:1fr}
header nav{gap:12px;font-size:.88rem}.panel{padding:14px}}
@media(max-width:460px){.controls{grid-template-columns:1fr}header>div{padding:8px 0}}
</style>
</head><body>
<header><div><strong>PDM Framework</strong><nav aria-label="Report sections"><a href="#explorer">Explorer</a>
<a href="#results">Metrics</a><a href="#paired" id="paired-link">Paired comparisons</a></nav></div></header>
<main><h1>Continual VideoQA</h1><p id="summary"></p>
<noscript>Enable JavaScript to inspect the predictions embedded in this report.</noscript>
<section id="explorer"><h2>Prediction explorer</h2><div class="panel" id="inspector">
<div class="controls">
<label>Clip<select id="clip"></select></label><label>Question<select id="question"></select></label>
<label>Method<select id="method"></select></label><label>Training stage<select id="stage"></select></label>
</div><div class="inspection"><div>
<video id="video" controls preload="metadata" playsinline aria-label="Question clip"></video>
<div class="playback"><button type="button" class="primary" id="play-evidence">Play evidence</button>
<a id="source">Open clip</a><small id="media-status" role="status"></small></div></div>
<div><div class="detail" id="question-meta"></div><h3 id="question-title"></h3>
<ol id="choices" aria-label="Answer choices"></ol>
<small id="reference"></small></div></div>
<h3 class="table-title">Answers at this stage</h3><div id="answers"></div></div></section>
<section id="results"><h2>Learning and resource use</h2><div class="panel" id="metric-tables"></div></section>
<section id="paired"><h2>Paired comparisons</h2><div id="sensor" class="panel" hidden>
<h3>Sensor contribution to evidence</h3><p class="muted">The same clip and checkpoint are scored with the sensor off and on.</p>
<div class="pair-grid"><div><video id="sensor-video" controls playsinline preload="metadata" aria-label="Sensor comparison clip"></video>
<small id="sensor-status" role="status"></small></div><div id="sensor-values"></div></div>
<details><summary>Window scores</summary><div id="sensor-scores"></div></details></div>
<div id="sampler" class="panel" hidden><h3>Frame sampling</h3><div id="sampling-values"></div></div></section>
</main><script type="application/json" id="run-data"><!--RESULT_DATA--></script>
<script>
'use strict';
const run = JSON.parse(document.getElementById('run-data').textContent);
const byId = id => document.getElementById(id);
const names = new Map(Object.entries({frozen:'Frozen',retrieval:'Retrieval only',lora:'Sequential LoRA',
  pdm:'PDM',pdm_uniform:'PDM · uniform selection',pdm_no_replay:'PDM · no replay'}));
const methodName = name => names.get(name) || name;
const stageName = stage => 'Stage ' + stage;
const pct = n => Number.isFinite(n) ? (100*n).toFixed(1)+'%' : '—';
const points = n => Number.isFinite(n) ? (100*n).toFixed(1)+' pp' : '—';
const fmt = (n, decimals) => Number.isFinite(n) ? n.toFixed(decimals) : '—';
const interval = times => times ? times.map(n => (n/1000).toFixed(n%10 ? 3 : 2)).join('–')+' s' : '—';
const letter = n => String.fromCharCode(65+n);
function element(tag, value, className) {
  const el = document.createElement(tag);
  if (value != null) el.textContent = value;
  if (className) el.className = className;
  return el;
}
function table(host, caption, columns, rows) {
  const wrapper = element('div', null, 'table-wrap');
  const tbl = element('table');
  tbl.createCaption().textContent = caption;
  const head = tbl.createTHead().insertRow();
  for (const name of columns) {
    const th = element('th', name); th.scope = 'col'; head.append(th);
  }
  const body = tbl.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    row.forEach((value, i) => {
      const cell = element(i===0 ? 'th' : 'td');
      if (i===0) cell.scope = 'row';
      if (value instanceof Node) cell.append(value); else cell.textContent = value ?? '—';
      tr.append(cell);
    });
  }
  wrapper.append(tbl); host.append(wrapper);
}
function options(el, pairs) {
  const old = el.value;
  el.replaceChildren(...pairs.map(([key, name]) => new Option(name, key)));
  if (pairs.some(([key]) => String(key) === old)) el.value = old;
}
function meter(n) {
  if (!Number.isFinite(n)) return '—';
  const box = element('span', pct(n), 'metric');
  const bar = element('span', null, 'meter');
  bar.setAttribute('aria-hidden','true');
  const fill = element('i'); fill.style.width = Math.max(0, Math.min(100,n*100)) + '%';
  bar.append(fill); box.append(bar); return box;
}
function player(video, status) {
  let stopAt = null, frame, ticket = 0;
  function clear() {
    ++ticket; cancelAnimationFrame(frame); stopAt = null; video.pause();
  }
  function boundary() {
    if (stopAt === null) return;
    if (video.currentTime >= stopAt) {
      const end = stopAt; clear(); video.currentTime = end;
    }
  }
  function tick() {
    boundary();
    if (stopAt !== null && !video.paused) frame = requestAnimationFrame(tick);
  }
  video.addEventListener('timeupdate', boundary);
  video.addEventListener('play', () => { cancelAnimationFrame(frame); tick(); });
  video.addEventListener('pause', () => { stopAt = null; cancelAnimationFrame(frame); });
  video.addEventListener('error', () => {
    status.textContent = 'Clip unavailable. Check its local path and browser codec support.';
  });
  return {
    clear,
    source(path) {
      if (video.getAttribute('src') === path) return;
      clear(); status.textContent = ''; video.src = path;
    },
    async play(times) {
      cancelAnimationFrame(frame);
      const request = ++ticket;
      async function start() {
        if (request !== ticket) return;
        try {
          stopAt = times[1]/1000; video.currentTime = times[0]/1000;
          await video.play();
          if (request === ticket) { cancelAnimationFrame(frame); tick(); }
        } catch (err) {
          if (request === ticket) { clear(); status.textContent = 'Evidence playback unavailable. Check the local clip.'; }
        }
      }
      if (video.readyState >= 1) await start();
      else if (!video.error) video.addEventListener('loadedmetadata', start, {once:true});
    }
  };
}
const mainPlayer = player(byId('video'), byId('media-status'));
const clip = byId('clip'), question = byId('question'), method = byId('method'), stage = byId('stage');
const recordById = new Map(run.records.map(r => [r.sample_id, r]));
const clipIds = [...new Set(run.records.map(r => r.clip_id))];
const predictionsById = new Map();
for (const p of run.predictions) {
  if (!predictionsById.has(p.sample_id)) predictionsById.set(p.sample_id, []);
  predictionsById.get(p.sample_id).push(p);
}
const methodIds = new Set(run.predictions.map(p=>p.condition));
byId('summary').textContent = [clipIds.length+' clips',run.records.length+' held-out questions',
  methodIds.size+' methods'].join(' · ');
options(clip, clipIds.map(id => [id,id]));
function changeClip() {
  options(question, run.records.filter(r=>r.clip_id===clip.value)
    .map(r=>[r.sample_id,r.question+' · '+r.sample_id]));
  changeQuestion();
}
function changeQuestion() {
  const candidates = predictionsById.get(question.value);
  const stages = [...new Set(candidates.map(p=>p.stage))].sort((a,b)=>b-a);
  options(stage, stages.map(n => [n,stageName(n)])); changeStage();
}
function changeStage() {
  const rows = predictionsById.get(question.value).filter(p => p.stage===Number(stage.value));
  options(method, rows.map(p => [p.condition,methodName(p.condition)])); display();
}
function display() {
  const r = recordById.get(question.value);
  const rows = predictionsById.get(r.sample_id).filter(p => p.stage===Number(stage.value));
  const p = rows.find(p => p.condition===method.value);
  mainPlayer.clear(); mainPlayer.source(r.clip_path);
  byId('source').href = r.clip_path;
  byId('question-meta').textContent = ['Task '+r.stage, r.domain, r.sample_id].filter(Boolean).join(' · ');
  byId('question-title').textContent = r.question;
  byId('choices').replaceChildren(...r.choices.map((choice,i) => {
    const row = element('li',null,'choice');
    row.append(element('span',letter(i),'letter'),element('span',choice,'choice-text'));
    if (i===r.answer_index) row.append(element('span','Label','tag label'));
    if (i===p.choice_index) row.append(element('span','Prediction','tag'));
    row.append(element('small',fmt(p.choice_scores[i],3)));
    return row;
  }));
  byId('reference').textContent = 'Scores are full-clip logits.' +
    (r.evidence_ms ? ' Labeled evidence '+interval(r.evidence_ms)+'.' : '');
  const button = byId('play-evidence');
  button.textContent = 'Play evidence · '+interval(p.evidence_ms);
  button.onclick = () => mainPlayer.play(p.evidence_ms);
  const answers = byId('answers'); answers.replaceChildren();
  table(answers,'Predictions for the selected question',
    ['Method','Training stage','Answer','Matches label','Evidence'],
    rows.map(row => [methodName(row.condition),stageName(row.stage),letter(row.choice_index)+' · '+row.answer,
      row.choice_index===r.answer_index ? 'Yes' : 'No',interval(row.evidence_ms)]));
}
clip.onchange = changeClip; question.onchange = changeQuestion; method.onchange = display; stage.onchange = changeStage;
if (run.records.length) changeClip();
else byId('inspector').replaceChildren(element('p','No evaluated questions in this run.'));

const accuracy = [], performance = [], costs = [];
for (const [condition, result] of Object.entries(run.metrics)) {
  result.accuracy_matrix.forEach((row,i) => accuracy.push([methodName(condition),stageName(i+1),...row.map(meter)]));
  for (const s of result.stages) {
    const label = [methodName(condition),stageName(s.stage)];
    performance.push([...label,meter(s.seen_average),points(s.forgetting),points(s.adaptation_gain),
      pct(s.mean_evidence_iou),s.evidence_count]);
    costs.push([...label,fmt(s.update_seconds,3),Number.isFinite(s.peak_cuda_bytes)
      ? fmt(s.peak_cuda_bytes/1048576,2) : '—',fmt(s.mean_latency_ms,2),fmt(s.p95_latency_ms,2),s.replay_count]);
  }
}
const metrics = byId('metric-tables');
table(metrics,'Accuracy on each seen task (— indicates an unseen task)',
  ['Method','Training stage','Task 1','Task 2'],accuracy);
table(metrics,'Equal-weight seen-task accuracy and change in percentage points',
  ['Method','Training stage','Seen average','Forgetting','Gain vs. frozen','Mean evidence IoU','Labeled intervals'],performance);
table(metrics,'Update and inference costs (— indicates unavailable)',
  ['Method','Training stage','Update (s)','Peak CUDA (MiB)','Mean latency (ms)','P95 latency (ms)','Replayed examples'],costs);

if (run.sensor) {
  byId('sensor').hidden = false;
  const sp = player(byId('sensor-video'), byId('sensor-status')); sp.source(run.sensor.clip_path);
  const windows = [], outcomes = [];
  for (const mode of ['off','on']) {
    const prediction = run.sensor[mode];
    const play = element('button','Play '+interval(prediction.evidence_ms),'text-button');
    play.type='button'; play.onclick = () => sp.play(prediction.evidence_ms);
    outcomes.push([mode,prediction.answer,play]);
    for (const w of prediction.window_scores || []) {
      windows.push([mode,interval(w.evidence_ms),fmt(w.video_score,3),fmt(w.sensor_score,3),
        fmt(w.combined_score,3),w.evidence_ms.every((n,i)=>n===prediction.evidence_ms[i]) ? 'Yes' : '']);
    }
  }
  table(byId('sensor-values'),'Predicted answer and evidence',['Sensor','Answer','Evidence'],outcomes);
  table(byId('sensor-scores'),'With the sensor enabled, combined score = 0.8 × video + 0.2 × sensor.',
    ['Sensor','Window','Video','Sensor score','Combined','Selected'],windows);
}
if (run.sampler) {
  byId('sampler').hidden = false;
  const s = run.sampler, host = byId('sampling-values');
  table(host,'Frozen checkpoint · Stage 2',['Sampler','Accuracy','Questions'],[
    ['Timestamp midpoints',meter(s.accuracy_midpoint),s.count],['VLM-RLAIF',meter(s.accuracy_vlm_rlaif),s.count]]);
  host.append(element('p',s.disagreements+' of '+s.count+' predictions disagree.'));
  for (const [label, ids] of [['Disagreement sample IDs',s.disagreement_ids],['All sampled IDs',s.sample_ids]]) {
    if (!ids?.length) continue;
    const details = element('details'); details.append(element('summary',label));
    const list = element('ul');
    for (const id of ids) list.append(element('li',id));
    details.append(list); host.append(details);
  }
}
byId('paired').hidden = byId('paired-link').hidden = !(run.sampler || run.sensor);
</script></body></html>
"""
