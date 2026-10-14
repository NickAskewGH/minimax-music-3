'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const form = $('#song-form');
const tags = ['intro', 'verse', 'pre-chorus', 'chorus', 'post-chorus', 'bridge', 'instrumental', 'solo', 'outro'];
let presets = {}, activeJob = null, timer = null, audioContext = null, nextPlayTime = 0, scheduled = 0, monitorJob = null;
const terminal = new Set(['complete', 'failed', 'cancelled', 'interrupted']);
const fields = {
  global: [['genre','Genre','Acoustic pop'],['subgenre','Subgenre','Indie folk'],['bpm','Tempo / BPM','96','number'],['key','Key','C'],['scale','Scale / mode','Major'],['scenario','Listening scenario','A quiet morning'],['mood','Emotional progression','Intimate verses, uplifting chorus','text',true],['production','Production profile','Warm tape, natural room sound','text',true],['global_metadata','Global metadata · additional details','Era, influences, dynamics…','area',true],['prompt','Free-form description · optional','Any additional musical instructions','area',true]],
  vocal: [['vocal_gender','Vocal gender','Female'],['timbre','Timbre','Warm, breathy'],['performance','Performance style','Close, intimate phrasing','text',true],['harmonies','Harmony & backing vocals','Stacked chorus harmonies','text',true],['vocal_effects','Vocal effects','Short plate reverb','text',true],['vocal_details','Vocal details · additional directions','How should the voice develop?','area',true]],
  arrangement: [['instruments','Primary & secondary instruments','Guitar, piano, upright bass','area',true],['groove','Groove','Relaxed, straight eighths'],['bass','Bass','Warm sustained notes'],['percussion','Percussion','Brushed drums'],['textures','Textures','Soft ambient pad'],['spatial_effects','Spatial effects','Wide chorus, intimate verses','text',true],['arrangement','Arrangement · additional directions','Describe instrument entrances, transitions, solos…','area',true]],
  advanced: [['seed','Seed','0','number'],['randomize_seed','Randomize seed','','checkbox'],['headroom','Playback headroom / seconds','0','number'],['steps','Flow-matching steps','30','number'],['guidance','Guidance scale','1.7','number'],['offload','Memory mode','','select']]
};
for (const [group, definitions] of Object.entries(fields)) {
  const container = $(`#${group}-fields`);
  for (const [name,label,placeholder,type='text',wide=false] of definitions) {
    const wrapper = document.createElement('label');
    if (wide) wrapper.className = 'wide';
    wrapper.append(document.createTextNode(label));
    const input = document.createElement(type === 'area' ? 'textarea' : type === 'select' ? 'select' : 'input');
    input.name = name;
    if (type === 'area') input.rows = 2;
    else if (type === 'select') {
      for (const [value,text] of [['model','CPU offload · recommended'],['none','All on GPU · higher VRAM'],['sequential','Layer offload · low VRAM, slower']]) input.add(new Option(text,value));
    } else input.type = type;
    if (placeholder) input.placeholder = placeholder;
    if (type === 'checkbox') { wrapper.classList.add('check'); wrapper.prepend(input); }
    else wrapper.append(input);
    container.append(wrapper);
  }
}
const bounds = {bpm:[20,300,1],seed:[0,2147483647,1],headroom:[0,60,1],steps:[4,60,1],guidance:[1,4,.1]};
for(const [name,[min,max,step]] of Object.entries(bounds)) Object.assign(form.elements[name],{min,max,step});
function error(message='') { $('#error').textContent = message; $('#error').hidden = !message; }
async function api(path, options={}) {
  const response = await fetch(path, {...options, headers:{'Content-Type':'application/json',...options.headers}});
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(e => `${e.loc.slice(1).join('.') || 'Song'}: ${e.msg}`).join('\n') : data.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}
function showDialog(title, text) { $('#dialog-title').textContent=title; $('#dialog-content').textContent=text; $('#detail-dialog').showModal(); }
$('#close-dialog').onclick=()=>$('#detail-dialog').close();
$('#detail-dialog').addEventListener('click', e=>{if(e.target===$('#detail-dialog')) $('#detail-dialog').close();});
for(const button of document.querySelectorAll('[data-tab]')) button.onclick=()=>{
  for(const tab of document.querySelectorAll('[data-tab]')) { const selected=tab===button; tab.classList.toggle('selected',selected); tab.setAttribute('aria-selected',selected); $(`#tab-${tab.dataset.tab}`).hidden=!selected; }
};
function addSection(section={name:'verse',lyrics:'',direction:''}) {
  const card=document.createElement('div'); card.className='section-card';
  card.innerHTML='<div class="section-top"><span class="tag">♮</span><input class="section-name" aria-label="Section name" list="section-tags"><button type="button" data-action="up" aria-label="Move section up">↑</button><button type="button" data-action="down" aria-label="Move section down">↓</button><button type="button" data-action="remove" aria-label="Remove section">×</button></div><textarea class="section-lyrics" aria-label="Section lyrics" rows="3" placeholder="Write the words to sing…"></textarea><label>ARRANGEMENT DIRECTION<textarea class="section-direction" rows="2" placeholder="Instruments, energy, dynamics…"></textarea></label>';
  $('.section-name',card).value=section.name;
  $('.section-lyrics',card).value=section.lyrics;
  $('.section-direction',card).value=section.direction;
  card.onclick=e=>{ const action=e.target.dataset.action; if(action==='remove')card.remove(); if(action==='up'&&card.previousElementSibling)card.previousElementSibling.before(card); if(action==='down'&&card.nextElementSibling)card.nextElementSibling.after(card); if(action)saveDraft(); };
  $('#sections').append(card);
}
const datalist=document.createElement('datalist');datalist.id='section-tags';for(const tag of tags)datalist.append(new Option(tag));document.body.append(datalist);
$('#add-section').onclick=()=>{addSection();saveDraft();};
function readSong() {
  const song={};
  for(const el of form.elements) {
    if(!el.name)continue;
    if(el.type==='checkbox')song[el.name]=el.checked;
    else if(el.type==='number'||el.type==='range')song[el.name]=el.value===''?null:Number(el.value);
    else song[el.name]=el.value;
  }
  song.sections=$('#lyrics-mode').value==='chart'?[...$('#sections').children].map(card=>({name:$('.section-name',card).value,lyrics:$('.section-lyrics',card).value,direction:$('.section-direction',card).value})):[];
  if($('#lyrics-mode').value==='chart')song.lyrics='';
  return song;
}
function setSong(song) {
  form.reset();
  for(const [key,value] of Object.entries(song)) {
    const el=form.elements[key];if(!el)continue;
    if(el.type==='checkbox')el.checked=Boolean(value);else el.value=value??'';
  }
  $('#sections').replaceChildren();
  for(const section of song.sections||[])addSection(section);
  $('#lyrics-mode').value=song.sections?.length?'chart':'raw';
  updateMode(); updateDerived(); saveDraft();
}
function updateMode() {const chart=$('#lyrics-mode').value==='chart';$('#chart-editor').hidden=!chart;$('#raw-editor').hidden=chart;}
$('#lyrics-mode').onchange=()=>{
  // Moving to raw lyrics preserves the sung text; section directions remain in Arrangement.
  if($('#lyrics-mode').value==='raw' && $('#sections').children.length) {
    const sections=[...$('#sections').children];
    form.elements.lyrics.value=sections.map(c=>`[${$('.section-name',c).value}]\n${$('.section-lyrics',c).value}`).join('\n\n');
    const directions=sections.map(c=>$('.section-direction',c).value.trim()?`[${$('.section-name',c).value}] ${$('.section-direction',c).value}`:'').filter(Boolean).join('\n');
    if(directions)form.elements.arrangement.value=[form.elements.arrangement.value,directions].filter(Boolean).join('\n');
    $('#sections').replaceChildren();
  } else if($('#lyrics-mode').value==='chart' && !$('#sections').children.length) {
    const raw=form.elements.lyrics.value;
    let current=null;
    for(const line of raw.split('\n')) {const match=line.match(/^\s*\[([^\]]+)\]\s*$/);if(match){if(current)addSection(current);current={name:match[1],lyrics:'',direction:''};}else{current??={name:'verse',lyrics:'',direction:''};current.lyrics+=(current.lyrics?'\n':'')+line;}}
    if(current)addSection(current);else addSection();
    form.elements.lyrics.value='';
  }
  updateMode();saveDraft();
};
function updateDerived() {
  $('#duration-value').textContent=`${form.elements.duration.value}s`;
  form.elements.seed.disabled=form.elements.randomize_seed.checked;
  $('#instrumental-hint').hidden=!form.elements.instrumental.checked;
  $('#vocal-fields').style.opacity=form.elements.instrumental.checked?'.45':'1';
}
function saveDraft() {try{localStorage.setItem('music3-draft',JSON.stringify(readSong()));}catch{}}
form.addEventListener('input',()=>{updateDerived();saveDraft();});
$('#preview').onclick=async()=>{try{const p=await api('/api/preview',{method:'POST',body:JSON.stringify(readSong())});showDialog('The prompt sent to the model',`${p.caption}\n\nLYRICS\n${p.lyrics}`);error();}catch(e){error(e.message);}};
function downloadJSON(song) {const url=URL.createObjectURL(new Blob([JSON.stringify(song,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='song-request.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('#export').onclick=async()=>{try{const song=readSong();await api('/api/preview',{method:'POST',body:JSON.stringify(song)});downloadJSON(song);error();}catch(e){error(e.message);}};
$('#import').onclick=()=>$('#import-file').click();
$('#import-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const song=JSON.parse(await file.text());await api('/api/preview',{method:'POST',body:JSON.stringify(song)});const schema=await api('/api/schema');const defaults=Object.fromEntries(Object.entries(schema.properties).filter(([,v])=>'default'in v).map(([k,v])=>[k,v.default]));setSong({...defaults,...song});error();}catch(err){error(err.message);}e.target.value='';};
async function loadHistory() {
  const jobs=await api('/api/jobs');$('#history').replaceChildren();
  if(!jobs.length){const p=document.createElement('p');p.className='muted small';p.textContent='Your songs will appear here.';$('#history').append(p);}
  for(const job of jobs) {const button=document.createElement('button');button.className='history-item';button.textContent=job.title||'Untitled session';const sub=document.createElement('span');sub.textContent=`${job.stage.toUpperCase()} · SEED ${job.seed}`;button.append(sub);button.onclick=()=>selectJob(job.id);$('#history').append(button);}
}
$('#refresh-history').onclick=()=>loadHistory().catch(e=>error(e.message));
function stopMonitoring() {monitorJob=null;scheduled=0;if(audioContext){audioContext.close().catch(()=>{});audioContext=null;}$('#stop-monitor').hidden=true;}
$('#stop-monitor').onclick=stopMonitoring;
async function monitor(job) {
  if(monitorJob!==job.id||!audioContext||!job.preview_ready)return;
  $('#stop-monitor').hidden=false;
  for(const file of job.chunks.slice(scheduled)) {
    const ctx=audioContext;
    const response=await fetch(`/api/jobs/${job.id}/files/${file}`);
    if(!response.ok)throw new Error('Could not load preview audio');
    const buffer=await ctx.decodeAudioData(await response.arrayBuffer());
    if(audioContext!==ctx||monitorJob!==job.id)return;
    const source=ctx.createBufferSource();source.buffer=buffer;source.connect(ctx.destination);
    nextPlayTime=Math.max(nextPlayTime,ctx.currentTime+.1);source.start(nextPlayTime);nextPlayTime+=buffer.duration;scheduled++;
  }
}
function renderJob(job) {
  $('#track-title').textContent=job.title||'Untitled session';$('#track-seed').textContent=`SEED ${job.seed} · ${job.request.steps} STEPS`;
  $('#job-message').textContent=job.message+(job.audio_seconds?` · ${job.audio_seconds.toFixed(1)}s audio`:'');
  $('#job-progress').hidden=terminal.has(job.stage);$('#job-progress').value=job.progress||0;
  $('#cancel').hidden=terminal.has(job.stage)||job.can_cancel===false;
  const audio=$('#audio');audio.hidden=job.stage!=='complete';
  if(job.stage==='complete'){const src=`/api/jobs/${job.id}/files/song.wav`;if(audio.getAttribute('src')!==src)audio.src=src;}
  const links=$('#job-links');links.replaceChildren();
  for(const [file,label] of [['request.json','Settings JSON'],['prompt.txt','Prompt'],['worker.log','Generation log'],...(job.stage==='complete'?[['song.wav','Download WAV'],['song.mp3','Download MP3']]:[])]){const a=document.createElement('a');a.href=`/api/jobs/${job.id}/files/${file}`;a.textContent=label;a.download=file;links.append(a);}
  const reuse=document.createElement('button');reuse.textContent='Use these settings';reuse.onclick=()=>{setSong(job.request);error();window.scrollTo({top:0,behavior:'smooth'});};links.append(reuse);
}
async function pollJob() {
  const id=activeJob;if(!id)return;
  try {
    const job=await api(`/api/jobs/${id}`);if(activeJob!==id)return;
    renderJob(job);await monitor(job);
    if(!terminal.has(job.stage))timer=setTimeout(pollJob,1000);else await loadHistory();
  }catch(e){error(e.message);if(activeJob===id)timer=setTimeout(pollJob,3000);}
}
function selectJob(id, keepMonitor=false) {
  if(!keepMonitor)stopMonitoring();clearTimeout(timer);activeJob=id;
  try{localStorage.setItem('music3-active-job',id);}catch{}
  $('#audio').pause();$('#audio').hidden=true;pollJob();
}
form.onsubmit=async e=>{
  e.preventDefault();error();$('#generate').disabled=true;stopMonitoring();
  if($('#monitor').checked){audioContext=new AudioContext();await audioContext.resume();nextPlayTime=0;}
  try{const job=await api('/api/jobs',{method:'POST',body:JSON.stringify(readSong())});if(audioContext)monitorJob=job.id;selectJob(job.id,true);await loadHistory();}
  catch(err){stopMonitoring();error(err.message);}finally{$('#generate').disabled=false;}
};
$('#cancel').onclick=async()=>{if(!activeJob)return;try{await api(`/api/jobs/${activeJob}/cancel`,{method:'POST'});stopMonitoring();clearTimeout(timer);pollJob();await loadHistory();}catch(e){error(e.message);}};
let systemReport;
async function checkSystem(){systemReport=await api('/api/system');$('#system-label').textContent=systemReport.ready?systemReport.gpu:'Runtime needs attention';$('#system-button .dot').style.background=systemReport.ready?'var(--lime)':'var(--accent)';}
$('#system-button').onclick=async()=>{try{await checkSystem();showDialog('Local runtime',JSON.stringify(systemReport,null,2));}catch(e){error(e.message);}};
$('#new-song').onclick=()=>{setSong({...presets['daybreak'],title:'Untitled session'});error();window.scrollTo({top:0,behavior:'smooth'});};
async function init(){
  presets=await api('/api/presets');
  for(const [key,song] of Object.entries(presets)){const button=document.createElement('button');button.className='preset';button.textContent=song.title;const detail=document.createElement('small');detail.textContent=`${song.genre} · ${song.bpm} BPM`;button.append(detail);button.onclick=()=>{setSong(song);error();};$('#presets').append(button);}
  let draft;try{draft=JSON.parse(localStorage.getItem('music3-draft'));}catch{}
  setSong(draft||presets['daybreak']);await loadHistory();
  const previous=localStorage.getItem('music3-active-job');if(previous){try{await api(`/api/jobs/${previous}`);selectJob(previous);}catch{localStorage.removeItem('music3-active-job');}}
  await checkSystem();
}
init().catch(e=>error(e.message));
