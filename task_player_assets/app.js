(function () {
'use strict';
try {
const meta = JSON.parse(document.getElementById('playerData').textContent);
const Core = globalThis.TaskPlayerCore, $ = id => document.getElementById(id);
if (!Core) throw Error('播放器核心脚本未加载。请用task_player_assets 里的“打开播放器.command”启动。');
let image = $('frameImage'), loadTicket = 0;
const storageKey = `task-ground-truth-v1:${meta.source_sha256}`;
let frame = 0, ready = false, playing = false, timer = null, preloads = [], selectedRow = meta.task.subtasks[0].row_id;
let draft = {schema_version: 1, episode_id: meta.episode_id, source_sha256: meta.source_sha256,
  fps: meta.fps, frame_count: meta.frame_count, segments: [], outcome: null, robot: null, manual_minutes: null, cursor: 0};
let activeSegment = null, editorDirty = false;
function tell(text) { $('message').textContent = text; }
function node(tag, text) {const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e;}
function framePath(n) {return `frames/${String(n).padStart(6, '0')}.jpg`;}
function stop() {playing = false; clearTimeout(timer); $('play').textContent = '自动前进';}
function persist() {
  draft.cursor = frame;
  try {localStorage.setItem(storageKey, JSON.stringify(draft)); $('saveState').textContent = '草稿已自动保存在本浏览器。请定期下载草稿备份，换浏览器或清除浏览器数据不会保留它。';}
  catch (_) {$('saveState').textContent = '浏览器未允许本地存储，请使用“下载草稿”保存。';}
}
function showFrame(requested) {
  if (!Number.isFinite(requested)) return;
  frame = Core.clampFrame(requested, meta.frame_count); ready = false;
  $('loading').hidden = false; image.style.opacity = '.3'; $('mark').disabled = true;
  $('position').textContent = `f${frame} / ${meta.frame_count - 1} · ${(frame / meta.fps).toFixed(3)} s（载入中）`;
  $('jumpFrame').value = frame; $('scrubber').value = frame;
  const ticket = ++loadTicket, loaded = new Image();
  loaded.id = 'frameImage'; loaded.alt = `原视频第 ${frame} 帧`; loaded.tabIndex = 0;
  loaded.onclick = () => loaded.focus();
  loaded.onload = () => {
    // A slower earlier request must never put the wrong image under a newer frame number.
    if (ticket !== loadTicket) return;
    image.replaceWith(loaded); image = loaded; onFrameReady();
  };
  loaded.onerror = () => {
    if (ticket !== loadTicket) return;
    stop(); ready = false; $('loading').textContent = '此帧图片缺失';
    tell('请保留 players 中对应视频的 frames 文件夹，或重新运行视频准备命令。');
  };
  $('loading').textContent = '载入帧…';
  loaded.src = framePath(frame);
  persist();
}
function onFrameReady() {
  ready = true; image.style.opacity = '1'; $('loading').hidden = true; $('mark').disabled = false;
  $('startup').hidden = true;
  $('position').textContent = `f${frame} / ${meta.frame_count - 1} · ${(frame / meta.fps).toFixed(3)} s`;
  preloads = [-10, -1, 1, 10].map(delta => {const img = new Image(); img.src = framePath(Core.clampFrame(frame + delta, meta.frame_count)); return img;});
  if (playing) {
    if (frame === meta.frame_count - 1) stop();
    else timer = setTimeout(() => showFrame(frame + 10), 1000 * 10 / meta.fps);
  }
}
function move(delta) {stop(); showFrame(frame + delta);}
function togglePlay() {
  if (playing) {stop(); return;}
  if (!ready || frame >= meta.frame_count - 1) return;
  playing = true; $('play').textContent = '暂停'; showFrame(frame + 10);
}
function taskOptions(select, selected) {
  meta.task.subtasks.forEach(row => select.append(new Option(`${row.row_id} · ${row.subtask}`, row.row_id)));
  select.value = selected;
}
function condition() {
  selectedRow = $('taskRow').value;
  const row = meta.task.subtasks.find(r => r.row_id === selectedRow);
  $('endCondition').textContent = `结束条件：${row.ends_when} ｜次数：${row.count}`;
}
function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2) + '\n'], {type: 'application/json'}));
  const a = node('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
function changeEnd(index, value) {
  const previous = index ? draft.segments[index - 1].f_end : 0;
  const limit = index + 1 < draft.segments.length ? draft.segments[index + 1].f_end - 1 : meta.frame_count - 1;
  if (!Number.isInteger(value) || value <= previous || value > limit) {tell(`结束帧需要在 ${previous + 1}–${limit} 之间。`); render(); return;}
  draft.segments[index].f_end = value; persist(); render();
}
function refreshEditor() {
  if (activeSegment == null || !draft.segments[activeSegment]) {
    activeSegment = null; $('editor').hidden = true; $('editingTarget').textContent = '尚未选择编辑项'; return;
  }
  const item = Core.annotation(meta, draft, {strict:false}).subtasks[activeSegment];
  const form = $('editForm');
  for (const [key,value] of Object.entries(item)) {
    const field = form.elements.namedItem(key); if (field) field.value = value ?? '';
  }
  form.elements.namedItem('recovery_frame').value = draft.segments[activeSegment].recovery_frame ?? '';
  form.elements.namedItem('new_step_interval').checked = draft.segments[activeSegment].new_step_interval === true;
  $('editorTitle').textContent = `编辑 subtask #${activeSegment + 1}`;
  $('editingTarget').textContent = `正在编辑 #${activeSegment + 1}：${item.name}`;
  $('editError').textContent = ''; editorDirty = false;
}
function openEditor(index, scroll = true) {
  if (editorDirty && !window.confirm('编辑区有尚未保存的修改，放弃这些修改并切换？')) return;
  activeSegment = index; $('editor').hidden = false; $('annotationPanel').open = true; refreshEditor();
  if (scroll) $('editor').scrollIntoView({block:'start',behavior:'smooth'});
}
function render() {
  $('segments').replaceChildren();
  const output = Core.annotation(meta, draft, {strict:false});
  output.subtasks.forEach((item, i) => {
    const s = draft.segments[i], card = node('div'); card.className = 'card' + (activeSegment === i ? ' selected' : '');
    card.append(node('b',`${i + 1}. ${item.name}`));
    card.append(node('p',`f${item.f_start} → f${item.f_end} · ${item.t_start.toFixed(3)}–${item.t_end.toFixed(3)} s · ${item.status} · recovery_t: ${item.recovery_t ?? 'null'}`));
    const bar = node('div'); bar.className = 'bar';
    const edit = node('button','编辑全部字段'); edit.setAttribute('aria-label',`编辑 subtask ${i+1} 全部字段`); edit.onclick = () => openEditor(i); bar.append(edit);
    const view = node('button','查看结束帧'); view.onclick = () => {stop(); showFrame(item.f_end); window.scrollTo({top:0,behavior:'smooth'});}; bar.append(view);
    const use = node('button','结束帧改为当前帧'); use.onclick = () => {if (ready) {changeEnd(i,frame); if (activeSegment === i && !editorDirty) refreshEditor();} else tell('请等当前帧加载完成。');}; bar.append(use);
    const remove = node('button','删除此 subtask'); remove.className = 'danger';
    remove.onclick = () => {
      if (editorDirty && !window.confirm('编辑区有未保存修改，仍然删除？')) return;
      draft.segments.splice(i,1); activeSegment = null; editorDirty = false; refreshEditor(); persist(); render();
    }; bar.append(remove); card.append(bar); $('segments').append(card);
  });
  const previous = draft.segments.at(-1)?.f_end ?? 0;
  $('nextStart').textContent = `下一段从 f${previous} 开始；共已标 ${draft.segments.length} 段。`;
  const checked = Core.validate(meta, draft);
  $('validation').textContent = [...checked.errors, ...checked.warnings].join('\n');
  $('jsonPreview').textContent = JSON.stringify(output,null,2);
  const table = node('table'); table.className = 'step-table';
  const header = node('tr'); ['step_n','step_name','f_start','f_end','t_start','t_end'].forEach(key=>header.append(node('th',key))); table.append(header);
  output.steps.forEach(step => {const row=node('tr'); ['step_n','step_name','f_start','f_end','t_start','t_end'].forEach(key=>row.append(node('td',String(step[key])))); table.append(row);});
  $('stepsPreview').replaceChildren(table);
}
function commitEditor() {
  if (activeSegment == null) return false;
  try {
    const form = $('editForm'), patch = {};
    for (const key of ['seq','step_n','occurrence','f_start','f_end','t_start','t_end','recovery_t','recovery_frame']) {
      const value = form.elements.namedItem(key).value;
      patch[key] = value === '' ? null : Number(value);
    }
    for (const key of ['name','object','object_detail','status','note']) patch[key] = form.elements.namedItem(key).value || null;
    patch.new_step_interval = form.elements.namedItem('new_step_interval').checked;
    draft = Core.updateSegment(meta,draft,activeSegment,patch);
    editorDirty = false; persist(); render(); refreshEditor(); tell('已保存。相邻边界、时间和 steps 已同步更新。'); return true;
  } catch (error) {$('editError').textContent = error.message; return false;}
}
function atFrame() {
  return Core.annotation(meta,draft,{strict:false}).subtasks.findIndex(s=>frame>=s.f_start && frame<=s.f_end);
}
function setRecovery() {
  stop(); if (!ready) {tell('请等当前帧加载完成。'); return;}
  if (editorDirty && !commitEditor()) return;
  const index = activeSegment ?? atFrame();
  if (index < 0 || !draft.segments[index]) {tell('先选择一个已记录的 subtask。'); return;}
  const item = Core.annotation(meta,draft,{strict:false}).subtasks[index];
  if (frame < item.f_start || frame > item.f_end) {tell(`当前帧不在所选 subtask #${index+1} 的 ${item.f_start}–${item.f_end} 帧内。`); return;}
  draft = Core.updateSegment(meta,draft,index,{status:'suboptimal',recovery_frame:frame});
  activeSegment=index; persist(); render(); openEditor(index,false);
  tell(`已记录 subtask #${index+1} 的 recovery_t = ${(frame/meta.fps).toFixed(3)} s（f${frame}），status = suboptimal。`);
}
function loadJSON(input) {
  let imported, notes = [];
  if (Array.isArray(input?.segments)) {
    if (input.source_sha256 !== meta.source_sha256 || input.episode_id !== meta.episode_id) throw Error('草稿与当前视频不匹配。');
    imported = input;
  } else {const result=Core.fromAnnotation(meta,input); imported=result.draft; notes=result.warnings;}
  if (draft.segments.length && !window.confirm('载入会替换当前草稿。确认载入？')) return;
  try {localStorage.setItem(storageKey+':before-import',JSON.stringify(draft));} catch (_) {}
  draft=imported; activeSegment=null; editorDirty=false; refreshEditor();
  $('outcome').value=draft.outcome ?? ''; $('robot').value=draft.robot ?? ''; $('minutes').value=draft.manual_minutes ?? '';
  stop(); showFrame(Number(draft.cursor)||0); persist(); render(); $('annotationPanel').open=true;
  tell(['已载入，可选择任意 subtask 编辑。',...notes].join('\n'));
}
function mark() {
  stop();
  if (editorDirty && !commitEditor()) return;
  if (!ready) {tell('请等当前帧加载完成。'); return;}
  const previous = draft.segments.length ? draft.segments[draft.segments.length - 1].f_end : 0;
  if (frame <= previous) {tell(`本段结束帧必须晚于 f${previous}。已有边界可以在下面修改。`); $('annotationPanel').open = true; return;}
  draft.segments.push({row_id: selectedRow, f_end: frame, status: 'ok', recovery_frame: null, note: null, object_detail: null});
  $('annotationPanel').open = true; persist(); render(); openEditor(draft.segments.length-1,false); tell(`已记录 ${selectedRow}：f${previous} → f${frame}。起止时间和 steps 已自动填充，可在编辑区补充 recovery_t 等字段。`);
}
$('title').textContent = meta.episode_id;
$('videoMeta').textContent = `${meta.frame_count} 帧 · ${meta.fps} FPS · 每 10 帧 = ${(10 / meta.fps).toFixed(2)} 秒`;
$('scrubber').max = meta.frame_count - 1; $('jumpFrame').max = meta.frame_count - 1;
taskOptions($('taskRow'), selectedRow); $('taskRow').onchange = condition; condition();
meta.task.notes.forEach(text => $('taskNotes').append(node('p', text)));
try {
  const stored = localStorage.getItem(storageKey);
  if (stored) {
    const saved = JSON.parse(stored);
    if (saved.source_sha256 === meta.source_sha256 && Array.isArray(saved.segments)) {
      draft = saved; frame = Core.clampFrame(Number(saved.cursor) || 0, meta.frame_count); tell('已恢复本浏览器上次保存的草稿和位置。');
    }
  }
} catch (_) {tell('没有载入浏览器草稿；可以手动载入下载过的草稿。');}
$('outcome').value = draft.outcome ?? ''; $('robot').value = draft.robot ?? ''; $('minutes').value = draft.manual_minutes ?? '';
$('outcome').onchange = () => {draft.outcome = $('outcome').value || null; persist(); render();};
$('robot').onchange = () => {draft.robot = $('robot').value || null; persist(); render();};
$('minutes').onchange = () => {draft.manual_minutes = $('minutes').value === '' ? null : Number($('minutes').value); persist();};
$('back10').onclick = () => move(-10); $('forward10').onclick = () => move(10);
$('back1').onclick = () => move(-1); $('forward1').onclick = () => move(1);
$('last').onclick = () => {stop(); showFrame(meta.frame_count - 1);}; $('play').onclick = togglePlay;
$('jump').onclick = () => {stop(); showFrame(Number($('jumpFrame').value)); document.activeElement?.blur();};
$('jumpFrame').onkeydown = e => {if (e.key === 'Enter') $('jump').click();};
$('scrubber').oninput = () => {stop(); showFrame(Number($('scrubber').value));};
$('scrubber').onchange = () => $('scrubber').blur();
$('mark').onclick = mark;
$('copy').onclick = async () => {try {await navigator.clipboard.writeText(String(frame)); tell(`已复制 ${frame}`);} catch (_) {$('jumpFrame').focus(); $('jumpFrame').select(); tell('帧号已选中，按 ⌘C / Ctrl+C 复制。');}};
$('saveDraft').onclick = () => {if (editorDirty && !commitEditor()) return; persist(); download(`${meta.episode_id}.ground_truth.draft.json`, draft);};
$('export').onclick = () => {
  if (editorDirty && !commitEditor()) return;
  try {
    const checked = Core.validate(meta, draft);
    if (checked.errors.length) throw Error(checked.errors.join('\n'));
    if (checked.warnings.length && !window.confirm(checked.warnings.join('\n') + '\n\n已确认这些次数符合实际视频，继续导出？')) return;
    download(`${meta.episode_id}.json`, Core.annotation(meta, draft));
    tell('已导出 ground truth JSON。也请下载草稿，保留人工耗时并便于继续修改。');
  } catch (e) {$('validation').textContent = e.message; tell('暂不能导出，请检查标注区列出的问题。');}
};
$('importDraft').onchange = async () => {
  try {const file=$('importDraft').files[0]; if(file) loadJSON(JSON.parse(await file.text()));}
  catch(error) {tell(`无法载入：${error.message}`);}
  finally {$('importDraft').value='';}
};
$('importPaste').onclick = () => {try {loadJSON(JSON.parse($('pasteJson').value));} catch(error) {tell(`无法载入：${error.message}`);}};
$('loadTemplate').hidden = !meta.annotation_template;
$('loadTemplate').onclick = () => {try {loadJSON(meta.annotation_template);} catch(error) {tell(`无法载入：${error.message}`);}};
$('downloadCurrent').onclick = () => {
  if (editorDirty && !commitEditor()) return;
  download(`${meta.episode_id}.json`,Core.annotation(meta,draft,{strict:false}));
  tell('已下载当前 JSON。outcome 可以是 null；尚未完成的覆盖范围和校验问题仍显示在页面中。');
};
$('editAtFrame').onclick = () => {const index=atFrame(); if(index<0) tell('当前位置还没有已记录的 subtask。'); else openEditor(index);};
$('setRecovery').onclick = setRecovery;
$('editForm').onsubmit = event => {event.preventDefault(); commitEditor();};
$('editForm').oninput = () => {editorDirty=true;};
$('cancelEdit').onclick = () => {if(editorDirty && !window.confirm('放弃尚未保存的编辑？'))return; activeSegment=null; editorDirty=false; refreshEditor(); render();};
for (const [frameName,timeName] of [['f_start','t_start'],['f_end','t_end'],['recovery_frame','recovery_t']]) {
  const form=$('editForm'), f=form.elements.namedItem(frameName), t=form.elements.namedItem(timeName);
  f.addEventListener('input',()=>{t.value=f.value===''?'':Core.seconds(Number(f.value),meta.fps); if(frameName==='recovery_frame' && f.value!=='')form.elements.namedItem('status').value='suboptimal';});
  t.addEventListener('input',()=>{f.value=t.value===''?'':Core.frameAt(t.value,meta.fps); if(timeName==='recovery_t' && t.value!=='')form.elements.namedItem('status').value='suboptimal';});
}
$('editForm').elements.namedItem('status').onchange = () => {
  if($('editForm').elements.namedItem('status').value!=='suboptimal') for(const key of ['recovery_frame','recovery_t'])$('editForm').elements.namedItem(key).value='';
};
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey || ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName)) return;
  if (e.key === 'ArrowRight') {e.preventDefault(); move(e.shiftKey ? 1 : 10);}
  else if (e.key === 'ArrowLeft') {e.preventDefault(); move(e.shiftKey ? -1 : -10);}
  else if (e.code === 'Space') {e.preventDefault(); if (!e.repeat) togglePlay();}
  else if (e.key === 'Home') {e.preventDefault(); stop(); showFrame(0);}
  else if (e.key === 'End') {e.preventDefault(); stop(); showFrame(meta.frame_count - 1);}
  else if (e.key.toLowerCase() === 'r') {e.preventDefault(); if (!e.repeat) setRecovery();}
  else if (e.key.toLowerCase() === 'b') {e.preventDefault(); if (!e.repeat) mark();}
});
document.addEventListener('visibilitychange', () => {if (document.hidden) stop();});
window.addEventListener('beforeunload', persist);
// Every imported player stays under this same origin, preserving existing drafts.
const importStatus = text => {$('videoImportStatus').textContent = text;};
let importing = false;
for (const task of meta.task_choices || []) $('videoTask').append(new Option(task.title, task.task));
async function videoList(selected) {
  try {
    const response = await fetch('/api/videos');
    if (!response.ok) throw Error('请关闭旧启动窗口，再双击“打开播放器.command”重启服务。');
    const data = await response.json();
    $('preparedVideo').replaceChildren();
    for (const video of data.videos) $('preparedVideo').append(new Option(`${video.episode_id} · ${video.frame_count} 帧`,video.url));
    $('preparedVideo').value = selected || location.pathname;
    $('openVideo').disabled = !$('preparedVideo').value;
  } catch (error) {importStatus('视频列表不可用：' + error.message);}
}
function openVideo(url) {
  if (!url || !/^\/(?:imports\/[a-f0-9]{32}\/)?index\.html$/.test(url)) return;
  if (editorDirty && !commitEditor()) {tell('请先修正并保存编辑区内容，再切换视频。'); return;}
  stop(); persist(); location.assign(url);
}
function uploadVideo(file, task) {
  return new Promise((resolve,reject) => {
    const request = new XMLHttpRequest();
    request.open('POST','/api/import?' + new URLSearchParams({filename:file.name,task}));
    request.setRequestHeader('Content-Type','application/octet-stream');
    request.setRequestHeader('X-Player-Import','1');
    request.upload.onprogress = event => {
      if (event.lengthComputable) {
        const percent = Math.round(event.loaded/event.total*100);
        $('videoImportProgress').value=percent; importStatus(`正在传入本机播放器：${percent}%`);
      }
    };
    request.onerror = () => reject(Error('连接中断，请确认播放器启动窗口仍在运行。'));
    request.onload = () => {
      try {
        const data = JSON.parse(request.responseText);
        if (request.status !== 202) throw Error(data.message || '导入失败');
        resolve(data.job_id);
      } catch (error) {reject(Error(request.status === 501 ? '请重启播放器启动窗口后再导入。' : error.message));}
    };
    request.send(file);
  });
}
async function watchImport(jobId) {
  $('videoImportProgress').removeAttribute('value');
  while (true) {
    const response=await fetch(`/api/imports/${jobId}`), job=await response.json();
    if (!response.ok || job.state==='error') throw Error(job.message || '导入失败');
    importStatus(job.message);
    if (job.state==='ready') {
      try {localStorage.removeItem('task-player-pending-import');} catch (_) {}
      await videoList(job.url);
      importStatus('导入完成，点击“打开所选视频”开始标注。');
      return;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
}
function importingState(value) {
  importing=value;
  for (const id of ['importVideo','videoFile','videoTask','openVideo']) $(id).disabled=value;
  $('videoImportProgress').hidden=!value;
}
$('videoFile').onchange=()=>{
  const slug=($('videoFile').files[0]?.name || '').replace(/\.[^.]+$/,'').replace(/^\d+_/,'');
  const match=(meta.task_choices || []).find(task=>slug===task.task);
  $('videoTask').value=match?.task || '';
};
$('importVideo').onclick=async()=>{
  if (importing) return;
  const file=$('videoFile').files[0];
  if (!file) {importStatus('请先选择视频文件。'); return;}
  if (!file.size || file.size>2*1024**3) {importStatus('请选择非空视频，最大支持 2 GB。'); return;}
  if (editorDirty && !commitEditor()) {importStatus('请先修正并保存编辑区内容。'); return;}
  stop(); persist(); importingState(true);
  try {
    const job=await uploadVideo(file,$('videoTask').value);
    try {localStorage.setItem('task-player-pending-import',job);} catch (_) {}
    await watchImport(job);
  } catch(error) {importStatus('无法导入：' + error.message);}
  finally {importingState(false);}
};
$('openVideo').onclick=()=>openVideo($('preparedVideo').value);
$('preparedVideo').onchange=()=>{$('openVideo').disabled=importing || !$('preparedVideo').value;};
$('refreshVideos').onclick=()=>videoList();
videoList();
try {
  const pending=localStorage.getItem('task-player-pending-import');
  if (pending && /^[a-f0-9]{32}$/.test(pending)) {
    $('videoImportPanel').open=true; importingState(true);
    watchImport(pending).catch(error=>{localStorage.removeItem('task-player-pending-import'); importStatus(error.message);}).finally(()=>importingState(false));
  }
} catch (_) {}
render(); showFrame(frame);
} catch (error) {
  const status = document.getElementById('startup');
  status.hidden = false;
  status.textContent = '播放器初始化失败：' + error.message + '。请用“打开播放器.command”启动，或将此错误反馈给我。';
  document.getElementById('loading').textContent = '初始化失败';
  console.error(error);
}
})();
