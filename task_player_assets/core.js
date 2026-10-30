/* Exact-frame annotation logic shared by the browser and regression tests. */
(function(root) {
  'use strict';
  const clone = value => JSON.parse(JSON.stringify(value));
  const clampFrame = (frame, count) => Math.max(0, Math.min(count - 1, Math.trunc(frame)));
  const seconds = (frame, fps) => Math.round(frame / fps * 1000) / 1000;
  const frameAt = (time, fps) => Math.round(Number(time) * fps);
  function fields(meta, segment) {
    const row = meta.task.subtasks.find(r => r.row_id === segment.row_id) || {};
    return {step_n: segment.step_n ?? row.step, name: segment.name ?? row.subtask ?? '', object: segment.object ?? row.object ?? ''};
  }
  function compose(meta, draft) {
    const counts = new Map(), subtasks = [], steps = [];
    let previous = 0;
    draft.segments.forEach((segment, i) => {
      const f = fields(meta, segment), key = JSON.stringify(f);
      const occurrence = (counts.get(key) || 0) + 1; counts.set(key, occurrence);
      const item = {seq: segment.seq ?? i + 1, step_n: f.step_n, name: f.name,
        occurrence: segment.occurrence ?? occurrence, object: f.object, object_detail: segment.object_detail ?? null,
        t_start: seconds(previous, meta.fps), t_end: seconds(segment.f_end, meta.fps),
        f_start: previous, f_end: segment.f_end, status: segment.status ?? 'ok',
        recovery_t: segment.recovery_frame == null ? null : seconds(segment.recovery_frame, meta.fps), note: segment.note ?? null};
      subtasks.push(item);
      const title = meta.task.steps.find(s => s.n === item.step_n)?.title ?? `Unknown step ${item.step_n}`;
      if (steps.length && steps[steps.length - 1].step_n === item.step_n && !segment.new_step_interval) {
        Object.assign(steps[steps.length - 1], {f_end: item.f_end, t_end: item.t_end});
      } else {
        steps.push({step_n: item.step_n, step_name: title, t_start: item.t_start,
          t_end: item.t_end, f_start: item.f_start, f_end: item.f_end});
      }
      previous = segment.f_end;
    });
    return {episode: {episode_id: draft.episode_id, robot: draft.robot ?? null, outcome: draft.outcome ?? null}, steps, subtasks};
  }
  function validate(meta, draft, complete = true) {
    const errors = [], warnings = [];
    if (draft.source_sha256 !== meta.source_sha256 || draft.episode_id !== meta.episode_id)
      errors.push('草稿不属于这个视频。');
    if (!Array.isArray(draft.segments) || draft.segments.some(s => !s || typeof s !== 'object'))
      return {errors: [...errors, '草稿没有合法的 segments。'], warnings};
    const result = compose(meta, draft), counts = new Map();
    result.subtasks.forEach((s, index) => {
      const label = `第 ${index + 1} 段`;
      const row = meta.task.subtasks.find(r => r.step === s.step_n && r.subtask === s.name && r.object === s.object);
      if (!row) errors.push(`${label}step_n / name / object 与 task sheet 不匹配。`);
      const key = JSON.stringify([s.step_n, s.name, s.object]);
      const occurrence = (counts.get(key) || 0) + 1; counts.set(key, occurrence);
      if (s.seq !== index + 1) errors.push(`${label}seq 应为 ${index + 1}。`);
      if (s.occurrence !== occurrence) errors.push(`${label}occurrence 应为 ${occurrence}。`);
      if (!Number.isInteger(s.f_end) || s.f_end <= s.f_start || s.f_end >= meta.frame_count)
        errors.push(`${label}结束帧必须晚于上一段，且在视频范围内。`);
      if (!['ok', 'suboptimal', 'failed'].includes(s.status)) errors.push(`${label}状态不合法。`);
      const recovery = draft.segments[index].recovery_frame;
      if (s.status === 'suboptimal' && (!Number.isInteger(recovery) || recovery < s.f_start || recovery > s.f_end))
        errors.push(`${label}需要填写段内的 recovery_t / recovery 帧。`);
      if (s.status !== 'suboptimal' && recovery != null) errors.push(`${label}只有 suboptimal 才能填写 recovery。`);
      if (s.status === 'failed' && (index !== draft.segments.length - 1 || draft.outcome !== 'failed'))
        errors.push(`${label}failed 必须在最后，且 episode outcome 是 failed。`);
      if (s.status !== 'ok' && !s.note) warnings.push(`${label}可在 note 中补充 retry / failure 的说明。`);
      for (const field of ['note', 'object_detail']) {
        if (s[field] != null && (typeof s[field] !== 'string' || /[\u4e00-\u9fff]/.test(s[field])))
          errors.push(`${label}${field} 请填写英文或留空。`);
      }
    });
    if (complete) {
      const end = result.subtasks.at(-1)?.f_end;
      if (end !== meta.frame_count - 1)
        errors.push(`还未覆盖完整视频：最后一段必须结束在 ${meta.frame_count - 1} 帧（当前 ${end ?? '未标记'}）。`);
      if (!['success', 'failed'].includes(draft.outcome)) errors.push('请填写 episode outcome。');
      meta.task.subtasks.forEach(row => {
        const expected = String(row.count).split(',')[0].trim();
        const actual = counts.get(JSON.stringify([row.step, row.subtask, row.object])) || 0;
        if (/^\d+$/.test(expected) && actual !== Number(expected))
          warnings.push(`${row.row_id} 实际 ${actual} 次，task sheet 为 ${expected} 次，请核对。`);
      });
    }
    return {errors, warnings};
  }
  function annotation(meta, draft, options = {}) {
    if (options.strict !== false) {
      const checked = validate(meta, draft);
      if (checked.errors.length) throw Error(checked.errors.join('\n'));
    }
    return compose(meta, draft);
  }
  function updateSegment(meta, draft, index, patch) {
    if (!Number.isInteger(index) || index < 0 || index >= draft.segments.length) throw Error('请选择要编辑的 subtask。');
    const result = clone(draft), current = compose(meta, draft).subtasks[index];
    let start = patch.f_start ?? (patch.t_start == null ? current.f_start : frameAt(patch.t_start, meta.fps));
    let end = patch.f_end ?? (patch.t_end == null ? current.f_end : frameAt(patch.t_end, meta.fps));
    for (const [frame, time] of [[start, patch.t_start], [end, patch.t_end]]) {
      if (time != null && frameAt(time, meta.fps) !== frame) throw Error('帧号和秒数不一致，请修改对应的帧号或时间。');
    }
    const lower = index > 1 ? draft.segments[index - 2].f_end : 0;
    const upper = index + 1 < draft.segments.length ? draft.segments[index + 1].f_end : meta.frame_count;
    if (!Number.isInteger(start) || !Number.isInteger(end) || (index === 0 ? start !== 0 : start <= lower) || start >= end || end >= upper)
      throw Error('边界顺序不合法；第一段从 0 开始，相邻段需要保留正时长。');
    if (index) result.segments[index - 1].f_end = start;
    const segment = result.segments[index]; segment.f_end = end;
    for (const field of ['step_n','name','object','object_detail','status','note','new_step_interval']) {
      if (Object.hasOwn(patch, field)) segment[field] = patch[field];
    }
    if (Object.hasOwn(patch, 'recovery_frame')) segment.recovery_frame = patch.recovery_frame;
    else if (Object.hasOwn(patch, 'recovery_t')) segment.recovery_frame = patch.recovery_t == null ? null : frameAt(patch.recovery_t, meta.fps);
    if (patch.recovery_t != null && patch.recovery_frame != null && frameAt(patch.recovery_t, meta.fps) !== patch.recovery_frame)
      throw Error('recovery_t 和 recovery 帧不一致。');
    const matching = meta.task.subtasks.find(r => r.step === segment.step_n && r.subtask === segment.name && r.object === segment.object);
    if (matching) segment.row_id = matching.row_id;
    // Defaults stay derived after deletion or another edit; intentional overrides remain editable.
    delete segment.seq; delete segment.occurrence;
    const derived = compose(meta, result).subtasks[index];
    for (const field of ['seq','occurrence']) {
      const supplied = patch[field] ?? draft.segments[index][field];
      if (supplied != null && supplied !== derived[field]) segment[field] = supplied;
    }
    return result;
  }
  function fromAnnotation(meta, input) {
    if (input?.episode?.episode_id !== meta.episode_id || !Array.isArray(input.subtasks))
      throw Error('请选择当前视频的 episode / steps / subtasks JSON。');
    const draft = {schema_version: 2, episode_id: meta.episode_id, source_sha256: meta.source_sha256,
      fps: meta.fps, frame_count: meta.frame_count, robot: input.episode.robot ?? null,
      outcome: input.episode.outcome ?? null, segments: [], manual_minutes: null, cursor: 0};
    const warnings = []; let previous = 0;
    input.subtasks.forEach((s, index) => {
      if (!Number.isInteger(s.f_end) || s.f_end <= previous || s.f_end >= meta.frame_count)
        throw Error(`第 ${index + 1} 段 f_end 不合法。`);
      const row = meta.task.subtasks.find(r => r.step === s.step_n && r.subtask === s.name && r.object === s.object);
      const segment = {row_id: row?.row_id ?? null, step_n: s.step_n, name: s.name, object: s.object,
        object_detail: s.object_detail ?? null, f_end: s.f_end, status: s.status ?? 'ok',
        recovery_frame: s.recovery_t == null ? null : frameAt(s.recovery_t, meta.fps), note: s.note ?? null};
      if (s.f_start !== previous || s.t_start !== seconds(previous, meta.fps) || s.t_end !== seconds(s.f_end, meta.fps))
        warnings.push(`第 ${index + 1} 段的 f_start / t_start / t_end 已按结束帧重新计算。`);
      const stepStart = input.steps?.find(step => step.step_n === s.step_n && step.f_start === previous && step.f_end > previous);
      if (index && stepStart && input.subtasks[index - 1].step_n === s.step_n) segment.new_step_interval = true;
      draft.segments.push(segment);
      const derived = compose(meta, draft).subtasks[index];
      if (s.seq != null && s.seq !== derived.seq) segment.seq = s.seq;
      if (s.occurrence != null && s.occurrence !== derived.occurrence) segment.occurrence = s.occurrence;
      previous = s.f_end;
    });
    warnings.push('steps 已根据实际 subtask 区间重新计算。');
    return {draft, warnings};
  }
  const api = {clampFrame, seconds, frameAt, fields, validate, annotation, updateSegment, fromAnnotation};
  root.TaskPlayerCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
