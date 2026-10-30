// Install before arming/submission. Call inside the SAME CUA invocation that
// submits/selects the actual detail tab, using its original overall deadline.
function completeCpuDetail(ax, ready) {
  const escaped = ready.task_name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (!new RegExp('\\d+ text ' + escaped + '\\n[^\\n]*\\d+ text 运行中(?:\\n|$)').test(ax)) return null;
  const ids = [...new Set([...ax.matchAll(/\d+ text ID:\s+(job-[a-z0-9]+)(?:\n|$)/g)].map(x => x[1]))];
  if (ids.length !== 1) return null;
  const host = ids[0] + '-master-0';
  if (!ax.split('\n').some(line => /^\s*\d+ text /.test(line) && line.trim().endsWith(' text ' + host))) return null;
  const roots = [], stack = [];
  for (const line of ax.split('\n')) {
    const m = line.match(/^([ \t]*)(\d+) (.+)$/);
    if (!m) continue;
    const depth = m[1].replace(/\t/g, '  ').length;
    const node = {role: m[3].split(' ', 1)[0], value: m[3], children: []};
    while (stack.length && stack[stack.length - 1].depth >= depth) stack.pop();
    (stack.length ? stack[stack.length - 1].node.children : roots).push(node);
    stack.push({depth, node});
  }
  const texts = node => (node.role === 'text' ? [node.value.slice(5)] : []).concat(...node.children.map(texts));
  const nodes = function* (items) { for (const item of items) { yield item; yield* nodes(item.children); } };
  const header = [['类型'], ['实例数量'], ['GPU类型 * 卡数'], ['CPU'], ['内存']];
  const tables = [];
  for (const node of nodes(roots)) {
    if (node.role !== 'table') continue;
    const rows = node.children.filter(x => x.role === 'row');
    const cells = row => row.children.filter(x => x.role === 'cell').map(texts);
    if (!rows.length || JSON.stringify(cells(rows[0])) !== JSON.stringify(header)) continue;
    if (rows.length !== 2) throw new Error('Actual resource table has multiple roles');
    tables.push(cells(rows[1]));
  }
  if (!tables.length) return null; // Header Running alone is never enough.
  if (JSON.stringify(tables) !== JSON.stringify([[['Worker'], ['1'], ['不使用加速芯片'], ['8 核'], ['64 GiB']]])) {
    throw new Error('Actual Worker resource row differs from 1x0GPU/8CPU/64GiB');
  }
  return {allocation_id: ids[0], hostname: host};
}

async function publishCompleteCpuProof({detailTab, fs, inbox, ready, submittedAtMs, overallDeadlineMs}) {
  if (!(overallDeadlineMs <= submittedAtMs + 55000 && Date.now() < overallDeadlineMs)) {
    throw new Error('Use the original bounded submission/proof deadline');
  }
  while (Date.now() < overallDeadlineMs - 1000) {
    const ax = await detailTab.getAXState({emit: false, disableDiffing: true});
    const actual = completeCpuDetail(ax, ready);
    if (actual) {
      const bytes = await detailTab.screenshot({fullPage: false});
      const after = await detailTab.getAXState({emit: false, disableDiffing: true});
      const checked = completeCpuDetail(after, ready);
      if (checked && checked.allocation_id === actual.allocation_id && checked.hostname === actual.hostname) {
        await fs.writeFile(inbox + '/actual-running.image', bytes); // Native PNG/JPEG bytes, no conversion.
        await fs.writeFile(inbox + '/actual-running.txt', after);
        if (Date.now() >= overallDeadlineMs || Date.now() / 1000 >= ready.expires_at_unix) {
          throw new Error('Proof capture missed its original deadline; no marker published');
        }
        const marker = {schema: 'robotwin_cpu_actual_ui_proof_ready_v1', run_id: ready.run_id,
          code_commit: ready.code_commit, nonce: ready.nonce, dispatcher_pid: ready.pid,
          captured_at_unix: Date.now() / 1000, cua_submitted_at_unix: submittedAtMs / 1000};
        await fs.writeFile(inbox + '/proof-ready.json.partial', JSON.stringify(marker));
        await fs.rename(inbox + '/proof-ready.json.partial', inbox + '/proof-ready.json');
        return {actual, elapsed_ms: Date.now() - submittedAtMs, native_bytes: bytes.length};
      }
    }
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  throw new Error('Actual hostname and complete resource row did not arrive in the original CUA deadline');
}
