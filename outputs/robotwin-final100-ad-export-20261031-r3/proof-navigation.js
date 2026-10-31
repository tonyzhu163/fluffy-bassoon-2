// Reviewed CUA CALLER sequence only; unchanged qualified capture helpers installed explicitly.
// Prerequisites: prepared fresh form/list fully loaded, exact ready ticket, modern dispatcher alive.
// Fresh AX/Submit index is read immediately before invocation. No backend access.
// Original deadline includes Submit, list, loading and instance refresh.
const submittedAtMs = Date.now();
const overallDeadlineMs = submittedAtMs + 55000;
await proofFormTab.click(freshObservedSubmitIndex); // ONCE
let ax = await proofListTab.getAXState({emit:false,disableDiffing:true});
while (Date.now() < overallDeadlineMs - 6000) {
  const exactTask = proofListTab.playwright.getByRole('link',{name:ready.task_name,exact:true});
  if (await exactTask.count() === 1) {
    await exactTask.click();
    ax = await proofListTab.getAXState({emit:false,disableDiffing:true});
    break;
  }
  // Refresh only after the task list has completed its loading render.
  if (!ax.includes('image loading') && ax.includes('暂无数据')) {
    const refresh = proofListTab.playwright.getByRole('button',{name:'reload',exact:true});
    if (await refresh.count() === 1) await refresh.click();
  }
  ax = await proofListTab.getAXState({emit:false,disableDiffing:true});
}
while (Date.now() < overallDeadlineMs - 3000) {
  const actual = completeCpuDetail(ax,ready);
  if (actual) {
    return await publishCompleteCpuProof({detailTab:proofListTab,fs:proofFs,inbox,ready,submittedAtMs,overallDeadlineMs});
  }
  // Unlike the failed R2 caller, explicitly refresh the Running detail's stale instance table.
  if (ax.includes('text 实例信息') && ax.includes('text 运行中') && !ax.includes('image loading')) {
    const refresh = proofListTab.playwright.getByRole('button',{name:'reload',exact:true});
    if (await refresh.count() === 1) await refresh.click();
  }
  ax = await proofListTab.getAXState({emit:false,disableDiffing:true});
}
throw new Error('Original CUA proof deadline expired; no marker or clock extension');
