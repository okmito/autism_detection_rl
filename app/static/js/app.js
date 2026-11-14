(() => {
  const $ = (s, el=document) => el.querySelector(s);
  const $$ = (s, el=document) => Array.from(el.querySelectorAll(s));
  const view = $('#view');
  const pageTitle = $('#pageTitle');
  const pageSub = $('#pageSub');
  const globalStatus = $('#globalStatus');
  const toast = $('#toast');

  const state = {
    meta: null,
    screening: { session: null, last: null, sessions: [] },
    obs: [],
    strategies: [],
    timeline: [],
    consent: [],
    research: null,
    rl: { scenarios: [], policies: [], replay: null, compare: null, idx: 0, playing: false, timer: null },
  };

  function esc(s){ return String(s ?? '').replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }
  function fmt(n, d=3){ return (n===null||n===undefined||Number.isNaN(Number(n))) ? '—' : Number(n).toFixed(d); }
  function safeJoin(v, sep=', '){ return Array.isArray(v) ? v.join(sep) : (v==null ? '' : String(v)); }
  function nowDate(){ return new Date().toISOString().slice(0,16); }
  // Stored timestamps are UTC ISO strings; a datetime-local input needs the user's wall-clock time.
  function toLocalInput(iso){
    const d = new Date(iso); if(Number.isNaN(d.getTime())) return '';
    const p = n => String(n).padStart(2,'0');
    return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
  }
  function showToast(msg){ toast.textContent = msg; toast.classList.add('show'); setTimeout(()=>toast.classList.remove('show'), 2600); }
  function setStatus(msg){ globalStatus.textContent = msg || ''; }

  async function api(path, opts={}){
    const res = await fetch(path, { headers: { 'Content-Type': 'application/json', ...(opts.headers||{}) }, ...opts });
    let data = {};
    try { data = await res.json(); } catch(e){}
    if(!res.ok || data.ok === false){ const e = new Error(data.error || res.statusText || 'Request failed'); e.status = res.status; throw e; }
    return data.data !== undefined ? data.data : data;
  }

  const routes = {
    overview: { title: 'Overview Dashboard', sub: 'Start or resume screening, review recent activity, and access research & support modules.', render: renderOverview },
    screening: { title: 'Guided Adaptive Screening', sub: 'Accessible, adaptive Q-CHAT-style screening flow (research prototype, not diagnostic).', render: renderScreening },
    context: { title: 'Context Explorer', sub: 'Record contextual observations across environments. Kept separate from validated questionnaire responses.', render: renderContext },
    support: { title: 'Support Studio', sub: 'Observation-grounded, non-diagnostic support ideas with tracking and notes.', render: renderSupport },
    timeline: { title: 'Observation Timeline', sub: 'Chronological view of observations, strategies, and session events.', render: renderTimeline },
    'rl-lab': { title: 'RL Intelligence Lab', sub: 'Interactive, reproducible synthetic replay using real policy/inference logic where available.', render: renderRLLab },
    explain: { title: 'Explainable AI', sub: 'Model explanations (attribution, counterfactuals, belief/decision traces). Attribution is not causality.', render: renderExplain },
    research: { title: 'Research & Analytics', sub: 'Actual available offline research artifacts and metrics. No fabricated results.', render: renderResearch },
    privacy: { title: 'Privacy & Consent', sub: 'Data handling, consent records, export, and deletion controls for app-stored information.', render: renderPrivacy },
  };

  function routeName(){ return (location.hash.replace(/^#\/?/, '') || 'overview').split('?')[0]; }
  async function navigate(){
    const name = routes[routeName()] ? routeName() : 'overview';
    $$('#nav a').forEach(a=>a.classList.toggle('active', a.dataset.route===name));
    pageTitle.textContent = routes[name].title;
    pageSub.textContent = routes[name].sub;
    setStatus('Loading…');
    try { await routes[name].render(); }
    catch(e){ view.innerHTML = `<div class="card"><div class="notice danger">Failed to load module: ${esc(e.message)}</div></div>`; }
    setStatus('');
  }
  window.addEventListener('hashchange', navigate);

  // ---------- Overview ----------
  async function renderOverview(){
    const [sessions, obs, strategies, researchDash, consent] = await Promise.all([
      api('/api/screening/sessions').catch(()=>[]),
      api('/api/observations').catch(()=>[]),
      api('/api/support/strategies').catch(()=>[]),
      api('/api/research/dashboard').catch(()=>({artifacts:{}})),
      api('/api/consent').catch(()=>[]),
    ]);
    state.screening.sessions = sessions; state.obs = obs; state.strategies = strategies; state.research = researchDash; state.consent = consent;
    const active = sessions.find(s=>s.status==='active');
    const completed = sessions.filter(s=>s.status==='completed').length;
    const tried = strategies.filter(s=>['tried','helpful','not_helpful','mixed'].includes(s.status)).length;
    const m = (researchDash.artifacts && researchDash.artifacts['step5_policy_benchmark_saudi.json']) || null;
    const h1 = m && m.h1 ? m.h1 : null;
    view.innerHTML = `
      <div class="grid cols-4">
        <div class="card metric"><div class="k">Screening sessions</div><div class="v">${sessions.length}</div><div class="n">${completed} completed${active? ', 1 resumable':''}</div></div>
        <div class="card metric"><div class="k">Observations recorded</div><div class="v">${obs.length}</div><div class="n">Across ${new Set(obs.map(o=>o.context)).size || 0} contexts</div></div>
        <div class="card metric"><div class="k">Support strategies</div><div class="v">${strategies.length}</div><div class="n">${tried} tried/reviewed</div></div>
        <div class="card metric"><div class="k">Consent records</div><div class="v">${consent.length}</div><div class="n">Local app records only</div></div>
      </div>
      <div class="grid cols-2">
        <div class="card">
          <h2>Guided screening</h2>
          <p class="sub">Adaptive questionnaire flow with transparent selection reasons and explainable results.</p>
          <div class="row">
            <a class="btn" href="#/screening">Start / resume screening</a>
            ${active ? `<span class="badge warn">Resumable session: ${esc(active.id)}</span>` : `<span class="badge">No active session</span>`}
          </div>
          ${sessions.length ? `<div class="list" style="margin-top:12px">${sessions.slice(0,4).map(s=>`<div class="item"><h4>${esc(s.policy)} · ${esc(s.mode)} · <span class="badge ${s.status==='completed'?'good':'warn'}">${esc(s.status)}</span></h4><p>${esc(s.updated_at||s.created_at)} · asked ${s.asked_count||0} · ${esc(s.id)}</p></div>`).join('')}</div>` : `<div class="empty" style="margin-top:12px">No screening sessions yet.</div>`}
        </div>
        <div class="card">
          <h2>Observation & support activity</h2>
          <p class="sub">Recent contextual observations and support-plan status (real app data).</p>
          ${obs.length ? `<div class="list">${obs.slice(0,4).map(o=>`<div class="item"><h4>${esc(o.context)}</h4><p>${esc((o.what_happened||'').slice(0,120))}</p></div>`).join('')}</div>` : `<div class="empty">No observations recorded yet. Use Context Explorer to add one.</div>`}
          <div class="row" style="margin-top:10px"><a class="btn secondary" href="#/context">Open Context Explorer</a><a class="btn secondary" href="#/support">Open Support Studio</a></div>
        </div>
      </div>
      <div class="grid cols-3">
        <div class="card"><h3>RL Intelligence Lab</h3><p class="sub">Replay synthetic cases and compare real policy decisions step-by-step.</p><a class="btn secondary" href="#/rl-lab">Open Intelligence Lab</a></div>
        <div class="card"><h3>Research & Analytics</h3><p class="sub">View actual benchmark artifacts (with limitations/caveats).</p><a class="btn secondary" href="#/research">Open Research Dashboard</a></div>
        <div class="card"><h3>Privacy & Consent</h3><p class="sub">Sensitive information handling, consent, export and deletion (app-stored data).</p><a class="btn secondary" href="#/privacy">Open Privacy & Consent</a></div>
      </div>
      <div class="card">
        <h3>Research signal snapshot (from available artifacts)</h3>
        ${h1 ? `<div class="notice ${h1.supported?'good':'warn'}" style="margin-bottom:10px">H1 (adaptive beats exact best fixed subset) at matched budget: <strong>${h1.supported?'supported in this artifact':'not supported in this artifact'}</strong>. Greedy Brier: ${fmt(h1.greedy_brier,4)}, Fixed Brier: ${fmt(h1.exact_fixed_subset_brier,4)} (lower is better). Interpret with dataset/label limitations.</div>` : `<div class="notice">Step-5 benchmark summary not available in current results artifacts; see Research Dashboard for loaded files.</div>`}
        <p class="tiny muted">Only metrics backed by existing files under <code>results/</code> are shown. No values are fabricated.</p>
      </div>`;
  }

  // ---------- Screening ----------
  async function renderScreening(){
    if(!state.meta){ state.meta = await api('/api/meta'); }
    const sessions = await api('/api/screening/sessions').catch(()=>[]);
    state.screening.sessions = sessions;
    const active = sessions.find(s=>s.status==='active');
    view.innerHTML = `
      <div class="split">
        <div class="card">
          <h2>Session</h2>
          <p class="sub">Question selection uses the actual policy implementation. Selection reasons and belief updates are shown transparently.</p>
          <div id="screeningArea"></div>
        </div>
        <div class="card">
          <h2>Start / resume</h2>
          <div class="form">
            <div class="field"><label for="polSel">Policy (real implementations)</label>
              <select id="polSel">${(Array.isArray(state.meta&&state.meta.policies_available)?state.meta.policies_available:['greedy','random','beta_greedy']).map(p=>`<option value="${esc(p)}">${esc(p)}</option>`).join('')}</select>
              <div class="help">greedy, random, beta_greedy ask a fixed number of questions. rl_adaptive is the trained RL agent: it picks each question from the answers so far and decides when to stop. ${rlAvailabilityText(state.meta&&state.meta.rl_policy)}</div></div>
            <div class="field"><label for="modeSel">Mode</label>
              <select id="modeSel"><option value="interactive">Interactive (answer each question)</option><option value="auto">Auto episode (replay on held-out test record)</option></select></div>
            <div class="row"><button class="btn" id="startSessBtn" type="button">Start new session</button>
              ${active?`<button class="btn secondary" id="resumeBtn" type="button">Resume active (${esc(active.id)})</button>`:''}
              <button class="btn ghost" id="refreshSessBtn" type="button">Refresh</button></div>
          </div>
          <hr class="sep">
          <h3>Recent sessions</h3>
          ${sessions.length?`<div class="list">${sessions.slice(0,6).map(s=>`<div class="item"><h4>${esc(s.id)} <span class="badge ${s.status==='completed'?'good':'warn'}">${esc(s.status)}</span></h4><p>${esc(s.policy)} · ${esc(s.mode)} · asked ${s.asked_count||0} · ${esc(s.updated_at||'')}</p>${s.status==='active'?`<button class="btn small secondary" data-resume="${esc(s.id)}">Resume</button>`:`<button class="btn small ghost" data-view="${esc(s.id)}">View</button>`}</div>`).join('')}</div>`:`<div class="empty">No sessions yet.</div>`}
          <div class="notice warn" style="margin-top:10px">This is a screening/referral-research estimate, not a diagnosis. Results are based on a research model and dataset with known limitations (including label circularity).</div>
        </div>
      </div>`;
    $('#startSessBtn').onclick = async ()=>{ try{ setStatus('Starting session (model may load)…'); const r=await api('/api/screening/start',{method:'POST',body:JSON.stringify({mode:$('#modeSel').value,policy:$('#polSel').value})}); state.screening.session=r; await renderScreeningActive(r);}catch(e){showToast(e.message)}finally{setStatus('')} };
    $('#refreshSessBtn').onclick = ()=>navigate();
    if($('#resumeBtn')) $('#resumeBtn').onclick = ()=>resumeSession(active.id);
    $$('[data-resume]').forEach(b=>b.onclick=()=>resumeSession(b.dataset.resume));
    $$('[data-view]').forEach(b=>b.onclick=()=>viewSession(b.dataset.view));
    if(active){ await resumeSession(active.id, true); } else { $('#screeningArea').innerHTML = `<div class="empty">Start a new screening session to begin.</div>`; }
  }

  async function resumeSession(id, silent){
    try{
      const s = await api(`/api/screening/sessions/${id}`);
      if(s.status==='completed'){ renderResult(s.result, s); return; }
      if(s.status==='expired'){ showToast('That session expired after a long pause. Start a new one.'); if(!silent) renderScreening(); return; }
      // reconstruct minimal client session for interactive step
      state.screening.session = { session_id: s.id, mode: s.mode, policy: s.policy, budget: s.budget, tau: s.tau,
        question: s.pending_question, belief: s.belief, asked: s.asked_count, selection: null, resumed: true, trace: s.trace||[],
        decision: s.decision||null, policy_info: s.policy_info||null };
      await renderScreeningActive(state.screening.session, true);
    }catch(e){ if(!silent) showToast(e.message); }
  }
  async function viewSession(id){
    try{ const s = await api(`/api/screening/sessions/${id}`); if(s.result) renderResult(s.result, s); else showToast('No completed result'); }catch(e){showToast(e.message)}
  }

  async function renderScreeningActive(sess, resumed=false){
    const area = $('#screeningArea');
    if(sess.result){ renderResult(sess.result, sess); return; }
    if(sess.mode==='auto'){ renderResult(sess.result, sess); return; }
    const q = sess.question;
    const budget = sess.budget || state.meta?.budget || 6;
    const asked = sess.asked || 0;
    const pct = Math.round((asked/budget)*100);
    const isRL = sess.policy === 'rl_adaptive';
    area.innerHTML = `
      ${fallbackBanner(sess.policy_info)}
      <div class="row" style="justify-content:space-between"><div><span class="badge">${esc(sess.policy)}</span> <span class="badge">Session ${esc(sess.session_id)}</span></div><div class="muted tiny">${isRL ? `Questions ${asked} (agent decides when to stop; at most ${budget})` : `Budget ${asked}/${budget}`} · τ=${fmt(sess.tau||0.5,2)}</div></div>
      <div class="progress" style="margin:10px 0"><i style="width:${pct}%"></i></div>
      ${q ? `<div class="card" style="background:var(--bg-soft);box-shadow:none"><div class="muted tiny">Next question (selected by policy)</div><h3 style="margin:6px 0 4px">${esc(q.code)}: ${esc(q.text)}</h3>
        <div class="row" style="margin-top:10px">
          <button class="btn" data-ans="1">Yes / Atypical response</button>
          <button class="btn secondary" data-ans="0">No / Typical response</button>
          ${isRL ? `<button class="btn ghost" id="skipBtn">Prefer not to answer</button>` : ''}
          <button class="btn ghost" id="stopBtn">Stop & show result</button>
        </div></div>` : `<div class="empty">No pending question.</div>`}
      <div class="grid cols-2" style="margin-top:12px">
        <div><h3>Current estimate</h3><div class="metric card" style="box-shadow:none"><div class="k">Model screening estimate p̂</div><div class="v">${fmt(sess.belief,3)}</div><div class="n">Updates after each observed response. Not a diagnosis.</div></div></div>
        <div><h3>Why this question?</h3>${sess.decision ? decisionHtml(sess.decision) : `<div class="notice">${selectionText(sess.selection)}</div>`}</div>
      </div>
      <div style="margin-top:12px"><h3>Decision history (trace)</h3>${renderTraceTable(sess.trace||state.screening.session?.trace||[])}</div>`;
    $$('[data-ans]').forEach(b=>b.onclick=()=>submitAnswer(sess.session_id, Number(b.dataset.ans), false, q ? q.index : null));
    const stopBtn = $('#stopBtn'); if(stopBtn) stopBtn.onclick=()=>submitAnswer(sess.session_id, null, true);
    const skipBtn = $('#skipBtn'); if(skipBtn) skipBtn.onclick=()=>submitAnswer(sess.session_id, null, false, q ? q.index : null, true);
  }

  function rlAvailabilityText(info){
    if(!info) return '';
    if(info.available) return `RL policy loaded (${esc(info.agent)}, trained on ${esc(info.trained_on)}).`;
    return `RL policy unavailable: ${esc(info.reason||'unknown reason')}. Choosing it falls back to greedy, and the session is labelled as a fallback.`;
  }
  function fallbackBanner(info){
    if(!info || !info.fallback) return '';
    return `<div class="notice warn" style="margin-bottom:10px"><strong>Fallback in use.</strong> ${esc(info.fallback.requested)} was requested but is unavailable (${esc(info.fallback.reason||'')}); this session uses ${esc(info.fallback.used)}.</div>`;
  }
  function decisionHtml(d){
    const vals = (d.action_values||[]).slice(0,5);
    return `<div class="notice"><ul style="margin:0;padding-left:18px">${(d.narrative||[]).map(l=>`<li>${esc(l)}</li>`).join('')}</ul></div>
      ${vals.length?`<table style="margin-top:8px"><thead><tr><th>Option</th><th>Agent value</th></tr></thead><tbody>${vals.map(v=>`<tr><td>${esc(v.action)}</td><td>${fmt(v.q,4)}</td></tr>`).join('')}</tbody></table>`:''}
      <p class="tiny muted">Values are the agent's own estimates (${esc(d.value_units||'')}). ${d.faithful_to_agent_internals ? 'Exact decomposition of the agent\'s value.' : 'Answer probabilities come from the belief model, not the network internals.'}</p>`;
  }

  function selectionText(sel){
    if(!sel) return 'Selection diagnostic unavailable for this step. The policy chose the next question according to its implemented rule.';
    if(sel.applicable===false) return 'Random/uniform policy: no information-gain ranking to display; action chosen from legal items.';
    if(sel.criterion_vacuous) return `Criterion vacuous/tied on current support (support size ${sel.support_size ?? '—'}, pure: ${sel.support_is_pure}). Policy falls back deterministically among tied items; this reflects data/support state, not a UI error.`;
    return `Support-consistent records: ${sel.support_size ?? '—'} (positive: ${sel.n_pos ?? '—'}). Support pure: ${sel.support_is_pure}. IG spread: ${fmt(sel.ig_spread,6)}; tied at max: ${sel.tie_at_max ?? '—'}. Top: ${(sel.top_items||[]).map(t=>`${t.code}(${fmt(t.ig,4)})`).join(', ')||'—'}.`;
  }

  let answering = false;
  function lockAnswerButtons(on){
    const area = $('#screeningArea'); if(!area) return;
    area.setAttribute('aria-busy', on ? 'true' : 'false');
    $$('button', area).forEach(b=>{ b.disabled = on; });
  }
  async function submitAnswer(sessionId, value, stop=false, itemIndex=null, skip=false){
    if(answering) return;   // a double-click must not submit the same answer twice
    answering = true; lockAnswerButtons(true);
    try{
      setStatus('Updating…');
      const r = await api('/api/screening/answer',{method:'POST',body:JSON.stringify({session_id:sessionId, value, stop, item_index:itemIndex, ...(skip ? {skip:true} : {})})});
      if(r.finished){ renderResult(r.result, {id: sessionId}); showToast('Session completed'); }
      else {
        const prev = state.screening.session || {session_id: sessionId};
        prev.question = r.next; prev.belief = r.belief_after; prev.asked = r.asked; prev.selection = r.selection; prev.decision = r.decision || null;
        prev.trace = [...(prev.trace||[]), r.trace_step];
        state.screening.session = prev;
        await renderScreeningActive(prev);
      }
    }catch(e){
      showToast(e.message);
      if(e.status === 409){ await resumeSession(sessionId, true); }   // stale/duplicate answer: show the current question
    }
    finally{ answering = false; lockAnswerButtons(false); setStatus(''); }
  }

  function renderTraceTable(trace){
    if(!trace || !trace.length) return `<div class="empty">No answers recorded yet.</div>`;
    return `<table><thead><tr><th>#</th><th>Item</th><th>Value</th><th>Belief before</th><th>Belief after</th></tr></thead><tbody>
      ${trace.map(t=>`<tr><td>${esc(t.step)}</td><td>${esc(t.item)}</td><td>${t.skipped ? 'skipped' : esc(t.value)}</td><td>${fmt(t.belief_before)}</td><td>${fmt(t.belief_after)}</td></tr>`).join('')}
    </tbody></table>`;
  }

  function renderResult(result, sess){
    const area = $('#screeningArea') || view;
    const r = result || {};
    const dec = r.decision || '—';
    const badge = dec.includes('REFER') && !dec.includes('NO_') ? 'warn' : 'good';
    const contrib = r.explanation?.contributions || r.explanation?.feature_contributions || r.explanation?.evidence?.contributions || [];
    // explanation schema: contributions list in outcome.py is under "contributions"
    const contribs = (r.explanation && r.explanation.contributions) ? r.explanation.contributions : [];
    const cf = r.counterfactual;
    const support = r.support_report || {};
    const recs = support.recommendations || [];
    area.innerHTML = `
      <div class="card" style="box-shadow:none;border-color:var(--line-strong)">
        <div class="row" style="justify-content:space-between"><h2>Screening result (research prototype)</h2><span class="badge ${badge}">${esc(dec.replace(/_/g,' '))}</span></div>
        <div class="grid cols-4" style="margin-top:10px">
          <div class="card metric" style="box-shadow:none"><div class="k">Estimate p̂</div><div class="v">${fmt(r.p_hat,3)}</div></div>
          <div class="card metric" style="box-shadow:none"><div class="k">Questions asked</div><div class="v">${r.asked ?? (r.items_asked||[]).length}</div></div>
          <div class="card metric" style="box-shadow:none"><div class="k">Stop reason</div><div class="v" style="font-size:16px">${esc(r.stop_reason||'—')}</div></div>
          <div class="card metric" style="box-shadow:none"><div class="k">Threshold τ</div><div class="v">${fmt(r.explanation?.outcome?.tau ?? state.meta?.tau ?? 0.5,2)}</div></div>
        </div>
        <div class="notice warn" style="margin-top:10px"><strong>Not a diagnosis.</strong> Screening/referral-research estimate only. Labels/models have documented limitations (including questionnaire-derived circular labels in development data). Interpret with professional assessment.</div>
        ${fallbackBanner(r.policy_info)}
        ${r.stop_explanation ? `<h3 style="margin-top:14px">Why the questionnaire stopped</h3><div class="notice"><ul style="margin:0;padding-left:18px">${(r.stop_explanation.narrative||[]).map(l=>`<li>${esc(l)}</li>`).join('')}</ul></div>
          ${(r.stop_explanation.limitations||[]).length?`<ul class="tiny muted">${r.stop_explanation.limitations.map(l=>`<li>${esc(l)}</li>`).join('')}</ul>`:''}` : ''}
        ${r.policy_info && r.policy_info.agent ? `<p class="tiny muted">Policy: ${esc(r.policy_info.agent)} · artifact ${esc(String(r.policy_info.artifact_sha256||'').slice(0,12))}</p>` : ''}
        <h3 style="margin-top:14px">Items asked & belief trace</h3>
        <p class="sub">Asked: ${safeJoin((r.items_asked||[]).map(esc), ', ') || '-'}</p>
        ${renderTraceTable(r.trace||[])}
        <div class="grid cols-2" style="margin-top:14px">
          <div><h3>Explanation (model attributions)</h3>
            ${contribs.length?`<div class="bar-chart">${contribs.slice(0,10).map(c=>{const v=Math.abs(Number(c.phi||0));const w=Math.min(100,v*200);return `<div class="bar-row"><div>${esc(c.item)} (${esc(c.direction||'')})</div><div class="bar"><i style="width:${w}%"></i></div><div>${fmt(c.phi,4)}</div></div>`}).join('')}</div><p class="tiny muted">Exact group-Shapley contributions over observed responses (baseline = prior/unasked). Attribution ≠ causality.</p>`:`<div class="empty">No contribution breakdown available.</div>`}
          </div>
          <div><h3>Counterfactual & uncertainty</h3>
            ${cf?`<div class="notice">Flipping <strong>${esc(cf.item||cf.feature||'an item')}</strong> from ${esc(cf.original_value)} to ${esc(cf.flipped_value)} changes p̂ to ${fmt(cf.new_p,3)} (decision flip example where supported).</div>`:`<div class="notice">No single-answer flip counterfactual found at this state/threshold (or unavailable).</div>`}
            <div class="notice" style="margin-top:8px">${esc(r.explanation?.uncertainty?.available ? `Uncertainty: ${JSON.stringify(r.explanation.uncertainty)}` : (r.explanation?.uncertainty?.reason || 'Predictive uncertainty interval not available for this predictor; interpretation must be cautious.'))}</div>
          </div>
        </div>
        <h3 style="margin-top:14px">Support report (from answers given; no extra questions)</h3>
        ${recs.length?`<div class="list">${recs.map(rec=>`<div class="item"><h4>${esc(rec.title||'Suggestion')}</h4><p><strong>Triggered by:</strong> ${esc((rec.triggering_question_ids||[]).map((code,i)=>`${code} (${(rec.triggering_responses||[])[i]===1?'atypical':'typical'})`).join(', ') || '-')}</p><p><strong>Why:</strong> ${esc(rec.why_selected||'')}</p><p><strong>Idea:</strong> ${esc(rec.description||'')}</p><p class="tiny muted">Source: ${esc(rec.source_attribution||'rule-based, evidence-linked')} · ${esc(rec.review_status||'review status as provided')}</p>${(rec.limitations||[]).length?`<ul class="tiny muted rec-limitations">${rec.limitations.map(l=>`<li>${esc(l)}</li>`).join('')}</ul>`:''}</div>`).join('')}</div>`:`<div class="notice">No evidence-linked suggestions triggered, or report unavailable. ${esc(support.disclaimer||'')}</div>`}
        ${(support.limitations||[]).length?`<div class="notice warn report-limitations" style="margin-top:8px"><strong>Limitations of this support report:</strong><ul>${support.limitations.map(l=>`<li>${esc(l)}</li>`).join('')}</ul></div>`:''}
        ${recs.length&&support.disclaimer?`<p class="tiny muted report-disclaimer">${esc(support.disclaimer)}</p>`:''}
        ${support.unassessed_areas?.length?`<p class="tiny muted" style="margin-top:8px"><strong>Not assessed by this questionnaire:</strong> ${esc(safeJoin(support.unassessed_areas, ', '))}</p>`:''}
        <div class="row" style="margin-top:12px"><a class="btn secondary" href="#/screening" id="newSessBtn">New session</a><a class="btn ghost" href="#/explain">Open Explainable AI</a></div>
      </div>`;
    // No inline handler (it blocks a Content-Security-Policy): start fresh by reloading on the screening route.
    const newBtn = $('#newSessBtn');
    if(newBtn) newBtn.onclick = (e)=>{ e.preventDefault(); location.hash = '#/screening'; location.reload(); };
  }

  // ---------- Context Explorer ----------
  const CONTEXTS = ['Classroom environments','Social interaction and play','Daily routines','Sensory environments','Communication contexts','Transitions and unexpected changes'];
  async function renderContext(){
    const obs = await api('/api/observations').catch(()=>[]); state.obs = obs;
    view.innerHTML = `
      <div class="grid cols-2">
        <div class="card">
          <h2>Record observation</h2>
          <p class="sub">Contextual observations are stored separately from validated questionnaire responses.</p>
          <form id="obsForm" class="form">
            <input type="hidden" name="id" id="obsId">
            <div class="field"><label for="obsCtx">Context</label><select id="obsCtx" name="context" required>${CONTEXTS.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('')}</select></div>
            <div class="field"><label for="obsWhen">When observed</label><input id="obsWhen" name="occurred_at" type="datetime-local"></div>
            <div class="field"><label for="obsWhat">What happened?</label><textarea id="obsWhat" name="what_happened" required></textarea></div>
            <div class="field"><label for="obsFreq">How frequently observed?</label><input id="obsFreq" name="frequency" type="text" placeholder="e.g. once, daily, several times a week"></div>
            <div class="field"><label for="obsDiff">What seemed difficult?</label><textarea id="obsDiff" name="difficulty"></textarea></div>
            <div class="field"><label for="obsHelp">What appeared to help?</label><textarea id="obsHelp" name="what_helped"></textarea></div>
            <div class="field"><label for="obsCons">Consistency</label><select id="obsCons" name="consistency"><option value="">Select</option><option>Consistent</option><option>Occasional</option><option>Insufficient information</option></select></div>
            <div class="field"><label><input id="obsIns" name="insufficient_info" type="checkbox"> Mark as insufficient information</label></div>
            <div class="field"><label for="obsNotes">Notes</label><textarea id="obsNotes" name="notes"></textarea></div>
            <div class="row"><button class="btn" type="submit" id="obsSave">Save observation</button><button class="btn ghost" type="button" id="obsReset">Clear</button></div>
          </form>
        </div>
        <div class="card">
          <h2>Observation history</h2>
          <div class="row" style="margin-bottom:8px"><input id="obsSearch" type="text" placeholder="Search observations" style="max-width:240px"><select id="obsFilter"><option value="">All contexts</option>${CONTEXTS.map(c=>`<option>${esc(c)}</option>`).join('')}</select></div>
          <div id="obsList"></div>
        </div>
      </div>`;
    $('#obsReset').onclick = ()=>$('#obsForm').reset();
    $('#obsForm').onsubmit = async (e)=>{
      e.preventDefault();
      const fd = new FormData(e.target);
      const body = Object.fromEntries(fd.entries());
      body.insufficient_info = $('#obsIns').checked;
      const id = body.id; delete body.id;
      try{
        if(id) await api(`/api/observations/${id}`,{method:'PUT',body:JSON.stringify(body)});
        else await api('/api/observations',{method:'POST',body:JSON.stringify(body)});
        showToast('Observation saved'); e.target.reset(); navigate();
      }catch(err){ showToast(err.message); }
    };
    const renderList = ()=>{
      const q=($('#obsSearch').value||'').toLowerCase(); const ctx=$('#obsFilter').value;
      let items = state.obs.slice();
      if(ctx) items = items.filter(i=>i.context===ctx);
      if(q) items = items.filter(i=>`${i.what_happened} ${i.difficulty||''} ${i.what_helped||''} ${i.notes||''}`.toLowerCase().includes(q));
      $('#obsList').innerHTML = items.length ? `<div class="list">${items.map(o=>`
        <div class="item"><h4>${esc(o.context)} <span class="badge">${esc(o.consistency||'—')}</span></h4>
        <p><strong>What:</strong> ${esc(o.what_happened)}</p>
        <p><strong>When:</strong> ${esc(o.occurred_at||o.created_at)} · <strong>Frequency:</strong> ${esc(o.frequency||'—')} ${o.insufficient_info?'<span class="badge warn">Insufficient info</span>':''}</p>
        ${o.difficulty?`<p><strong>Difficulty:</strong> ${esc(o.difficulty)}</p>`:''}
        ${o.what_helped?`<p><strong>What helped:</strong> ${esc(o.what_helped)}</p>`:''}
        ${o.notes?`<p><strong>Notes:</strong> ${esc(o.notes)}</p>`:''}
        <div class="row" style="margin-top:6px"><button class="btn small secondary" data-edit="${esc(o.id)}">Edit</button><button class="btn small ghost" data-del="${esc(o.id)}">Remove</button><a class="btn small ghost" href="#/support" data-use="${esc(o.id)}">Use in Support Studio</a></div></div>`).join('')}</div>` : `<div class="empty">No observations match. Add one using the form.</div>`;
      $$('[data-edit]').forEach(b=>b.onclick=()=>{
        const o = state.obs.find(x=>x.id===b.dataset.edit); if(!o) return;
        $('#obsId').value=o.id; $('#obsCtx').value=o.context; $('#obsWhat').value=o.what_happened||''; $('#obsFreq').value=o.frequency||'';
        $('#obsDiff').value=o.difficulty||''; $('#obsHelp').value=o.what_helped||''; $('#obsCons').value=o.consistency||''; $('#obsNotes').value=o.notes||'';
        $('#obsIns').checked=!!o.insufficient_info; $('#obsWhen').value = o.occurred_at ? toLocalInput(o.occurred_at) : '';
        window.scrollTo({top:0,behavior:'smooth'});
      });
      $$('[data-del]').forEach(b=>b.onclick=async()=>{ if(!confirm('Remove this observation record?')) return; try{ await api(`/api/observations/${b.dataset.del}`,{method:'DELETE'}); showToast('Observation removed'); navigate(); }catch(e){showToast(e.message)} });
      $$('[data-use]').forEach(b=>b.onclick=()=>{ sessionStorage.setItem('support_obs', b.dataset.use); });
    };
    $('#obsSearch').oninput = renderList; $('#obsFilter').onchange = renderList;
    renderList();
  }

  // ---------- Support Studio ----------
  async function renderSupport(){
    const [strategies, observations] = await Promise.all([api('/api/support/strategies').catch(()=>[]), api('/api/observations').catch(()=>[])]);
    state.strategies=strategies; state.obs=observations;
    const pre = sessionStorage.getItem('support_obs') || '';
    view.innerHTML = `
      <div class="grid cols-2">
        <div class="card">
          <h2>Suggest from observation</h2>
          <p class="sub">Suggestions are non-diagnostic and grounded in recorded observations (rule-based, transparent).</p>
          <div class="field"><label for="supObs">Observation</label><select id="supObs"><option value="">Select an observation</option>${observations.map(o=>`<option value="${esc(o.id)}">${esc(o.context)} — ${esc((o.what_happened||'').slice(0,60))}</option>`).join('')}</select></div>
          <div class="row" style="margin:8px 0"><button class="btn secondary" id="genSug">Generate suggestions</button></div>
          <div id="sugList" class="list"></div>
        </div>
        <div class="card">
          <h2>Save / track strategy</h2>
          <form id="stratForm" class="form">
            <input type="hidden" id="stId">
            <div class="field"><label>Title</label><input id="stTitle" required></div>
            <div class="field"><label>Domain/context</label><input id="stDomain"></div>
            <div class="field"><label>Triggered observation ID</label><input id="stObs"></div>
            <div class="field"><label>Why this may be useful</label><textarea id="stWhy"></textarea></div>
            <div class="field"><label>How to try</label><textarea id="stHow"></textarea></div>
            <div class="field"><label>How usefulness can be recorded</label><textarea id="stRec"></textarea></div>
            <div class="field"><label>Status</label><select id="stStatus"><option value="saved">Saved</option><option value="tried">Tried</option><option value="helpful">Helped</option><option value="not_helpful">Did not help</option><option value="mixed">Mixed/unclear</option></select></div>
            <div class="field"><label>Outcome / usefulness notes</label><textarea id="stOutcome"></textarea></div>
            <div class="field"><label>Personal notes</label><textarea id="stNotes"></textarea></div>
            <div class="row"><button class="btn" type="submit">Save strategy</button><button class="btn ghost" type="button" id="stReset">Clear</button></div>
          </form>
        </div>
      </div>
      <div class="card">
        <h2>Saved strategies & history</h2>
        <div id="stratList"></div>
      </div>`;
    if(pre) $('#supObs').value = pre;
    $('#stReset').onclick=()=>$('#stratForm').reset();
    $('#genSug').onclick = async ()=>{
      const oid = $('#supObs').value; if(!oid){showToast('Select an observation'); return;}
      try{ const sugs = await api('/api/support/suggest',{method:'POST',body:JSON.stringify({observation_id:oid})});
        $('#sugList').innerHTML = sugs.length ? sugs.map((s,i)=>`<div class="item"><h4>${esc(s.title)}</h4><p><strong>Triggered by observation:</strong> ${esc(s.triggered_observation_id)}</p><p class="tiny muted"><strong>Matched:</strong> ${(s.matched_terms||[]).length ? (s.matched_terms||[]).map(m=>`“${esc(m.term)}” (${esc(m.field)})`).join(', ') : 'no keyword (general check-in)'} · simple keyword rules, so dismiss it if the match is wrong</p><p><strong>Why:</strong> ${esc(s.why_useful)}</p><p><strong>How to try:</strong> ${esc(s.how_to_try)}</p><p><strong>Record usefulness:</strong> ${esc(s.how_to_record)}</p><button class="btn small secondary" data-i="${i}">Use this</button></div>`).join('') : `<div class="empty">No suggestions.</div>`;
        $$('[data-i]',$('#sugList')).forEach(b=>b.onclick=()=>{ const s=sugs[Number(b.dataset.i)]; $('#stTitle').value=s.title; $('#stDomain').value=s.domain||''; $('#stObs').value=s.triggered_observation_id||oid; $('#stWhy').value=s.why_useful||''; $('#stHow').value=s.how_to_try||''; $('#stRec').value=s.how_to_record||''; showToast('Suggestion loaded into form'); });
      }catch(e){showToast(e.message)}
    };
    $('#stratForm').onsubmit = async (e)=>{
      e.preventDefault();
      const body = { title:$('#stTitle').value, domain:$('#stDomain').value, triggered_observation_id:$('#stObs').value, why_useful:$('#stWhy').value, how_to_try:$('#stHow').value, how_to_record:$('#stRec').value, status:$('#stStatus').value, outcome:$('#stOutcome').value, notes:$('#stNotes').value, source:'observation-grounded rule suggestion (user-saved)' };
      const id = $('#stId').value;
      try{
        if(id) await api(`/api/support/strategies/${id}`,{method:'PUT',body:JSON.stringify(body)});
        else await api('/api/support/strategies',{method:'POST',body:JSON.stringify(body)});
        showToast('Strategy saved'); navigate();
      }catch(err){showToast(err.message)}
    };
    const list = strategies;
    $('#stratList').innerHTML = list.length ? `<div class="list">${list.map(s=>`<div class="item"><h4>${esc(s.title)} <span class="badge">${esc(s.status||'saved')}</span></h4>
      <p><strong>Domain:</strong> ${esc(s.domain||'—')} · <strong>Triggered observation:</strong> ${esc(s.triggered_observation_id||'—')}</p>
      <p><strong>Why useful:</strong> ${esc(s.why_useful||'—')}</p><p><strong>How to try:</strong> ${esc(s.how_to_try||'—')}</p><p><strong>Record usefulness:</strong> ${esc(s.how_to_record||'—')}</p>
      <p><strong>Outcome:</strong> ${esc(s.outcome||'—')}</p><p><strong>Notes:</strong> ${esc(s.notes||'—')}</p>
      <div class="row" style="margin-top:6px"><button class="btn small secondary" data-edit="${esc(s.id)}">Edit</button><button class="btn small ghost" data-tried="${esc(s.id)}">Mark tried/helped</button><button class="btn small danger" data-del="${esc(s.id)}">Remove</button></div></div>`).join('')}</div>` : `<div class="empty">No strategies saved yet. Generate suggestions from an observation.</div>`;
    $$('[data-edit]',$('#stratList')).forEach(b=>b.onclick=()=>{ const s=strategies.find(x=>x.id===b.dataset.edit); if(!s) return; $('#stId').value=s.id; $('#stTitle').value=s.title||''; $('#stDomain').value=s.domain||''; $('#stObs').value=s.triggered_observation_id||''; $('#stWhy').value=s.why_useful||''; $('#stHow').value=s.how_to_try||''; $('#stRec').value=s.how_to_record||''; $('#stStatus').value=s.status||'saved'; $('#stOutcome').value=s.outcome||''; $('#stNotes').value=s.notes||''; window.scrollTo({top:0}); });
    $$('[data-tried]',$('#stratList')).forEach(b=>b.onclick=async()=>{ const outcome = prompt('Outcome/usefulness (e.g. helped, mixed, did not help):','helpful'); if(outcome===null) return; try{ await api(`/api/support/strategies/${b.dataset.tried}`,{method:'PUT',body:JSON.stringify({status:'tried', outcome})}); await api(`/api/support/strategies/${b.dataset.tried}/attempts`,{method:'POST',body:JSON.stringify({status:'tried', outcome})}); showToast('Updated'); navigate(); }catch(e){showToast(e.message)} });
    $$('[data-del]',$('#stratList')).forEach(b=>b.onclick=async()=>{ if(!confirm('Remove this strategy?')) return; try{ await api(`/api/support/strategies/${b.dataset.del}`,{method:'DELETE'}); showToast('Removed'); navigate(); }catch(e){showToast(e.message)} });
  }

  // ---------- Timeline ----------
  async function renderTimeline(){
    const events = await api('/api/timeline').catch(()=>[]); state.timeline = events;
    view.innerHTML = `
      <div class="card">
        <div class="row" style="justify-content:space-between"><h2>Timeline</h2><div class="row"><input id="tlSearch" type="text" placeholder="Search"><select id="tlKind"><option value="">All kinds</option><option value="observation">observation</option><option value="observation_update">observation_update</option><option value="observation_delete">observation_delete</option><option value="support">support</option><option value="support_update">support_update</option><option value="support_attempt">support_attempt</option><option value="support_delete">support_delete</option><option value="screening">screening</option><option value="privacy">privacy</option></select><button class="btn secondary" id="tlExport">Export (JSON)</button></div></div>
        <p class="sub">Chronological records of observations, support strategies, and app events. Handle exports as sensitive.</p>
        <div id="tlList"></div>
      </div>`;
    const render = ()=>{
      const q=($('#tlSearch').value||'').toLowerCase(), k=$('#tlKind').value;
      let items = state.timeline.slice();
      if(k) items = items.filter(e=>e.kind===k);
      if(q) items = items.filter(e=>`${e.title} ${e.summary||''} ${e.notes||''} ${e.context||''}`.toLowerCase().includes(q));
      $('#tlList').innerHTML = items.length ? `<table><thead><tr><th>Time</th><th>Kind</th><th>Title</th><th>Context</th><th>Summary</th><th>Notes</th></tr></thead><tbody>
        ${items.map(e=>`<tr><td>${esc(e.event_time||e.created_at)}</td><td><span class="badge">${esc(e.kind)}</span></td><td>${esc(e.title)}</td><td>${esc(e.context||'—')}</td><td>${esc(e.summary||'—')}</td><td>${esc(e.notes||'—')}</td></tr>`).join('')}</tbody></table>` : `<div class="empty">No timeline events yet. Record observations or screening activity to populate.</div>`;
    };
    $('#tlSearch').oninput=render; $('#tlKind').onchange=render; render();
    $('#tlExport').onclick = async ()=>{ try{ const data = await api('/api/timeline/export',{method:'POST'}); const blob = new Blob([JSON.stringify(data,null,2)],{type:'application/json'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='neurobridge_export.json'; a.click(); showToast('Export downloaded'); }catch(e){showToast(e.message)} };
  }

  // ---------- RL Lab ----------
  async function renderRLLab(){
    const meta = await api('/api/rl/scenarios').catch(()=>({scenarios:[],policies:[],budgets:[6]}));
    state.rl.scenarios = meta.scenarios||[]; state.rl.policies = meta.policies||[];
    view.innerHTML = `
      <div class="grid cols-2">
        <div class="card">
          <h2>Replay configuration</h2>
          <p class="sub">Synthetic, reproducible demonstration cases only. Replays use real policy/inference logic for available policies.</p>
          <div class="form">
            <div class="field"><label>Scenario</label><select id="rlScenario">${state.rl.scenarios.map(s=>`<option value="${esc(s.id)}">${esc(s.name)}</option>`).join('')}</select><div class="help">Synthetic cases (not real people).</div></div>
            <div class="field"><label>Policy</label><select id="rlPolicy"><option value="greedy">greedy (available)</option><option value="random">random (available)</option><option value="beta_greedy">beta_greedy (available)</option><option value="random_fixed">random_fixed (available)</option></select></div>
            <div class="field"><label>Question budget</label><select id="rlBudget">${(meta.budgets||[1,2,3,4,5,6]).map(b=>`<option value="${b}" ${b==6?'selected':''}>${b}</option>`).join('')}</select></div>
            <div class="row"><button class="btn" id="rlRun">Replay</button><button class="btn secondary" id="rlCompare">Compare policies</button></div>
          </div>
          <div class="notice" style="margin-top:10px"><strong>Policy availability (honest):</strong> greedy/random/beta_greedy/random_fixed runnable live. DQN/PPO shown as unavailable without loaded checkpoints; exact DP is a limited reference (not run as a general live policy here).</div>
        </div>
        <div class="card">
          <h2>Policy status</h2>
          <div class="list">${state.rl.policies.map(p=>`<div class="item"><h4>${esc(p.name)} <span class="badge ${p.status==='available'?'good':(p.status.startsWith('available')?'warn':'danger')}">${esc(p.status)}</span></h4><p>${esc(p.kind)} — ${esc(p.reason||'')}</p></div>`).join('')}</div>
        </div>
      </div>
      <div class="card">
        <h2>Decision flow replay</h2>
        <div class="flow" aria-label="Decision flow"><div class="node">Current State</div><div class="arrow">→</div><div class="node">Policy Evaluation</div><div class="arrow">→</div><div class="node">Selected Question/Action</div><div class="arrow">→</div><div class="node">New Response</div><div class="arrow">→</div><div class="node">Updated State</div><div class="arrow">→</div><div class="node">Next Decision / Stop</div></div>
        <div class="row" style="margin:10px 0"><button class="btn small" id="rlPlay">Play</button><button class="btn small secondary" id="rlPause">Pause</button><button class="btn small secondary" id="rlStep">Step</button><button class="btn small ghost" id="rlRestart">Restart</button><span class="badge" id="rlStepBadge">Step 0</span></div>
        <div class="grid cols-2">
          <div><h3>Current step</h3><div id="rlCurrent" class="trace">Run a replay to inspect state, evaluation, chosen action, and belief updates.</div></div>
          <div><h3>Decision history & summary</h3><div id="rlHistory" class="trace">—</div></div>
        </div>
      </div>
      <div class="card"><h2>Policy comparison (same synthetic case)</h2><div id="rlCompareOut" class="empty">Run comparison to view question order, counts, belief/decision outcomes side-by-side.</div></div>`;
    $('#rlRun').onclick = runReplay;
    $('#rlCompare').onclick = runCompare;
    $('#rlPlay').onclick = ()=>play(true);
    $('#rlPause').onclick = ()=>play(false);
    $('#rlStep').onclick = stepOnce;
    $('#rlRestart').onclick = ()=>{ state.rl.idx=0; renderReplayStep(); };
  }

  async function runReplay(){
    const body = { scenario: $('#rlScenario').value, policy: $('#rlPolicy').value, budget: Number($('#rlBudget').value) };
    try{ setStatus('Replaying…'); const r = await api('/api/rl/replay',{method:'POST',body:JSON.stringify(body)}); state.rl.replay=r; state.rl.idx=0; renderReplayStep(); showToast('Replay ready'); }catch(e){showToast(e.message)}finally{setStatus('')}
  }
  function renderReplayStep(){
    const r = state.rl.replay; if(!r || !r.ok){ $('#rlCurrent').textContent = r? (r.error||'Unavailable') : 'Run a replay.'; return; }
    const steps = r.steps||[]; const i = Math.max(0, Math.min(state.rl.idx, steps.length-1)); state.rl.idx=i;
    $('#rlStepBadge').textContent = `Step ${i+1}/${steps.length}`;
    $('#rlCurrent').textContent = JSON.stringify(steps[i], null, 2);
    $('#rlHistory').textContent = JSON.stringify({scenario:r.scenario, policy:r.policy, budget:r.budget, synthetic:r.synthetic, summary:r.summary, trace:r.trace}, null, 2);
  }
  function stepOnce(){ const r=state.rl.replay; if(!r||!r.steps) return; state.rl.idx = Math.min(state.rl.idx+1, r.steps.length-1); renderReplayStep(); if(state.rl.idx>=r.steps.length-1) play(false); }
  function play(on){ if(state.rl.timer){clearInterval(state.rl.timer); state.rl.timer=null;} state.rl.playing=on; if(on){ state.rl.timer=setInterval(stepOnce, 900);} }

  async function runCompare(){
    const scenario = $('#rlScenario').value; const budget = Number($('#rlBudget').value);
    const policies = ['greedy','random','beta_greedy','random_fixed'];
    try{ setStatus('Comparing…'); const r = await api('/api/rl/compare',{method:'POST',body:JSON.stringify({scenario, policies, budget})}); state.rl.compare=r;
      $('#rlCompareOut').innerHTML = `<div class="notice">Synthetic comparison on <strong>${esc(r.scenario)}</strong> (budget ${r.budget}). Uses real inference/policies where available.</div>
        <table><thead><tr><th>Policy</th><th>Available</th><th>Questions asked</th><th>Final p̂</th><th>Decision</th><th>Stop reason</th><th>Question order</th></tr></thead><tbody>
        ${r.results.map(x=> x.ok ? `<tr><td>${esc(x.policy)}</td><td><span class="badge good">yes</span></td><td>${x.summary?.questions_asked ?? '—'}</td><td>${fmt(x.summary?.final_p_hat)}</td><td>${esc(x.summary?.decision||'—')}</td><td>${esc(x.summary?.stop_reason||'—')}</td><td>${esc((x.trace||[]).map(t=>t.item).join(' → ')||'—')}</td></tr>` : `<tr><td>${esc(x.policy||'—')}</td><td><span class="badge danger">no</span></td><td colspan="5">${esc(x.error||'Unavailable')}</td></tr>`).join('')}
        </tbody></table>`;
    }catch(e){showToast(e.message)}finally{setStatus('')}
  }

  // ---------- Explainable AI ----------
  async function renderExplain(){
    const sessions = await api('/api/screening/sessions').catch(()=>[]);
    const completed = sessions.filter(s=>s.status==='completed');
    view.innerHTML = `
      <div class="grid cols-2">
        <div class="card"><h2>Explain a completed screening session</h2><p class="sub">Uses existing explanation outputs (group-Shapley attribution, counterfactuals, belief trace, uncertainty/limitations) computed by the backend.</p>
          <div class="field"><label>Completed session</label><select id="exSess"><option value="">Select</option>${completed.map(s=>`<option value="${esc(s.id)}">${esc(s.id)} — ${esc(s.policy)} — ${esc(s.updated_at||'')}</option>`).join('')}</select></div>
          <div class="row" style="margin-top:8px"><button class="btn" id="exLoad">Load explanation</button></div>
          <div class="notice" style="margin-top:10px">Attribution ≠ causality. Explanations describe model behavior on observed inputs. "Supporting/opposing" reflect contribution sign relative to model baseline, not clinical evidence.</div>
        </div>
        <div class="card"><h2>Belief trace / decision path (from session)</h2><div id="exTrace" class="empty">Select a session to view its belief trace and decision history.</div></div>
      </div>
      <div class="card"><h2>Attribution & counterfactuals</h2><div id="exDetail" class="empty">No explanation loaded.</div></div>
      <div class="card"><h2>Synthetic policy explanation (RL Lab)</h2><p class="sub">For step-by-step policy evaluation reasons on synthetic cases, use the RL Intelligence Lab replay (shows state, evaluation, chosen action, updates).</p><a class="btn secondary" href="#/rl-lab">Open RL Intelligence Lab</a></div>`;
    $('#exLoad').onclick = async ()=>{
      const id = $('#exSess').value; if(!id) return;
      try{ const s = await api(`/api/screening/sessions/${id}`); const r=s.result; if(!r){$('#exDetail').innerHTML='<div class="empty">No result.</div>'; return;}
        $('#exTrace').innerHTML = renderTraceTable(r.trace||[]) || '<div class="empty">No trace.</div>';
        const ex = r.explanation || {};
        const contribs = ex.contributions || [];
        const cf = r.counterfactual;
        $('#exDetail').innerHTML = `
          <div class="grid cols-2"><div><h3>Group-Shapley contributions (observed responses)</h3>
            ${contribs.length?`<table><thead><tr><th>Item</th><th>Rank</th><th>φ (contribution)</th><th>Direction</th></tr></thead><tbody>${contribs.map(c=>`<tr><td>${esc(c.item)}</td><td>${esc(c.rank??'')}</td><td>${fmt(c.phi,6)}</td><td>${esc(c.direction||'')}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No contributions.</div>'}
            <p class="tiny muted">Baseline p: ${fmt(ex.attribution?.baseline_p)} → full p: ${fmt(ex.attribution?.full_p)} · additivity error: ${fmt(ex.attribution?.additivity_error,8)} · method: ${esc(ex.attribution?.method||'')}</p></div>
          <div><h3>Counterfactuals & uncertainty</h3>
            ${cf?`<div class="notice">Single-answer flip: ${esc(cf.item||'')} ${esc(cf.original_value)}→${esc(cf.flipped_value)} gives p̂=${fmt(cf.new_p)}.</div>`:`<div class="notice">No single-flip decision change found (per backend counterfactual search) or unavailable.</div>`}
            ${ex.uncertainty?`<div class="notice" style="margin-top:8px"><strong>Uncertainty (model):</strong> ${esc(JSON.stringify(ex.uncertainty))}</div>`:''}
            ${ex.limitations?.length?`<div class="notice warn" style="margin-top:8px"><strong>Limitations:</strong><ul>${ex.limitations.map(l=>`<li>${esc(l)}</li>`).join('')}</ul></div>`:''}
            <div class="notice" style="margin-top:8px">${esc(ex.disclaimer||'Research prototype; screening/referral-research only, not a diagnosis.')}</div></div></div>`;
      }catch(e){showToast(e.message)}
    };
  }

  // ---------- Research Dashboard ----------
  function artifactCard(name, obj){
    if(!obj) return `<div class="card"><h3>${esc(name)}</h3><div class="empty">Artifact not available in <code>results/</code>.</div></div>`;
    const summary = summarizeArtifact(name,obj);
    return `<div class="card"><h3>${esc(name)}</h3>${summary}<details style="margin-top:8px"><summary>Raw (truncated)</summary><pre class="trace">${esc(JSON.stringify(obj,null,2).slice(0,6000))}${JSON.stringify(obj).length>6000?'…':''}</pre></details></div>`;
  }
  function summarizeArtifact(name,obj){
    try{
      if(name==='predictor_saudi_metrics.json') return `<div class="kv"><div>Test Brier</div><div>${fmt(obj.test_brier,4)}</div><div>Test AUROC</div><div>${fmt(obj.test_auroc,4)}</div><div>Test ECE</div><div>${fmt(obj.test_ece,4)}</div><div>Source</div><div>${esc(obj.source||'—')}</div><div>Note</div><div>${esc(obj.note||'Questionnaire-derived labels; interpret as research/dev metric')}</div></div>`;
      if(name==='step5_policy_benchmark_saudi.json'){
        const rows = obj.per_budget || obj.rows || [];
        // This artifact does not always carry an H1 verdict; say so instead of printing "undefined".
        const h1Note = obj.h1 && typeof obj.h1.supported === 'boolean'
          ? `<div class="notice ${obj.h1.supported?'good':'warn'}">H1 supported: <strong>${String(obj.h1.supported)}</strong> (greedy ${fmt(obj.h1.greedy_brier,4)} vs fixed ${fmt(obj.h1.exact_fixed_subset_brier,4)}). Policies: ${safeJoin(obj.policies, ', ')||'-'}.</div>`
          : `<div class="notice warn">This artifact does not record an H1 verdict. Policies: ${safeJoin(obj.policies, ', ')||'-'}.</div>`;
        return h1Note + (rows.length?`<table><thead><tr><th>B</th><th>Policy</th><th>Brier</th><th>AUROC</th><th>Items mean</th></tr></thead><tbody>${rows.slice(0,30).map(r=>`<tr><td>${esc(r.B)}</td><td>${esc(r.policy)}</td><td>${fmt(r.brier,4)}</td><td>${fmt(r.auroc,4)}</td><td>${fmt(r.items_asked_mean,2)}</td></tr>`).join('')}</tbody></table>`:'');
      }
      if(name==='benchmark_multiseed_saudi.json'){
        const rows = obj.aggregate || obj.rows || [];
        return rows.length?`<table><thead><tr><th>Policy</th><th>B</th><th>Mean Brier</th><th>Std</th></tr></thead><tbody>${rows.slice(0,30).map(r=>`<tr><td>${esc(r.policy||r.name)}</td><td>${esc(r.B||r.budget)}</td><td>${fmt(r.mean_brier??r.brier_mean??r.brier,4)}</td><td>${fmt(r.std_brier??r.brier_std,4)}</td></tr>`).join('')}</tbody></table>`:`<div class="notice">Artifact loaded (structure varies); see raw.</div>`;
      }
      if(name==='outcome_explainability_saudi.json') return `<div class="notice">Outcome explainability evaluation artifact loaded. Contains faithfulness/additivity/rank-stability metrics as produced by pipeline (see raw for details).</div>`;
      if(name==='lambda_sweep_saudi.json') return `<div class="notice">λ sweep artifact loaded (V-6 evidence input; not a signed-off threshold). See raw for grid and saturation/degeneracy findings.</div>`;
      if(name==='evoi_scale_saudi.json') return `<div class="notice">EVOI/scale analysis artifact loaded (decision input; no threshold selected). See raw.</div>`;
    }catch(e){}
    return `<div class="notice">Artifact loaded. Summary view not specialized; inspect raw JSON.</div>`;
  }
  async function renderResearch(){
    const data = await api('/api/research/dashboard').catch(()=>({artifacts:{},tables:{},limitations:[]}));
    state.research = data;
    const keys = Object.keys(data.artifacts||{});
    view.innerHTML = `
      <div class="card"><h2>Research integrity & limitations</h2>
        <div class="list">${(data.limitations||[]).map(l=>`<div class="notice warn">${esc(l)}</div>`).join('')}</div>
        <p class="sub" style="margin-top:8px">Offline research benchmarks are shown separately from live app inference. Metrics refer to their recorded splits/labels as per artifacts.</p>
      </div>
      <div class="grid cols-2">
        ${keys.map(k=>artifactCard(k, data.artifacts[k])).join('') || `<div class="card"><div class="empty">No research artifacts found in results/.</div></div>`}
      </div>
      <div class="card"><h2>Tabular artifacts (CSV)</h2>
        ${Object.entries(data.tables||{}).map(([name, rows])=> rows ? `<h3>${esc(name)}</h3><div style="overflow:auto;max-height:360px"><table><thead><tr>${Object.keys(rows[0]||{}).map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,80).map(r=>`<tr>${Object.keys(rows[0]||{}).map(h=>`<td>${esc(r[h])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : `<p class="muted">${esc(name)} not available.</p>`).join('') || '<div class="empty">No CSV tables loaded.</div>'}
      </div>`;
  }

  // ---------- Privacy & Consent ----------
  async function renderPrivacy(){
    const consent = await api('/api/consent').catch(()=>[]);
    view.innerHTML = `
      <div class="grid cols-2">
        <div class="card"><h2>Privacy information</h2>
          <div class="list">
            <div class="item"><h4>What is stored locally</h4><p>Screening sessions/traces, observations, support strategies/attempts, timeline events, and consent records are stored in a local app database (<code>data/neurobridge.db</code> by default). Development research datasets/artifacts remain in the project as-is.</p></div>
            <div class="item"><h4>Data minimization & safety</h4><p>Provide only information you are comfortable recording. Avoid entering unnecessary identifying details in free-text notes. No data is sent to external AI services by this app. Logs should not be used to expose sensitive content.</p></div>
            <div class="item"><h4>Not a diagnostic or clinical system</h4><p>Outputs are for screening/research and support-planning exploration only, not medical advice or diagnosis. No claims of clinical/regulatory compliance are made.</p></div>
          </div>
        </div>
        <div class="card"><h2>Consent</h2>
          <p class="sub">Keep a log of your consent choices for this research prototype. This is a record only: it does not switch any feature on or off, and app data is stored on this computer either way.</p>
          <div class="form">
            <div class="field"><label>Scope</label><select id="consentScope"><option value="general_research_use">General research/prototype use</option><option value="store_observations">Store observations & support plans locally</option><option value="store_screening_sessions">Store screening sessions locally</option></select></div>
            <div class="field"><label><input type="checkbox" id="consentGrant"> I understand this is a non-diagnostic research prototype and I consent to the selected local data handling.</label></div>
            <div class="field"><label>Details (optional)</label><textarea id="consentDetails"></textarea></div>
            <button class="btn" id="consentSave">Save consent record</button>
          </div>
          <h3 style="margin-top:12px">Consent records</h3>
          ${consent.length?`<div class="list">${consent.map(c=>`<div class="item"><h4>${esc(c.scope)} <span class="badge ${c.granted?'good':'warn'}">${c.granted?'granted':'not granted'}</span></h4><p>${esc(c.updated_at||c.created_at)} ${c.details?`· ${esc(c.details)}`:''}</p></div>`).join('')}</div>`:`<div class="empty">No consent records yet.</div>`}
        </div>
      </div>
      <div class="card"><h2>Export & deletion (app-stored data)</h2>
        <p class="sub">Controls apply to everything this app stores about people: observations, support strategies, screening sessions, the timeline and the consent log. "Delete all app data" removes everything except the consent log. None of these controls touch research datasets or results artifacts.</p>
        <div class="row">
          <button class="btn secondary" id="expAll">Export all app data (JSON)</button>
          <button class="btn danger" id="delObs">Delete all observations</button>
          <button class="btn danger" id="delSess">Delete all screening sessions</button>
          <button class="btn danger" id="delAll">Delete all app data</button>
        </div>
        <div id="privMsg" style="margin-top:10px"></div>
      </div>`;
    $('#consentSave').onclick = async ()=>{
      try{ await api('/api/consent',{method:'POST',body:JSON.stringify({scope:$('#consentScope').value, granted:$('#consentGrant').checked, details:$('#consentDetails').value})}); showToast('Consent saved'); navigate(); }catch(e){showToast(e.message)}
    };
    $('#expAll').onclick = async ()=>{ try{ const data = await api('/api/timeline/export',{method:'POST'}); const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='neurobridge_export.json'; a.click(); showToast('Export downloaded'); }catch(e){showToast(e.message)} };
    $('#delSess').onclick = async ()=>{ if(!confirm('Delete ALL stored screening sessions (answers, results and reports)? This cannot be undone.')) return; try{ await api('/api/privacy/delete_sessions',{method:'POST'}); showToast('Screening sessions deleted'); navigate(); }catch(e){showToast(e.message)} };
    $('#delAll').onclick = async ()=>{ if(!confirm('Delete ALL app data: observations, support strategies, screening sessions and the timeline? The consent log is kept. This cannot be undone.')) return; try{ await api('/api/privacy/delete_all',{method:'POST'}); showToast('All app data deleted'); navigate(); }catch(e){showToast(e.message)} };
    $('#delObs').onclick = async ()=>{ if(!confirm('Delete ALL app-stored observations? This cannot be undone.')) return; try{ await api('/api/privacy/delete_observations',{method:'POST'}); showToast('Observations deleted'); navigate(); }catch(e){showToast(e.message)} };
  }

  // ---------- init ----------
  function initTheme(){
    // localStorage can throw (private windows, blocked site data); the theme is a convenience, not a requirement.
    const t = (()=>{ try{ return localStorage.getItem('nb_theme') || ''; }catch(e){ return ''; } })();
    if(t) document.documentElement.setAttribute('data-theme', t);
    $('#themeBtn').onclick = ()=>{
      const cur = document.documentElement.getAttribute('data-theme')==='dark' ? '' : 'dark';
      if(cur) document.documentElement.setAttribute('data-theme', cur); else document.documentElement.removeAttribute('data-theme');
      try{ localStorage.setItem('nb_theme', cur); }catch(e){ /* preference just won't persist */ }
    };
  }

  async function boot(){
    initTheme();
    try { state.meta = await api('/api/meta'); } catch(e){ /* lazy */ }
    if(!location.hash) location.hash = '#/overview';
    navigate();
  }
  boot();
})();
