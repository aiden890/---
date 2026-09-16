"use strict";

fetch("inference.json").then(r=>r.json()).then(data=>{
  document.getElementById("inf-assumption").textContent=`학습 완료 가정: ${data.assumption}`;
  document.getElementById("inf-layers").innerHTML=data.layers.map((x,i)=>`<div class="inf-layer"><b>${i+1}. ${x.name}</b><span>${x.role}</span></div>`).join("");
  document.getElementById("inf-skills").innerHTML=data.skills.map(x=>`<tr><td><b>${x.name}</b><br><code>${x.adapter}</code></td><td>${x.args}</td><td>${x.instruction}</td><td><code>${x.done}</code></td></tr>`).join("");
  const steps=document.getElementById("inf-steps"), detail=document.getElementById("inf-detail"), bar=document.getElementById("inf-progress"), prev=document.getElementById("inf-prev"), next=document.getElementById("inf-next");
  let current=0;
  steps.innerHTML=data.trace.map((x,i)=>`<button class="inf-step" data-i="${i}">${i+1}. ${x.phase}<br>${x.skill||"SYSTEM"}</button>`).join("");
  function show(i){current=i;const x=data.trace[i];[...steps.children].forEach((b,j)=>b.classList.toggle("active",j===i));bar.style.width=`${(i+1)/data.trace.length*100}%`;prev.disabled=i===0;next.disabled=i===data.trace.length-1;detail.innerHTML=`<div class="inf-phase">${x.actor} · ${x.phase}</div><h3>${x.title}</h3>${x.skill?`<div class="adapter-key">${x.skill}</div>`:""}<div class="inf-io"><div><b>INPUT</b>${x.input}</div><div><b>OUTPUT</b>${x.output}</div></div><div class="inf-decision">→ ${x.decision}</div>`}
  steps.addEventListener("click",e=>{const b=e.target.closest("button");if(b)show(+b.dataset.i)});prev.onclick=()=>show(current-1);next.onclick=()=>show(current+1);show(0);
}).catch(()=>document.getElementById("inf-detail").innerHTML='<span class="no">inference.json 로드 실패</span>');

// ===== Adapter-free architecture smoke test (t_1d5b404f) =====
fetch("smoke.json").then(r=>r.json()).then(d=>{
  const esc=s=>String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const yn=b=>b?'<span style="color:#3ddc84;font-weight:600">PASS</span>':'<span style="color:#ff5c5c;font-weight:600">FAIL</span>';
  const stat=s=>({SUCCESS:"#3ddc84",TIMEOUT:"#ffb020",FAILED:"#ff5c5c"}[s]||"#8a90a0");

  document.getElementById("smoke-intro").innerHTML=
    `대상: <b>${esc(d.task)}</b> · 목표: ${esc(d.goal)} · split ${esc(d.split)}. `+
    `어댑터 학습 전 <b>전체 skill-conditioned 추론 아키텍처 배선</b>을 검증하는 통합 테스트(정책 성능 벤치마크 아님). `+
    `Planner는 <b>${esc(d.planner.kind)}</b> — 학습된 VLM planner가 아니라 스크립트 스텁(${esc(d.planner.note)}). `+
    `Policy는 pinned base Xiaomi RoboCasa365 VLA, <b>adapter_mode=${esc(d.adapter_mode)}, adapter_checkpoint=${d.adapter_checkpoint===null?"null":esc(d.adapter_checkpoint)}</b>.`;

  document.getElementById("smoke-banner").innerHTML=
    `<div style="margin:12px 0;padding:12px 14px;border-radius:10px;border:1px solid ${d.arch_pass?"#2a5":"#a33"};background:${d.arch_pass?"rgba(61,220,132,.08)":"rgba(255,92,92,.08)"}">`+
    `<b style="font-size:16px">아키텍처 판정: ${d.arch_pass?"✅ PASS":"❌ FAIL"}</b> — `+
    `${d.checks.filter(c=>c.pass).length}/${d.checks.length} 아키텍처 체크 통과 · 오프라인 체크 ${d.offline_checks.passed}/${d.offline_checks.total} 통과`+
    `<br><span class="dim">${esc(d.offline_checks.note)}</span></div>`;

  document.getElementById("smoke-kpi").innerHTML=
    `<div><div class="n">${d.n_episodes}</div><div class="l">에피소드 (seed ${d.seeds.join(",")})</div></div>`+
    `<div><div class="n">${d.checks.filter(c=>c.pass).length}/${d.checks.length}</div><div class="l">아키텍처 체크</div></div>`+
    `<div><div class="n">${d.offline_checks.passed}/${d.offline_checks.total}</div><div class="l">오프라인 체크</div></div>`+
    `<div><div class="n">${d.task_success_count}/${d.n_episodes}</div><div class="l">task 성공(참고)</div></div>`;

  document.getElementById("smoke-checks").innerHTML=d.checks.map(c=>
    `<tr><td><b>${esc(c.name)}</b></td><td>${yn(c.pass)}</td><td class="dim">${esc(c.evidence)}</td></tr>`).join("");

  const tl=ep=>ep.timeline.map(t=>{
    if(t.kind==="PLAN") return `<div class="smk-tl plan"><b>PLAN #${t.planner_calls}</b> → ${t.skill?esc(t.skill):"TERMINATE"} `+
      `${t.args?`<code>${esc(JSON.stringify(t.args))}</code>`:""}<br><span class="dim">obs ${esc(t.obs_ref)} · ${esc(t.rationale)}</span>`+
      `${t.instruction?`<br><i>"${esc(t.instruction)}"</i>`:""}</div>`;
    if(t.kind==="ROUTE") return `<div class="smk-tl route"><b>ROUTE</b> ${esc(t.skill)} · adapter=<b>${esc(t.adapter_mode)}</b>/ckpt=${t.adapter_checkpoint===null?"null":esc(t.adapter_checkpoint)} · budget ${t.max_steps} · can_start=${t.can_start}</div>`;
    if(t.kind==="VERIFY") return `<div class="smk-tl verify"><b>VERIFY</b> ${esc(t.skill)} → <b style="color:${t.decision==="ADVANCE"?"#3ddc84":"#ffb020"}">${esc(t.decision)}</b> <span class="dim">(${t.elapsed} steps · ${esc(t.reason)})</span></div>`;
    if(t.kind==="RESULT") return `<div class="smk-tl result"><b>RESULT</b> ${esc(t.skill)} = <b style="color:${stat(t.status)}">${esc(t.status)}</b> · ${t.steps} steps · terminated_by ${esc(t.terminated_by)} · planner next: ${esc(t.next)}</div>`;
    return "";
  }).join("");

  const body=document.getElementById("smoke-eps");
  body.innerHTML=d.episodes.map(ep=>{
    const seq=ep.skills.map(s=>`<span style="color:${stat(s.status)}">${esc(s.skill.replace("_OBJECT",""))}</span>`).join(" → ");
    const row=`<tr class="smk-row" data-seed="${ep.seed}" style="cursor:pointer"><td>▶ seed ${ep.seed}</td><td>${esc(ep.terminal)}</td><td>${ep.planner_calls}</td><td>${ep.steps_used}</td><td>${seq}</td><td>${ep.task_success?"✅":"—"}</td></tr>`;
    const drop=`<tr class="smk-drop" data-seed="${ep.seed}" style="display:none"><td colspan="6"><div style="display:flex;gap:16px;flex-wrap:wrap">`+
      `<video src="${esc(ep.video)}" controls muted style="width:420px;max-width:100%;border-radius:8px;background:#000"></video>`+
      `<div style="flex:1;min-width:280px;max-height:420px;overflow:auto">${tl(ep)}</div>`+
      `</div></td></tr>`;
    return row+drop;
  }).join("");
  body.addEventListener("click",e=>{
    const r=e.target.closest(".smk-row");if(!r)return;
    const seed=r.dataset.seed, drop=body.querySelector(`.smk-drop[data-seed="${seed}"]`);
    const open=drop.style.display==="none";
    drop.style.display=open?"table-row":"none";
    r.children[0].textContent=(open?"▼":"▶")+` seed ${seed}`;
  });

  const fails=d.episodes.flatMap(ep=>ep.base_failures.map(f=>({seed:ep.seed,...f})));
  document.getElementById("smoke-fail").innerHTML=fails.length?fails.map(f=>
    `<tr><td>seed ${f.seed}</td><td>${esc(f.skill)}</td><td style="color:${stat(f.status)}">${esc(f.status)}</td><td class="dim">${esc(f.reason)}</td></tr>`).join("")
    :`<tr><td colspan="4" class="dim">base-policy 실패 없음</td></tr>`;
}).catch(e=>{const el=document.getElementById("smoke-intro");if(el)el.innerHTML='<span class="no">smoke.json 로드 실패</span>';});
