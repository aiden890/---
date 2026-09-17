"use strict";

// ===== Inference 영상: canonical data-driven groups =====
// 새 영상과 판정 정보는 inference_versions.json에만 추가한다.
fetch("inference_versions.json").then(r=>{
  if(!r.ok)throw new Error(`HTTP ${r.status}`);
  return r.json();
}).then(d=>{
  const esc=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const versions=d.versions||[], root=document.getElementById("infvid-groups");
  if(!versions.length){root.innerHTML='<span class="dim">등록된 inference 영상이 없습니다.</span>';return;}

  const groups=[
    {
      id:"e2e-transition",
      title:"End-to-End skill transition",
      desc:"Manager 배선은 검증됐지만 VLM earliest-boundary 품질은 낮다. 각 카드에서 timeout과 false-positive를 함께 확인한다."
    },
    {
      id:"verifier-unit",
      title:"Verifier unit / calibration",
      desc:"기존 v2 verifier 단품 및 endpoint calibration 자료. E2E manager handoff 결과와 구분한다."
    }
  ];
  const factLabels={current_skill:"Current skill",boundary_verdict:"Boundary verdict",boundary_quality:"Boundary precision / recall",transition:"Transition / retry",strict_gate:"Strict gate",sim_gt:"Sim GT"};
  const renderCard=v=>{
    const facts=v.facts||{};
    const factRows=Object.keys(factLabels).filter(k=>facts[k]!=null).map(k=>{
      const bad=k==="boundary_quality"||(k==="sim_gt"&&String(facts[k]).toUpperCase()==="FALSE")||(k==="strict_gate"&&facts[k]==="FAIL")||String(facts[k]).includes("TIMEOUT");
      return `<div><span>${factLabels[k]}</span><b class="${bad?'fact-bad':'fact-ok'}">${esc(facts[k])}</b></div>`;
    }).join("");
    const e2e=v.group==="e2e-transition";
    const warning=v.quality_warning?" is-warning":"";
    return `<article class="inference-card ${e2e?'is-e2e':''}${warning}">
      <video src="${esc(v.video)}" controls muted loop preload="metadata" playsinline aria-label="${esc(v.label)}"></video>
      <h3>${esc(v.label)}</h3>
      ${factRows?`<div class="inference-facts">${factRows}</div>`:""}
      ${v.quality_warning?`<div class="quality-warning">${esc(v.quality_warning)}</div>`:""}
      <p>${esc(v.desc||"")}</p>
      <small class="dim">${esc([v.stage,v.date].filter(Boolean).join(" · "))}</small>
    </article>`;
  };
  root.innerHTML=groups.map(g=>{
    const items=versions.filter(v=>(v.group||"verifier-unit")===g.id);
    if(!items.length)return"";
    return `<section class="inference-group" data-group="${g.id}">
      <div class="inference-group-head"><h2>${g.title}</h2><p class="dim">${g.desc}</p></div>
      <div class="inference-grid">${items.map(renderCard).join("")}</div>
    </section>`;
  }).join("");
}).catch(()=>{const p=document.getElementById("infvid-groups");if(p)p.innerHTML='<span class="no">inference_versions.json 로드 실패</span>';});
