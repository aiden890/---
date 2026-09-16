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

// ===== 인퍼런스 영상 (아키텍처 실행) =====
// 슬림화(operator 요청): 작동 방식 + 영상만. 하네스 8항목 검증표·에피소드 상세·obs verifier
// 상세는 제거. 근거 데이터는 smoke.json / obs_verifier.json + REPORT/ 에 보존.
fetch("smoke.json").then(r=>r.json()).then(d=>{
  const esc=s=>String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const hv=d.harness_verification||{};
  const src=hv.verify_video_all||d.verify_video_all;
  const box=document.getElementById("smoke-hv-video");
  if(box) box.innerHTML=src
    ? `<video src="${esc(src)}" controls muted loop preload="metadata" playsinline style="width:520px;max-width:100%;border-radius:8px;background:#000"></video>`
    : `<span class="dim">영상 로드 대기중</span>`;
}).catch(()=>{const el=document.getElementById("smoke-hv-video");if(el)el.innerHTML='<span class="no">smoke.json 로드 실패</span>';});
