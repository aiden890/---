"use strict";

// ===== Inference 영상: 버전 선택 재생기 =====
// 데이터: inference_versions.json (versions[] = {id,label,stage,date,desc,video})
// 새 모델/아키텍처 버전은 json에 항목만 추가하면 드롭다운에 자동으로 뜬다.
fetch("inference_versions.json").then(r=>r.json()).then(d=>{
  const esc=s=>String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const versions=(d.versions||[]);
  const sel=document.getElementById("infvid-select");
  const player=document.getElementById("infvid-player");
  const desc=document.getElementById("infvid-desc");
  const badge=document.getElementById("infvid-badge");
  if(!versions.length){ player.innerHTML='<span class="dim">등록된 인퍼런스 영상이 없습니다.</span>'; return; }

  sel.innerHTML=versions.map((v,i)=>`<option value="${i}">${esc(v.label)}</option>`).join("");

  function show(i){
    const v=versions[i];
    player.innerHTML=`<video src="${esc(v.video)}" controls muted loop preload="auto" playsinline style="width:520px;max-width:100%;border-radius:8px;background:#000"></video>`;
    desc.innerHTML=esc(v.desc||"");
    badge.textContent=[v.stage,v.date].filter(Boolean).join(" · ");
  }
  sel.addEventListener("change",e=>show(+e.target.value));
  show(0);
}).catch(()=>{const p=document.getElementById("infvid-player");if(p)p.innerHTML='<span class="no">inference_versions.json 로드 실패</span>';});
