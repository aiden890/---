"use strict";

// tab switch
function openHashTab(){
  const id=location.hash.slice(1), button=document.querySelector(`[data-tab="${id}"]`);
  if(button) button.click();
}
window.addEventListener('DOMContentLoaded',openHashTab);
window.addEventListener('hashchange',openHashTab);
document.querySelectorAll('.nav button[data-tab]').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById(b.dataset.tab).classList.add('active');
  };
});
// board (running / agent todo / user todo)
const BADGE={run:'b-run',wait:'b-wait',rev:'b-rev',blk:'b-blk',ok:'b-ok'};
const BLABEL={run:'진행',wait:'대기',rev:'리뷰',blk:'블로커',ok:'완료'};
fetch('board.json').then(r=>r.json()).then(b=>{
  const P=b.purpose||{};
  document.getElementById('b-purpose').innerHTML=P.north_star?`<div class="tt">${P.north_star}</div>`:'<div class="dim">없음</div>';
  document.getElementById('b-agent').innerHTML=(b.agent_todo||[]).map(x=>`
    <div class="item"><div class="tt">${x.title}${x.note?`<div class="nt">${x.note}</div>`:''}</div><span class="badge ${BADGE[x.badge]||'b-wait'}">${BLABEL[x.badge]||''}</span></div>`).join('')||'<div class="dim">없음</div>';
  document.getElementById('b-user').innerHTML=(b.user_todo||[]).map(x=>`
    <div class="item"><span class="chk ${x.done?'on':''}"></span><div class="tt" style="${x.done?'color:var(--dim)':''}"><span style="${x.done?'text-decoration:line-through':''}">${x.title}</span>${x.why?`<div class="nt" style="font-size:12px"><b style="color:var(--acc)">왜:</b> ${x.why}</div>`:''}</div></div>`).join('')||'<div class="dim">없음</div>';
  document.getElementById('b-note').innerHTML=(b.note||[]).map(n=>`<div class="item"><span style="color:var(--acc);flex:none">•</span><div class="tt nt" style="font-size:13px">${n}</div></div>`).join('')||'<div class="dim">없음</div>';
}).catch(e=>{});
// load demos (grouped by task) then benchmark tasks
let TASKS=[], DEMOS={}, filt='all', sortK=null, sortDir=1;
function srColor(sr){ if(sr>=70)return'var(--ok)'; if(sr>=40)return'var(--run)'; if(sr>=15)return'#fb923c'; return'var(--red)'; }
function renderTasks(){
  const q=(document.getElementById('tsearch').value||'').toLowerCase();
  let rows=TASKS.filter(t=>{
    if(filt==='eval') return t.eval && t.name.toLowerCase().includes(q);
    return (filt==='all'||t.cat===filt)&&t.name.toLowerCase().includes(q);
  });
  if(sortK){
    rows=rows.slice().sort((a,b)=>{
      let x=a[sortK], y=b[sortK];
      if(x==null&&y==null)return 0; if(x==null)return 1; if(y==null)return -1; // nulls last
      if(typeof x==='string')return x.localeCompare(y)*sortDir;
      return (x-y)*sortDir;
    });
  }
  document.getElementById('task-body').innerHTML=rows.map((t,i)=>{
    const sr = t.sr==null ? '<span class="dim">—</span>'
      : `<span class="bar" style="width:${Math.max(t.sr*0.6,2)}px;background:${srColor(t.sr)}"></span>${t.sr.toFixed(0)}%`;
    const ds=DEMOS[t.name]||[];
    const demo = ds.length ? `<a href="#" onclick="toggleDemo('${t.name}',this);return false">▶ ${ds.length}</a>` : '<span class="dim">—</span>';
    const main=`<tr data-task="${t.name}"><td class="dim">${i+1}</td><td>${t.name}</td><td><span class="dim">${t.cat}</span></td>
      <td class="sr">${sr}</td><td class="dim">${t.h||'—'}</td><td>${demo}</td></tr>`;
    const drop = ds.length ? `<tr class="drop" id="drop-${t.name}" style="display:none"><td colspan="6" style="padding:0 10px 14px">
      <div class="grid">${ds.map(d=>`<div class="card" style="margin:0">
        <video src="${d.file}" controls preload="metadata"></video>
        <div class="vcap"><b>${d.skill}</b> ${d.success?'<span class="ok">성공</span>':'<span class="no">실패</span>'}</div>
        <div class="vsub">${d.instr}${d.reason?'<br>사유: '+d.reason:''}</div></div>`).join('')}</div></td></tr>` : '';
    return main+drop;
  }).join('');
  // header arrows
  document.querySelectorAll('th.sort').forEach(h=>{
    const a=h.querySelector('.ar'); if(a)a.remove();
    if(h.dataset.k===sortK) h.insertAdjacentHTML('beforeend',`<span class="ar">${sortDir>0?'▲':'▼'}</span>`);
  });
}
function toggleDemo(name,el){
  const r=document.getElementById('drop-'+name); if(!r)return;
  const open=r.style.display==='none';
  r.style.display=open?'table-row':'none';
  el.textContent=(open?'▼ ':'▶ ')+(DEMOS[name]||[]).length;
}
Promise.all([
  fetch('demos.json').then(r=>r.json()).catch(()=>[]),
  fetch('robocasa_tasks.json').then(r=>r.json())
]).then(([demos,d])=>{
  demos.forEach(x=>{(DEMOS[x.task]=DEMOS[x.task]||[]).push(x);});
  TASKS=d.tasks;
  document.getElementById('k-total').textContent=d.total;
  document.getElementById('k-eval').textContent=d.eval_count;
  document.getElementById('k-atomic').textContent=d.atomic;
  document.getElementById('k-comp').textContent=d.composite;
  renderTasks();
}).catch(e=>{document.getElementById('task-body').innerHTML='<tr><td colspan=6 class="dim">데이터 로드 실패</td></tr>'});
document.getElementById('tsearch').addEventListener('input',renderTasks);
document.querySelectorAll('.filt').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.filt').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); filt=b.dataset.f; renderTasks();
});
document.querySelectorAll('th.sort').forEach(h=>h.onclick=()=>{
  const k=h.dataset.k;
  if(sortK===k){ sortDir=-sortDir; } else { sortK=k; sortDir = (k==='sr'||k==='h') ? -1 : 1; }
  renderTasks();
});
// ===== RL 환경 리워드 동기화 뷰어 =====
fetch('rlenv.json').then(r=>r.json()).then(RL=>{
  const LAB=RL.predicate_labels, clips=RL.clips;
  const tabs=document.getElementById('rl-clip-tabs');
  const vid=document.getElementById('rl-video'), seek=document.getElementById('rl-seek');
  let cur=0;
  function fmt(v){ if(v===true)return'<span class="pt">true</span>'; if(v===false)return'<span class="pf">false</span>'; return v; }
  function nearest(clip, fidx){
    // find last timeline entry whose frame <= fidx
    let lo=0,hi=clip.timeline.length-1,ans=0;
    while(lo<=hi){const m=(lo+hi)>>1; if(clip.timeline[m].f<=fidx){ans=m;lo=m+1;}else hi=m-1;}
    return clip.timeline[ans];
  }
  function paint(row, clip, idx){
    // 성공 리워드(terminal)만 누적 표시 — hold shaping은 학습 전용이라 delta에만 표시
    const termCum = clip.timeline.slice(0,idx+1).reduce((s,r)=>s+r.d.term,0);
    document.getElementById('rl-cum').textContent = termCum.toFixed(2);
    document.getElementById('rl-cum').style.color = termCum>0?'var(--ok)':(termCum<0?'var(--red)':'var(--tx)');
    let dl=[];
    if(row.d.term)dl.push(`<span class="fire">스킬 성공 +${row.d.term}</span>`);
    if(row.d.shape)dl.push(`접근 shaping ${row.d.shape>0?'+':''}${row.d.shape}`);
    if(row.d.hold)dl.push(row.d.hold>0?`<span class="pt">정지 홀드 +${row.d.hold}</span>`:`<span class="fire">표류 홀드 ${row.d.hold}</span>`);
    if(row.d.pen)dl.push(`<span class="fire">감점 ${row.d.pen}</span>`);
    if(row.end)dl.push(`<span class="fire">${row.end.note} → 스킬 종료 (최종 ${row.end.total})</span>`);
    document.getElementById('rl-delta').innerHTML=dl.join(' · ')||(row.inactive?'<span class="dim">스킬 종료 후 (리워드 계산 없음 · 표류 관찰 구간)</span>':'<span class="dim">이 스텝 리워드 변화 없음</span>');
    // live target checklist: light a target once its reward channel has fired by now.
    let termFired=false, holdFired=false;
    for(let j=0;j<=idx;j++){ if(clip.timeline[j].d.term>0.5) termFired=true; if(clip.timeline[j].d.hold) holdFired=true; }
    document.getElementById('rl-targets').innerHTML=clip.targets.map(t=>{
      const on = t.key==='__term' ? termFired : (t.key==='__hold' ? holdFired : false);
      return `<div class="tgt"><span class="tk ${on?'on':''}"></span><span>${t.name}</span><span class="tb">+${t.bonus}</span></div>`;
    }).join('');
    document.getElementById('rl-pred').innerHTML=Object.entries(row.p).map(([k,v])=>
      `<tr><td>${LAB[k]||k}</td><td>${fmt(v)}</td></tr>`).join('');
    const tag = row.inactive ? ' · 스킬 종료 후' : (clip.success_step===row.s ? ' · ★성공 판정' : '');
    document.getElementById('rl-stepinfo').textContent=`step ${row.s} / ${clip.n} · frame ${row.f}${tag}`;
  }
  function load(i){
    cur=i; const clip=clips[i];
    [...tabs.children].forEach((b,j)=>b.classList.toggle('active',j===i));
    vid.src=clip.mp4; vid.load();
    const maxf=clip.timeline[clip.timeline.length-1].f;
    seek.max=maxf; seek.value=0;
    document.getElementById('rl-meta').innerHTML=`<b>${clip.case||''}</b> · 목표: ${clip.goal} · 판정 <b>${clip.outcome}</b> · 학습 리워드 합계 <b>${clip.total}</b>`;
    paint(clip.timeline[0], clip, 0);
  }
  function idxFromFrame(clip, fidx){
    let lo=0,hi=clip.timeline.length-1,ans=0;
    while(lo<=hi){const m=(lo+hi)>>1; if(clip.timeline[m].f<=fidx){ans=m;lo=m+1;}else hi=m-1;}
    return ans;
  }
  function syncFromFrame(fidx){ const clip=clips[cur]; const idx=idxFromFrame(clip,fidx); paint(clip.timeline[idx],clip,idx); }
  tabs.innerHTML=''; clips.forEach((c,i)=>{const b=document.createElement('button');b.innerHTML=`${c.label} <span style="opacity:.6;font-size:11px">${c.outcome}</span>`;b.title=c.case||'';b.onclick=()=>load(i);tabs.appendChild(b);});
  vid.addEventListener('timeupdate',()=>{ const clip=clips[cur]; const fidx=Math.round(vid.currentTime*clip.fps); seek.value=Math.min(fidx,seek.max); syncFromFrame(fidx); });
  seek.addEventListener('input',()=>{ const clip=clips[cur]; const fidx=+seek.value; vid.currentTime=fidx/clip.fps; syncFromFrame(fidx); });
  load(0);
}).catch(e=>{});

// ===== 스킬 경계(hold) 뷰어 =====
fetch('rlenv_hold.json').then(r=>r.json()).then(RL=>{
  const LAB=RL.predicate_labels, clips=RL.clips;
  const tabs=document.getElementById('hold-tabs');
  const vid=document.getElementById('hold-video'), seek=document.getElementById('hold-seek');
  let cur=0;
  function fmt(v){ if(v===true)return'<span class="pt">true</span>'; if(v===false)return'<span class="pf">false</span>'; return v; }
  function idxFromFrame(clip,fidx){let lo=0,hi=clip.timeline.length-1,ans=0;while(lo<=hi){const m=(lo+hi)>>1;if(clip.timeline[m].f<=fidx){ans=m;lo=m+1;}else hi=m-1;}return ans;}
  function paint(row,clip){
    const post = row.phase==='post' || row.s>(clip.success_step||1e9);
    const ph=document.getElementById('hold-phase');
    ph.textContent = clip.success_step==null ? '성공 없음' : (post?'성공 후 (hold)':'스킬 수행 중');
    ph.style.color = post?'var(--run)':'var(--tx)';
    document.getElementById('hold-verdict').innerHTML = post && clip.verdict==='KEEPS_MOVING'
      ? `<span class="fire">계속 움직임 → ${clip.drift||'다음 동작으로 표류'}</span>`
      : (post?'<span class="pt">정지 유지</span>':'<span class="dim">성공 지점 이후를 보세요</span>');
    document.getElementById('hold-pred').innerHTML=Object.entries(row.p).map(([k,v])=>`<tr><td>${LAB[k]||k}</td><td>${fmt(v)}</td></tr>`).join('');
    const tag = row.s===clip.success_step ? ' · ★성공 판정' : (post?' · hold':'');
    document.getElementById('hold-stepinfo').textContent=`step ${row.s} / ${clip.n}${tag}  (성공 step ${clip.success_step??'-'})`;
    document.getElementById('hold-video').style.boxShadow = post ? '0 0 0 3px var(--run)' : 'none';
  }
  function load(i){
    cur=i; const clip=clips[i];
    [...tabs.children].forEach((b,j)=>b.classList.toggle('active',j===i));
    vid.src=clip.mp4; vid.load();
    seek.max=clip.timeline[clip.timeline.length-1].f; seek.value=0;
    const vv = clip.success_step==null ? '이 클립은 성공하지 못함' : (clip.verdict==='KEEPS_MOVING'?`<b style="color:var(--run)">판정: 성공 후 계속 움직임</b> — ${clip.drift}`:'<b class="pt">판정: 정지 유지</b>');
    document.getElementById('hold-meta').innerHTML=`<b>${clip.label}</b> · ${clip.case} · 지시 ${clip.instr} · ${vv}`;
    paint(clip.timeline[0],clip);
  }
  function sync(fidx){const clip=clips[cur];paint(clip.timeline[idxFromFrame(clip,fidx)],clip);}
  tabs.innerHTML=''; clips.forEach((c,i)=>{const b=document.createElement('button');b.innerHTML=`${c.label}${c.label==='place'?' '+(c.success_step!=null?'✓':'✗'):''}`;b.title=c.case||'';b.onclick=()=>load(i);tabs.appendChild(b);});
  vid.addEventListener('timeupdate',()=>{const clip=clips[cur];const fidx=Math.round(vid.currentTime*clip.fps);seek.value=Math.min(fidx,seek.max);sync(fidx);});
  seek.addEventListener('input',()=>{const clip=clips[cur];const fidx=+seek.value;vid.currentTime=fidx/clip.fps;sync(fidx);});
  load(0);
}).catch(e=>{});

// ===== searchable experiment index =====
fetch('experiments.json').then(r=>r.json()).then(db=>{
  const exps=[...(db.experiments||[])].sort((a,b)=>(b.date+b.id).localeCompare(a.date+a.id));
  const search=document.getElementById('exp-search'), task=document.getElementById('exp-task'), model=document.getElementById('exp-model'), status=document.getElementById('exp-status');
  const uniq=k=>[...new Set(exps.map(e=>e[k]).filter(Boolean))].sort();
  task.innerHTML+=[...uniq('task')].map(x=>`<option>${x}</option>`).join('');model.innerHTML+=[...uniq('model')].map(x=>`<option>${x}</option>`).join('');
  const badge=s=>s==='complete'?'<span class="badge b-ok">완료</span>':s==='running'?'<span class="badge b-run">진행</span>':'<span class="badge b-rev">Pilot</span>';
  document.getElementById('exp-kpi').innerHTML=`<div><div class="n">${exps.length}</div><div class="l">전체 실험</div></div><div><div class="n">${exps.filter(e=>e.status==='complete').length}</div><div class="l">완료</div></div><div><div class="n">${new Set(exps.map(e=>e.task)).size}</div><div class="l">Task</div></div><div><div class="n">${new Set(exps.map(e=>e.model)).size}</div><div class="l">모델</div></div>`;
  function render(){
    const q=search.value.trim().toLowerCase();
    const rows=exps.filter(e=>(!task.value||e.task===task.value)&&(!model.value||e.model===model.value)&&(!status.value||e.status===status.value)&&(!q||[e.id,e.title,e.question,e.summary,e.model,e.task,...(e.tags||[])].join(' ').toLowerCase().includes(q)));
    document.getElementById('exp-empty').style.display=rows.length?'none':'block';
    document.getElementById('exp-list').innerHTML=rows.map(e=>{
      const wb=e.wandb;
      const wbUrl=wb?(wb.workspace||wb.project):null;
      const wbBtn=wbUrl?`<a href="${wbUrl}" target="_blank" onclick="event.stopPropagation()" style="display:inline-flex;align-items:center;gap:3px;margin-top:4px;font-size:11px;color:var(--acc);text-decoration:none;opacity:.85" title="wandb 대시보드 열기">📊 wandb</a>`:'';
      const armRuns=(e.arms||[]).filter(a=>(a.wandb_runs||[]).length).map(a=>{
        const label=a.exp||a.name||'arm';
        const links=a.wandb_runs.map((u,i)=>`<a href="${u}" target="_blank" onclick="event.stopPropagation()" style="color:var(--acc);text-decoration:none;margin-right:4px" title="${u}">run${a.wandb_runs.length>1?i+1:''}</a>`).join('');
        return `<span style="font-size:11px;color:var(--dim);margin-right:8px">${label}: ${links}</span>`;
      }).join('');
      const armSection=armRuns?`<div style="margin-top:4px;line-height:1.6">${armRuns}</div>`:'';
      const goNav=e.page?`<small>상세 →</small>`:'';
      return `<a class="exp-list-row" href="${e.page||'#'}" ${!e.page?'onclick="return false"':''}><span class="exp-date"><b>${e.date}</b><small>${e.id}</small></span><span class="exp-title"><b>${e.title}</b><small>${e.summary}</small>${armSection}</span><span class="exp-context"><b>${e.task}</b><small>${e.model}</small></span><span class="go">${badge(e.status)}${goNav}${wbBtn}</span></a>`;
    }).join('');
  }
  [search,task,model,status].forEach(x=>x.addEventListener(x===search?'input':'change',render));render();
}).catch(()=>document.getElementById('exp-list').innerHTML='<p class="no">실험 목록을 불러오지 못했습니다.</p>');

// dataset/report 탭은 제거됨(operator 2026-09-16). 관련 렌더 코드도 삭제.

