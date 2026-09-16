// obs-verifier calibration card (task t_32a4f9b6)
// data: obs_verifier_calibration.json
(function(){
  fetch("obs_verifier_calibration.json").then(r=>r.json()).then(d=>{
    const set=(id,txt)=>{const e=document.getElementById(id); if(e) e.textContent=txt;};
    set("obsv-why", d.why);
    set("obsv-cfg", "성공게이트 설정 — PLACE: "+d.success_gate_config.PLACE_OBJECT+" | GRASP: "+d.success_gate_config.GRASP_OBJECT);
    set("obsv-loop", "개선 루프: "+d.loop);
    set("obsv-inv", "obs-only 무결성: "+d.obs_only_invariant);
    set("obsv-caveat", "주의: "+d.caveat);
    set("obsv-tests", "단위테스트: "+d.unit_tests+" · corpus: "+d.corpus);
    const roles=document.getElementById("obsv-roles");
    if(roles && d.roles){
      roles.innerHTML=d.roles.map(r=>
        '<div style="padding:8px 12px;border:1px solid var(--bd,#333);border-radius:8px;margin-bottom:6px">'+
        '<b>'+r.role+'</b> <code>'+r.module+'</code><br>'+
        '<span class="dim" style="font-size:13px">강도: '+r.strictness+'<br>용도: '+r.purpose+'</span></div>').join("");
    }
    const tb=document.querySelector("#obsv-ba tbody");
    if(tb && d.before_after){
      tb.innerHTML=d.before_after.map(r=>{
        const arrow=(a,b)=>a+" → <b>"+b+"</b>";
        const good = r.after_precision>=0.9;
        return '<tr><td><b>'+r.skill+'</b></td>'+
          '<td>'+arrow(r.before_precision, r.after_precision)+(good?' ✅':'')+'</td>'+
          '<td>'+arrow(r.before_fp, r.after_fp)+'</td>'+
          '<td>'+(r.after_recall)+'</td>'+
          '<td>'+r.tp+'/'+r.fn+'/'+r.tn+'</td></tr>';
      }).join("");
    }
  }).catch(()=>{const c=document.getElementById("obsv-calib-card"); if(c) c.style.display="none";});
})();
