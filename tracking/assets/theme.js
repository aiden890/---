"use strict";
(function(){
  const KEY="aiden-ui-theme";
  const THEMES=new Set(["comfort","dark"]);
  let stored=null;
  try{stored=localStorage.getItem(KEY);}catch(_e){}
  const initial=THEMES.has(stored)?stored:"comfort";
  document.documentElement.dataset.theme=initial;

  function mount(){
    const host=document.querySelector("header .nav, header .bar");
    if(!host||host.querySelector(".theme-control"))return;
    const label=document.createElement("label");
    label.className="theme-control";
    label.innerHTML='<span>Theme</span><select aria-label="색상 테마"><option value="comfort">Comfort</option><option value="dark">Dark</option></select>';
    const select=label.querySelector("select");
    select.value=document.documentElement.dataset.theme;
    select.addEventListener("change",()=>{
      document.documentElement.dataset.theme=select.value;
      try{localStorage.setItem(KEY,select.value);}catch(_e){}
    });
    const live=host.querySelector(".live");
    host.insertBefore(label,live||null);
  }
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",mount,{once:true});
  else mount();
})();
