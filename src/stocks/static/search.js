// Search bar with suggestions: type a symbol or company name, Enter / click opens /report/<SYMBOL>.
function setupSearch(input){
  const box = input.parentElement.querySelector(".suggest");
  let items = [], active = -1, timer = null;
  const go = sym => { if(sym) location.href = "/report/" + encodeURIComponent(sym.trim().toUpperCase()); };
  function draw(){
    box.replaceChildren();
    items.forEach((s,i)=>{
      const d = document.createElement("div"); if(i===active) d.className="act";
      const a = document.createElement("b"); a.textContent = s.symbol;
      const b = document.createElement("span"); b.className="muted small";
      b.textContent = (s.name && s.name!==s.symbol ? s.name : "") + (s.nifty50 ? "" : " (not in NIFTY 50)");
      d.append(a,b); d.onmousedown = e => { e.preventDefault(); go(s.symbol); };
      box.append(d);
    });
    box.style.display = items.length ? "block" : "none";
  }
  input.addEventListener("input", ()=>{
    clearTimeout(timer);
    const q = input.value.trim();
    if(!q){ items=[]; draw(); return; }
    timer = setTimeout(async ()=>{
      try{ const r = await fetch("/api/search?q="+encodeURIComponent(q)); items = r.ok ? await r.json() : []; }
      catch(e){ items = []; }
      active = items.length ? 0 : -1; draw();
    }, 150);
  });
  input.addEventListener("keydown", e=>{
    if(e.key==="ArrowDown"){ active=Math.min(active+1,items.length-1); draw(); e.preventDefault(); }
    else if(e.key==="ArrowUp"){ active=Math.max(active-1,0); draw(); e.preventDefault(); }
    else if(e.key==="Enter"){ go(active>=0 && items[active] ? items[active].symbol : input.value); }
    else if(e.key==="Escape"){ items=[]; draw(); }
  });
  input.addEventListener("blur", ()=>setTimeout(()=>{box.style.display="none";},120));
}
const fmt = (v, d=2) => v==null || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d});
const rs = v => v==null ? "–" : "₹" + Number(v).toLocaleString("en-IN",{maximumFractionDigits:0});
const sgn = v => v==null ? "–" : (v>0?"+":"") + fmt(v);
const cls = v => v==null ? "" : v>0 ? "up" : v<0 ? "down" : "";
function el(tag, cls, text){const e=document.createElement(tag); if(cls) e.className=cls; if(text!=null) e.textContent=text; return e;}
