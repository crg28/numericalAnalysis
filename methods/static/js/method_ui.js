(function(){
  // Helpers
  function $(sel, root=document){ return root.querySelector(sel); }
  function $all(sel, root=document){ return Array.from(root.querySelectorAll(sel)); }

  // Inicializa una barra de ecuación (soporta historia y preview LaTeX)
  function initEqBar({rootId, storageKey, onGraph}) {
    const root = document.getElementById(rootId);
    if(!root) return;

    const input = $('.eq-input', root);
    const preview = $('.preview', root);
    const historySel = $('.eq-history', root);
    const btnGraph = $('.btn-graph', root);
    const btnSave  = $('.btn-save', root);
    const btnClear = $('.btn-clear', root);

    // Cargar historial
    const history = JSON.parse(localStorage.getItem(storageKey) || '[]');
    renderHistory();

    function renderHistory() {
      historySel.innerHTML = '';
      const opt0 = document.createElement('option');
      opt0.value = ''; opt0.textContent = 'History…';
      historySel.appendChild(opt0);
      history.forEach(v=>{
        const o = document.createElement('option');
        o.value = v; o.textContent = v;
        historySel.appendChild(o);
      });
    }

    historySel.addEventListener('change', ()=>{
      if(historySel.value){ input.value = historySel.value; updatePreview(); }
    });

    function updatePreview(){
      // Muestra el LaTeX simple: f(x)= <texto>
      preview.innerHTML = `\\(\\displaystyle ${input.value.replaceAll('*','\\cdot ')}\\)`;
      if(window.MathJax && MathJax.typesetPromise){ MathJax.typesetPromise([preview]); }
    }

    input.addEventListener('input', updatePreview);
    updatePreview();

    btnGraph?.addEventListener('click', ()=>{
      if(onGraph) onGraph(input.value);
    });

    btnSave?.addEventListener('click', ()=>{
      const v = input.value.trim();
      if(!v) return;
      if(!history.includes(v)){
        history.unshift(v);
        if(history.length>20) history.pop();
        localStorage.setItem(storageKey, JSON.stringify(history));
        renderHistory();
      }
    });

    btnClear?.addEventListener('click', ()=>{
      input.value = ''; updatePreview();
    });

    return { get: ()=>input.value, set: v=>{ input.value=v; updatePreview(); } };
  }

  // Graficar f(x) usando math.js + Plotly
  function graphFx(containerId, fx, xrange=[-10,10], n=400){
    const plot = document.getElementById(containerId);
    if(!plot) return;
    let compiled;
    try{
      // admitir "sin", "cos", "^", etc.
      compiled = math.compile(fx);
    }catch(e){
      Plotly.react(plot, [], {title: 'Invalid expression'});
      return;
    }
    const xs=[], ys=[];
    const [a,b] = xrange;
    for(let i=0;i<n;i++){
      const x = a + i*(b-a)/(n-1);
      let y = NaN;
      try{ y = compiled.evaluate({x}); }catch(_){}
      xs.push(x); ys.push(y);
    }
    const trace = {x: xs, y: ys, mode:'lines', name: fx};
    Plotly.react(plot, [trace], {
      margin:{l:30,r:10,t:10,b:30},
      xaxis:{title:'x'}, yaxis:{title:'y'},
      displayModeBar:false
    }, {responsive:true});
  }

  // Exponer globalmente para usar desde templates
  window.NA_UI = { initEqBar, graphFx };
})();
