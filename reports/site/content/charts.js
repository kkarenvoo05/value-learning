// Small dependency-free SVG charts for the value-learning progress pages.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmt = (v, d = 3) => (v == null || isNaN(v) ? "–" : (+v).toFixed(d));
  const nice = (lo, hi, n = 5) => {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
    const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, out = [];
    for (let v = a; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  function tipFor(host) {
    host.classList.add("rel");
    let t = host.querySelector(".tip");
    if (!t) { t = document.createElement("div"); t.className = "tip"; host.appendChild(t); }
    return {
      show(html, x, y) {
        t.innerHTML = html; t.style.display = "block";
        const w = t.offsetWidth, hw = host.clientWidth;
        t.style.left = Math.min(Math.max(0, x + 14), hw - w) + "px"; t.style.top = (y + 12) + "px";
      },
      hide() { t.style.display = "none"; }
    };
  }

  // Horizontal bars with optional ±sd whiskers. spec: {rows:[{label,value,sd,color,note}], unit, title, domain}
  window.barChart = function (host, spec) {
    const rows = spec.rows, lw = spec.labelWidth || 230, W = spec.width || 760, rh = 30, top = 6, H = top + rh * rows.length + 28;
    const maxv = Math.max(...rows.map(r => r.value + (r.sd || 0)), 0), minv = Math.min(0, ...rows.map(r => r.value - (r.sd || 0)));
    const ticks = nice(spec.domain ? spec.domain[0] : minv, spec.domain ? spec.domain[1] : maxv * 1.08, 4);
    const x0 = ticks[0], x1 = ticks[ticks.length - 1], R = W - 70;
    const x = v => lw + (R - lw) * (v - x0) / (x1 - x0);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.title || "bar chart" });
    ticks.forEach(t => { el("line", { x1: x(t), x2: x(t), y1: top, y2: H - 22, class: "gl" }, svg);
      el("text", { x: x(t), y: H - 6, "text-anchor": "middle", class: "mut" }, svg).textContent = +t.toFixed(3); });
    el("line", { x1: x(0), x2: x(0), y1: top, y2: H - 22, class: "ax" }, svg);
    const tip = tipFor(host);
    rows.forEach((r, j) => {
      const y = top + j * rh, cy = y + rh / 2;
      el("text", { x: lw - 10, y: cy + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = r.label;
      const a = x(Math.min(0, r.value)), b = x(Math.max(0, r.value));
      el("rect", { x: a, y: y + 7, width: Math.max(1, b - a), height: rh - 14, rx: 4, fill: r.color || "var(--s1)", opacity: r.dim ? 0.45 : 1 }, svg);
      if (r.sd) { const lo = x(r.value - r.sd), hi = x(r.value + r.sd);
        el("line", { x1: lo, x2: hi, y1: cy, y2: cy, stroke: "var(--fg)", "stroke-width": 1.5 }, svg);
        [lo, hi].forEach(v => el("line", { x1: v, x2: v, y1: cy - 4, y2: cy + 4, stroke: "var(--fg)", "stroke-width": 1.5 }, svg)); }
      el("text", { x: x(Math.max(0, r.value) + (r.sd || 0)) + 8, y: cy + 4, class: "lbl" }, svg).textContent =
        fmt(r.value, spec.digits ?? 3) + (r.sd ? " ± " + fmt(r.sd, spec.digits ?? 3) : "");
      const hit = el("rect", { x: 0, y, width: W, height: rh, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", ev => { const b = host.getBoundingClientRect();
        tip.show(`<b>${r.label}</b><br>${spec.unit || "value"}: ${fmt(r.value, spec.digits ?? 3)}${r.sd ? " ± " + fmt(r.sd, spec.digits ?? 3) + " (sd)" : ""}${r.note ? "<br>" + r.note : ""}`, ev.clientX - b.left, ev.clientY - b.top); });
      hit.addEventListener("mouseleave", () => tip.hide());
    });
    host.appendChild(svg);
  };

  // Scatter with error crosses. spec: {pts:[{label,x,y,xsd,ysd,color,short}], xlab, ylab, xdir, ydir}
  window.scatter = function (host, spec) {
    const W = spec.width || 760, H = spec.height || 420, L = 62, Rr = W - 20, T = 16, B = H - 48;
    const xs = spec.pts.flatMap(p => [p.x - (p.xsd || 0), p.x + (p.xsd || 0)]), ys = spec.pts.flatMap(p => [p.y - (p.ysd || 0), p.y + (p.ysd || 0)]);
    const pad = (a, b) => [a - (b - a) * 0.12, b + (b - a) * 0.12];
    const [xa, xb] = pad(Math.min(...xs), Math.max(...xs)), [ya, yb] = pad(Math.min(...ys), Math.max(...ys));
    const xt = spec.xticks || nice(xa, xb, 5), yt = spec.yticks || nice(ya, yb, 5);
    const x = v => L + (Rr - L) * (v - xt[0]) / (xt[xt.length - 1] - xt[0]);
    const y = v => B - (B - T) * (v - yt[0]) / (yt[yt.length - 1] - yt[0]);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.title || "scatter" });
    xt.forEach(t => { el("line", { x1: x(t), x2: x(t), y1: T, y2: B, class: "gl" }, svg);
      el("text", { x: x(t), y: B + 18, "text-anchor": "middle", class: "mut" }, svg).textContent = +t.toFixed(3); });
    yt.forEach(t => { el("line", { x1: L, x2: Rr, y1: y(t), y2: y(t), class: "gl" }, svg);
      el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end", class: "mut" }, svg).textContent = +t.toFixed(3); });
    el("line", { x1: L, x2: Rr, y1: B, y2: B, class: "ax" }, svg);
    el("text", { x: (L + Rr) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = spec.xlab;
    const yl = el("text", { x: 14, y: (T + B) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(T + B) / 2})` }, svg); yl.textContent = spec.ylab;
    if (spec.better) el("text", { x: Rr - 4, y: T + 12, "text-anchor": "end", class: "mut" }, svg).textContent = spec.better;
    const tip = tipFor(host);
    (spec.ref || []).forEach(p => el("circle", { cx: x(p.x), cy: y(p.y), r: 5, fill: "none", stroke: "var(--fg3)", "stroke-width": 1.5 }, svg));
    spec.pts.forEach(p => {
      const g = el("g", {}, svg), cx = x(p.x), cy = y(p.y);
      if (p.xsd) el("line", { x1: x(p.x - p.xsd), x2: x(p.x + p.xsd), y1: cy, y2: cy, stroke: p.color, "stroke-width": 1.5, opacity: .7 }, g);
      if (p.ysd) el("line", { x1: cx, x2: cx, y1: y(p.y - p.ysd), y2: y(p.y + p.ysd), stroke: p.color, "stroke-width": 1.5, opacity: .7 }, g);
      el("circle", { cx, cy, r: spec.r || 7, fill: p.color, stroke: "var(--surface)", "stroke-width": 2 }, g);
      el("text", { x: cx + (p.dx ?? 11), y: cy + (p.dy ?? -9), "text-anchor": p.anchor || "start", class: "lbl" }, g).textContent = p.short ?? p.label;
      el("circle", { cx, cy, r: 16, fill: "transparent" }, g);
      g.addEventListener("mousemove", ev => { const b = host.getBoundingClientRect();
        tip.show(`<b>${p.label}</b><br>${spec.xname}: ${fmt(p.x)}${p.xsd ? " ± " + fmt(p.xsd) : ""}<br>${spec.yname}: ${fmt(p.y)}${p.ysd ? " ± " + fmt(p.ysd) : ""}${p.note ? "<br>" + p.note : ""}`, ev.clientX - b.left, ev.clientY - b.top); });
      g.addEventListener("mouseleave", () => tip.hide());
    });
    host.appendChild(svg);
  };

  // Episode player: value traces + scrubbable camera frames.
  window.episodePlayer = function (root, data, opts) {
    const SER = opts.series; // [{key,label,color}]
    const tabs = root.querySelector(".tabs"), chart = root.querySelector(".epchart"), legend = root.querySelector(".legend");
    const camA = root.querySelector(".cam.a"), camW = root.querySelector(".cam.w"), ro = root.querySelector(".readout"), cap = root.querySelector(".epcap"), mt = root.querySelector(".epmetrics");
    const hidden = new Set();
    let cur = opts.episodes[0].id, idx = 0;
    opts.episodes.forEach(ep => { const b = document.createElement("button"); b.textContent = ep.tab; b.setAttribute("aria-pressed", ep.id === cur);
      b.onclick = () => { cur = ep.id; tabs.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); draw(); }; tabs.appendChild(b); });
    legend.innerHTML = `<span><i class="dash"></i>target (true discounted return)</span>` + SER.map(s => `<span data-k="${s.key}"><i style="background:${s.color}"></i>${s.label}</span>`).join("");
    legend.querySelectorAll("span[data-k]").forEach(sp => sp.onclick = () => { const k = sp.dataset.k; hidden.has(k) ? hidden.delete(k) : hidden.add(k); sp.classList.toggle("off"); draw(); });
    function setFrame(d, i) {
      idx = i; const px = d.px, n = d.t.length, size = camA.clientWidth || 140, sc = size / px;
      [[camA, 0], [camW, 1]].forEach(([c, row]) => { c.style.backgroundImage = `url(${opts.assetBase}ep${cur}_frames.jpg)`;
        c.style.backgroundSize = `${n * size}px ${2 * size}px`; c.style.backgroundPosition = `${-i * size}px ${-row * size}px`; });
      ro.innerHTML = `<div><span>time</span><b>${d.t[i].toFixed(1)} s</b></div><div><span>target</span><b>${fmt(d.target[i], 2)}</b></div>` +
        SER.map(s => `<div><span><i style="display:inline-block;width:10px;height:3px;background:${s.color};margin-right:6px;vertical-align:middle"></i>${s.label}</span><b>${fmt(d.series[s.key][i], 2)}</b></div>`).join("");
    }
    function draw() {
      const d = data[cur], ep = opts.episodes.find(e => e.id === cur);
      cap.innerHTML = ep.caption;
      mt.innerHTML = `<table><tr><th>backbone</th><th class="n">MAE (this ep.)</th><th class="n">flicker (this ep.)</th><th class="n">jump at pot placement</th></tr>` +
        SER.map(s => { const m = d.metrics[s.key], a = data.median_all_val[s.key];
          return `<tr><td><i style="display:inline-block;width:10px;height:3px;background:${s.color};margin-right:6px;vertical-align:middle"></i>${s.label}</td><td class="n">${fmt(m.mae)} <span class="mut" style="color:var(--fg3)">(median ${fmt(a.mae)})</span></td><td class="n">${fmt(m.flicker)} <span style="color:var(--fg3)">(${fmt(a.flicker)})</span></td><td class="n">${fmt(m.jump, 3)} <span style="color:var(--fg3)">(${fmt(a.jump)})</span></td></tr>`; }).join("") +
        `</table><div class="cap">Medians in brackets are over all 120 validation episodes (jump: episodes with a pot placement). Lower MAE = more accurate; lower flicker = steadier; bigger jump = V reacts to progress.</div>`;
      chart.innerHTML = "";
      const W = 600, H = 320, L = 46, R = W - 12, T = 12, B = H - 36;
      const tmax = d.t[d.t.length - 1], yt = [-1, -0.8, -0.6, -0.4, -0.2, 0];
      const x = t => L + (R - L) * t / tmax, y = v => B - (B - T) * (v + 1);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "value over time" });
      yt.forEach(v => { el("line", { x1: L, x2: R, y1: y(v), y2: y(v), class: "gl" }, svg);
        el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", class: "mut" }, svg).textContent = v.toFixed(1); });
      nice(0, tmax, 8).filter(t => t <= tmax).forEach(t => el("text", { x: x(t), y: B + 18, "text-anchor": "middle", class: "mut" }, svg).textContent = t + " s");
      el("line", { x1: L, x2: R, y1: B, y2: B, class: "ax" }, svg);
      el("text", { x: 12, y: (T + B) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(T + B) / 2})` }, svg).textContent = "predicted value V";
      if (d.milestone_s) { el("line", { x1: x(d.milestone_s), x2: x(d.milestone_s), y1: T, y2: B, stroke: "var(--fg3)", "stroke-dasharray": "3 3" }, svg);
        el("text", { x: x(d.milestone_s) + 5, y: T + 12, class: "mut" }, svg).textContent = "1st pot on stove"; }
      const path = arr => arr.map((v, i) => `${i ? "L" : "M"}${x(d.t[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d: path(d.target), fill: "none", stroke: "var(--fg)", "stroke-width": 1.5, "stroke-dasharray": "5 3" }, svg);
      SER.forEach(s => { if (!hidden.has(s.key)) el("path", { d: path(d.series[s.key]), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }, svg); });
      const cross = el("line", { x1: 0, x2: 0, y1: T, y2: B, stroke: "var(--fg2)", "stroke-width": 1 }, svg);
      const hit = el("rect", { x: L, y: T, width: R - L, height: B - T, fill: "transparent", style: "cursor:crosshair" }, svg);
      const move = ev => { const b = svg.getBoundingClientRect(), px = (ev.clientX - b.left) * W / b.width;
        const t = Math.max(0, Math.min(tmax, (px - L) / (R - L) * tmax)); let i = 0, best = 1e9;
        d.t.forEach((tt, j) => { if (Math.abs(tt - t) < best) { best = Math.abs(tt - t); i = j; } });
        cross.setAttribute("x1", x(d.t[i])); cross.setAttribute("x2", x(d.t[i])); setFrame(d, i); };
      hit.addEventListener("mousemove", move); hit.addEventListener("touchmove", e => { move(e.touches[0]); e.preventDefault(); }, { passive: false });
      chart.appendChild(svg);
      const i0 = Math.min(d.t.length - 1, Math.round(d.t.length * 0.6)); cross.setAttribute("x1", x(d.t[i0])); cross.setAttribute("x2", x(d.t[i0])); setFrame(d, i0);
    }
    draw();
    window.addEventListener("resize", () => setFrame(data[cur], idx));
  };

  window.themeToggle = function (btn) {
    const r = document.documentElement;
    btn.onclick = () => { const dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      r.dataset.theme = dark ? "light" : "dark"; try { localStorage.setItem("vl-theme", r.dataset.theme); } catch (e) {} };
    try { const s = localStorage.getItem("vl-theme"); if (s) r.dataset.theme = s; } catch (e) {}
  };
})();
// Open any collapsed <details> that holds the linked anchor, then scroll to it.
(function () {
  function reveal() {
    const id = decodeURIComponent(location.hash.slice(1)); if (!id) return;
    const t = document.getElementById(id); if (!t) return;
    for (let a = t.parentElement; a; a = a.parentElement) if (a.tagName === "DETAILS") a.open = true;
    t.scrollIntoView();
  }
  window.addEventListener("hashchange", reveal); window.addEventListener("load", reveal);
})();
