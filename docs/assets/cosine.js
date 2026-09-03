/* Cosine-similarity demo — pairs with cosine.css.
 *
 * Markup:
 *   <div class="cosine-demo" data-config='{
 *      "chunks": [
 *        {"label": "“reranking” chunk", "angle": 68, "target": true},
 *        {"label": "“chunking” chunk",  "angle": 22}
 *      ],
 *      "query": 30
 *   }'></div>
 *
 * angle: degrees from the x-axis (0–90). target: the chunk we WANT on top.
 * The slider rotates the query vector; readout shows cosine to each chunk (cos of the
 * angle between them, since all vectors are unit length) and a HIT/MISS badge for whether
 * the target chunk is currently top-1. Vanilla JS, no dependencies.
 */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var Ox = 48, Oy = 198, L = 150, W = 384, H = 244;

  function el(tag, attrs) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function rad(d) { return d * Math.PI / 180; }
  function tip(a) { return [Ox + L * Math.cos(rad(a)), Oy - L * Math.sin(rad(a))]; }

  function build(root) {
    var cfg = {};
    try { cfg = JSON.parse(root.getAttribute("data-config") || "{}"); } catch (e) { cfg = {}; }
    var chunks = cfg.chunks || [
      { label: "“reranking” chunk", angle: 68, target: true },
      { label: "“chunking” chunk", angle: 22 }
    ];
    var qStart = cfg.query != null ? cfg.query : 30;

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "cosine-svg" });
    svg.appendChild(el("line", { x1: Ox, y1: Oy, x2: Ox + 168, y2: Oy, class: "cos-axis" }));
    svg.appendChild(el("line", { x1: Ox, y1: Oy, x2: Ox, y2: Oy - 168, class: "cos-axis" }));

    chunks.forEach(function (c) {
      var t = tip(c.angle);
      var tone = c.target ? "is-target" : "is-distractor";
      svg.appendChild(el("line", { x1: Ox, y1: Oy, x2: t[0], y2: t[1], class: "cos-chunk " + tone }));
      svg.appendChild(el("circle", { cx: t[0], cy: t[1], r: 4, class: "cos-chunk " + tone }));
      var lab = el("text", { x: t[0] + 8, y: t[1] - 7, class: "cos-label " + tone });
      lab.textContent = c.label;
      svg.appendChild(lab);
    });

    var qLine = el("line", { x1: Ox, y1: Oy, class: "cos-query" });
    var qDot = el("circle", { r: 4.5, class: "cos-query" });
    var qLab = el("text", { class: "cos-label is-query" });
    qLab.textContent = "query";
    svg.appendChild(qLine); svg.appendChild(qDot); svg.appendChild(qLab);
    root.appendChild(svg);

    var controls = document.createElement("div");
    controls.className = "cos-controls";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = 0; slider.max = 90; slider.step = 1;
    slider.value = qStart; slider.className = "cos-slider";
    slider.setAttribute("aria-label", "query direction in degrees");
    var read = document.createElement("div");
    read.className = "cos-read";
    controls.appendChild(slider); controls.appendChild(read);
    root.appendChild(controls);

    function update() {
      var qa = +slider.value, t = tip(qa);
      qLine.setAttribute("x2", t[0]); qLine.setAttribute("y2", t[1]);
      qDot.setAttribute("cx", t[0]); qDot.setAttribute("cy", t[1]);
      qLab.setAttribute("x", t[0] + 8); qLab.setAttribute("y", t[1] - 6);

      var best = -2, bestIdx = 0;
      chunks.forEach(function (c, i) {
        c._cos = Math.cos(rad(Math.abs(qa - c.angle)));
        if (c._cos > best) { best = c._cos; bestIdx = i; }
      });
      var rows = chunks.map(function (c, i) {
        return '<div class="cos-row' + (i === bestIdx ? " is-top" : "") + '">' +
               "<span>" + c.label + "</span><b>" + c._cos.toFixed(2) + "</b></div>";
      }).join("");
      var targetIdx = -1;
      chunks.forEach(function (c, i) { if (c.target) targetIdx = i; });
      var badge = "";
      if (targetIdx >= 0) {
        var hit = bestIdx === targetIdx;
        badge = '<span class="cos-badge ' + (hit ? "hit" : "miss") + '">' +
                (hit ? "HIT · target chunk is top-1" : "MISS · wrong chunk on top") + "</span>";
      }
      read.innerHTML = rows + badge;
    }
    slider.addEventListener("input", update);
    update();
  }

  function boot() { document.querySelectorAll(".cosine-demo").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
