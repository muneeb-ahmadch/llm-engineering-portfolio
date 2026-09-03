/* Theme toggle, experiment filters, table-of-contents highlighting. No dependencies. */
(function () {
  "use strict";

  function effectiveTheme() {
    var set = document.documentElement.getAttribute("data-theme");
    if (set === "light" || set === "dark") return set;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark" : "light";
  }

  function initTheme() {
    var button = document.querySelector(".theme-toggle");
    if (!button) return;
    var sync = function () {
      var now = effectiveTheme();
      button.setAttribute("aria-pressed", now === "dark" ? "true" : "false");
      button.setAttribute("aria-label", now === "dark"
        ? "Switch to the light theme" : "Switch to the dark theme");
    };
    sync();
    button.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
      sync();
    });
  }

  function initFilters() {
    var chips = Array.prototype.slice.call(document.querySelectorAll(".chip"));
    var rows = Array.prototype.slice.call(document.querySelectorAll(".exp"));
    var status = document.querySelector(".chips-status");
    if (!chips.length || !rows.length) return;
    var tag = "all";
    var predictionOnly = false;

    var apply = function () {
      var shown = 0;
      rows.forEach(function (row) {
        var tags = (row.getAttribute("data-tags") || "").split(" ");
        var ok = (tag === "all" || tags.indexOf(tag) !== -1) &&
                 (!predictionOnly || row.getAttribute("data-prediction") === "true");
        row.hidden = !ok;
        if (ok) shown++;
      });
      chips.forEach(function (chip) {
        var pressed = chip.dataset.filter === "prediction"
          ? predictionOnly : chip.dataset.filter === tag;
        chip.setAttribute("aria-pressed", pressed ? "true" : "false");
      });
      if (status) {
        status.textContent = shown === rows.length
          ? "Showing all " + rows.length + " experiments."
          : "Showing " + shown + " of " + rows.length + " experiments.";
      }
    };

    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        if (chip.dataset.filter === "prediction") predictionOnly = !predictionOnly;
        else tag = chip.dataset.filter;
        apply();
      });
    });
    apply();
  }

  function initToc() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".toc a[href^='#']"));
    if (!links.length || !window.IntersectionObserver) return;
    var byId = {};
    var targets = [];
    links.forEach(function (link) {
      var el = document.getElementById(link.getAttribute("href").slice(1));
      if (el) { byId[el.id] = link; targets.push(el); }
    });
    var mark = function (id) {
      links.forEach(function (link) { link.removeAttribute("aria-current"); });
      if (byId[id]) byId[id].setAttribute("aria-current", "true");
    };
    var observer = new IntersectionObserver(function (entries) {
      var visible = entries.filter(function (e) { return e.isIntersecting; });
      if (visible.length) mark(visible[0].target.id);
    }, { rootMargin: "-20% 0px -70% 0px", threshold: 0 });
    targets.forEach(function (el) { observer.observe(el); });
  }

  initTheme();
  initFilters();
  initToc();
})();
