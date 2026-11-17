/**
 * MathGuard shared front-end
 * Loads docs/data/site_results.json once; never hardcodes experimental numbers.
 */
(function () {
  "use strict";

  const STEPS = [
    { id: "problem", href: "problem.html", title: "Problem", num: 1 },
    { id: "data", href: "data.html", title: "Data", num: 2 },
    { id: "methods", href: "methods.html", title: "Methods", num: 3 },
    { id: "results", href: "results.html", title: "Results", num: 4 },
    { id: "explore", href: "explore.html", title: "Review lab", num: 5 },
    { id: "errors", href: "errors.html", title: "Errors", num: 6 },
    { id: "limitations", href: "limitations.html", title: "Limits", num: 7 },
  ];

  const MODEL_META = {
    baseline_confidence: { label: "Confidence baseline", color: "var(--baseline)", colorHex: "#64748b" },
    logistic_l2: { label: "Logistic regression", color: "var(--logistic)", colorHex: "#2563eb" },
    random_forest: { label: "Random forest", color: "var(--forest)", colorHex: "#7c3aed" },
    svm_rbf: { label: "SVM (RBF)", color: "var(--svm)", colorHex: "#0d9488" },
  };

  const MODEL_ORDER = ["baseline_confidence", "logistic_l2", "svm_rbf", "random_forest"];

  let cachedData = null;
  let loadPromise = null;

  function currentPage() {
    const path = (location.pathname.split("/").pop() || "index.html").toLowerCase();
    return path || "index.html";
  }

  function pageId() {
    const p = currentPage().replace(".html", "");
    return p === "index" ? "home" : p;
  }

  function loadResults() {
    if (cachedData) return Promise.resolve(cachedData);
    if (loadPromise) return loadPromise;
    loadPromise = fetch("data/site_results.json")
      .then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then((d) => {
        cachedData = d;
        return d;
      })
      .catch((err) => {
        loadPromise = null;
        throw err;
      });
    return loadPromise;
  }

  function fmt(x, digits) {
    if (x == null || Number.isNaN(x)) return "—";
    return Number(x).toFixed(digits != null ? digits : 3);
  }

  function pct(x, digits) {
    if (x == null || Number.isNaN(x)) return "—";
    return (Number(x) * 100).toFixed(digits != null ? digits : 1) + "%";
  }

  function renderNav(activeId) {
    const el = document.getElementById("site-nav");
    if (!el) return;
    const links = [
      { href: "index.html", title: "Home", id: "home" },
      ...STEPS.map((s) => ({ href: s.href, title: s.title, id: s.id })),
      { href: "reproduce.html", title: "Reproduce", id: "reproduce" },
      {
        href: "https://github.com/Janil-ship-it/es335-math-recognition-error-prediction",
        title: "GitHub",
        id: "repo",
        external: true,
      },
    ];
    el.innerHTML = `
      <div class="wrap nav-inner">
        <button type="button" class="nav-toggle" aria-expanded="false" aria-controls="nav-links" id="nav-toggle">Menu</button>
        <ul class="nav-links" id="nav-links">
          ${links
            .map((l) => {
              const cls = [l.id === activeId ? "active" : "", l.id === "repo" ? "repo" : ""].filter(Boolean).join(" ");
              const extra = l.external ? ' target="_blank" rel="noopener"' : "";
              return `<li><a href="${l.href}" class="${cls}"${extra}>${l.title}</a></li>`;
            })
            .join("")}
        </ul>
      </div>`;
    const toggle = document.getElementById("nav-toggle");
    const list = document.getElementById("nav-links");
    if (toggle && list) {
      toggle.addEventListener("click", () => {
        const open = list.classList.toggle("open");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }
  }

  function renderStepChrome(stepId) {
    const step = STEPS.find((s) => s.id === stepId);
    if (!step) return;
    const idx = STEPS.indexOf(step);
    const progress = document.getElementById("step-progress");
    if (progress) {
      const pctDone = ((idx + 1) / STEPS.length) * 100;
      progress.innerHTML = `
        <p class="step-meta">Step ${step.num} of ${STEPS.length}</p>
        <div class="progress-bar" role="progressbar" aria-valuenow="${step.num}" aria-valuemin="1" aria-valuemax="${STEPS.length}" aria-label="Walkthrough progress">
          <span style="width:${pctDone}%"></span>
        </div>`;
    }
    const crumb = document.getElementById("breadcrumb");
    if (crumb) {
      crumb.innerHTML = `<a href="index.html">Home</a><span>/</span><span>${step.title}</span>`;
    }
    const nav = document.getElementById("page-nav");
    if (nav) {
      const prev = STEPS[idx - 1];
      const next = STEPS[idx + 1];
      let html = "";
      if (prev) {
        html += `<a href="${prev.href}" class="prev"><span class="dir">Previous</span><span class="title">${prev.title}</span></a>`;
      } else {
        html += `<a href="index.html" class="prev"><span class="dir">Previous</span><span class="title">Home</span></a>`;
      }
      if (next) {
        html += `<a href="${next.href}" class="next"><span class="dir">Next</span><span class="title">${next.title}</span></a>`;
      } else {
        html += `<a href="reproduce.html" class="next"><span class="dir">Appendix</span><span class="title">Reproduce</span></a>`;
      }
      nav.innerHTML = html;
    }
  }

  function fillHeadlineStats(data, containerId) {
    const el = document.getElementById(containerId);
    if (!el || !data) return;
    const m = data.metrics || {};
    const lr = m.logistic_l2 || {};
    const shuf = m.shuffled_label_control || {};
    el.innerHTML = `
      <div class="stat-card">
        <div class="value">${fmt(lr.pr_auc)}</div>
        <div class="label">Logistic PR-AUC<br><span class="ci">95% CI [${fmt(lr.pr_auc_ci && lr.pr_auc_ci[0])}, ${fmt(lr.pr_auc_ci && lr.pr_auc_ci[1])}]</span></div>
      </div>
      <div class="stat-card">
        <div class="value">${lr.delta_vs_baseline != null ? (lr.delta_vs_baseline >= 0 ? "+" : "") + fmt(lr.delta_vs_baseline) : "—"}</div>
        <div class="label">Δ vs baseline<br><span class="ci">CI includes zero</span></div>
      </div>
      <div class="stat-card">
        <div class="value">${pct(lr["recall@30"])}</div>
        <div class="label">Recall @ 30% review</div>
      </div>
      <div class="stat-card">
        <div class="value">${fmt(shuf.pr_auc)}</div>
        <div class="label">Shuffled-label control<br><span class="ci">near prevalence</span></div>
      </div>`;
  }

  function fillResultsTable(data, tbodyId) {
    const tbody = document.getElementById(tbodyId);
    if (!tbody || !data) return;
    const m = data.metrics || {};
    let bestPrauc = -1;
    let bestKey = null;
    MODEL_ORDER.forEach((k) => {
      if (m[k] && m[k].pr_auc > bestPrauc) {
        bestPrauc = m[k].pr_auc;
        bestKey = k;
      }
    });
    tbody.innerHTML = MODEL_ORDER.map((k) => {
      const r = m[k];
      if (!r || r.pr_auc == null) return "";
      const meta = MODEL_META[k] || { label: k };
      const delta =
        r.delta_vs_baseline != null
          ? `${r.delta_vs_baseline >= 0 ? "+" : ""}${fmt(r.delta_vs_baseline)} <span class="ci">[${fmt(r.delta_ci && r.delta_ci[0])}, ${fmt(r.delta_ci && r.delta_ci[1])}]</span>`
          : "—";
      const best = k === bestKey ? ' class="best"' : "";
      return `<tr${best}>
        <td><strong>${meta.label}</strong></td>
        <td class="num">${fmt(r.pr_auc)} <span class="ci">[${fmt(r.pr_auc_ci && r.pr_auc_ci[0])}, ${fmt(r.pr_auc_ci && r.pr_auc_ci[1])}]</span></td>
        <td class="num">${delta}</td>
        <td class="num">${pct(r["recall@20"])}</td>
        <td class="num">${pct(r["recall@30"])}</td>
        <td class="num">${pct(r["recall@50"])}</td>
        <td class="num">${fmt(r.brier)}</td>
      </tr>`;
    }).join("");
  }

  function fillFeatureImportance(data, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined" || !data) return;
    const fi = (data.metrics || {}).feature_importance_logistic || {};
    const names = Object.keys(fi);
    if (!names.length) return;
    const vals = names.map((n) => fi[n]);
    const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: names,
        datasets: [{ label: "Coefficient", data: vals, backgroundColor: vals.map((v) => (v >= 0 ? "#dc2626aa" : "#2563ebaa")) }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: "Standardised coefficient" },
            ticks: { color: isDark ? "#9aa3b2" : "#5a6270" },
            grid: { color: isDark ? "#2a2f3a" : "#e2e5eb" },
          },
          y: { ticks: { color: isDark ? "#9aa3b2" : "#5a6270" }, grid: { display: false } },
        },
      },
    });
  }

  function fillPrCurves(data, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined" || !data) return;
    const curves = data.pr_curves || {};
    const datasets = [];
    MODEL_ORDER.forEach((k) => {
      const c = curves[k];
      if (!c || !c.recall || !c.recall.length) return;
      const meta = MODEL_META[k];
      datasets.push({
        label: meta.label,
        data: c.recall.map((r, i) => ({ x: r, y: c.precision[i] })),
        borderColor: meta.colorHex,
        backgroundColor: "transparent",
        pointRadius: 0,
        tension: 0.15,
        borderWidth: 2,
      });
    });
    if (!datasets.length) {
      const wrap = canvas.closest(".chart-box");
      if (wrap) {
        wrap.innerHTML =
          '<p class="note">PR curve points are not in <code>site_results.json</code> for this build. Table metrics above are complete.</p>';
      }
      return;
    }
    new Chart(canvas, {
      type: "line",
      data: { datasets },
      options: {
        responsive: true,
        scales: {
          x: { type: "linear", title: { display: true, text: "Recall" }, min: 0, max: 1 },
          y: { title: { display: true, text: "Precision" }, min: 0, max: 1 },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  function initFadeIn() {
    const nodes = document.querySelectorAll(".fade-in");
    if (!nodes.length) return;
    if (!("IntersectionObserver" in window)) {
      nodes.forEach((n) => n.classList.add("visible"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("visible");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    nodes.forEach((n) => io.observe(n));
  }

  function init(options) {
    options = options || {};
    const id = options.page || pageId();
    renderNav(id);
    if (id !== "home" && id !== "reproduce") {
      renderStepChrome(id);
    }
    initFadeIn();

    const needsData = options.stats || options.table || options.charts || options.onData;
    if (!needsData) return;

    document.querySelectorAll("[data-loading]").forEach((el) => {
      el.innerHTML = '<div class="loading-block">Loading results…</div>';
    });

    loadResults()
      .then((data) => {
        if (options.stats) fillHeadlineStats(data, options.stats);
        if (options.table) fillResultsTable(data, options.table);
        if (options.featureChart) fillFeatureImportance(data, options.featureChart);
        if (options.prChart) fillPrCurves(data, options.prChart);
        if (typeof options.onData === "function") options.onData(data);
      })
      .catch((err) => {
        document.querySelectorAll("[data-loading]").forEach((el) => {
          el.innerHTML = `<div class="error-block">Could not load <code>data/site_results.json</code>: ${err.message}. Run <code>python -m src.05_evaluate_test</code> and open via a local server.</div>`;
        });
      });
  }

  window.MathGuard = {
    init,
    loadResults,
    MODEL_META,
    MODEL_ORDER,
    STEPS,
    fmt,
    pct,
  };
})();
