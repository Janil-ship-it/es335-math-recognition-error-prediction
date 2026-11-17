(function () {
  const METRICS = {
    baseline: {
      label: "Confidence threshold",
      blurb: "Ranks lines by mean or min token confidence (chosen on train folds). The natural operational baseline: flag what the recogniser was unsure about.",
      prauc: 0.715, r20: 0.25, r30: 0.3587, r50: 0.5870, brier: 0.237,
      ser20: 0.1564, ser30: 0.1324, ser50: 0.0910
    },
    logistic: {
      label: "L2 logistic regression",
      blurb: "Uses all allowed features (image quality, confidences, structure), class-balanced, with L2 penalty. Selected by grouped CV + one-SE rule on development students.",
      prauc: 0.816, r20: 0.2935, r30: 0.4239, r50: 0.6413, brier: 0.200,
      ser20: 0.1519, ser30: 0.1196, ser50: 0.0795
    },
    forest: {
      label: "Random forest",
      blurb: "Bagged shallow trees on feature subsets. Non-linear alternative with capacity controlled by depth. Strong ranking on the test set.",
      prauc: 0.830, r20: 0.25, r30: 0.4130, r50: 0.6739, brier: 0.195,
      ser20: 0.1591, ser30: 0.1234, ser50: 0.0792
    }
  };

  function interp(m, rate) {
    if (rate <= 20) return m.r20 * (rate / 20);
    if (rate <= 30) return m.r20 + (m.r30 - m.r20) * ((rate - 20) / 10);
    if (rate <= 50) return m.r30 + (m.r50 - m.r30) * ((rate - 30) / 20);
    return m.r50 + (1 - m.r50) * ((rate - 50) / 50) * 0.25;
  }
  function interpSer(m, rate) {
    if (rate <= 20) return m.ser20;
    if (rate <= 30) return m.ser20 + (m.ser30 - m.ser20) * ((rate - 20) / 10);
    if (rate <= 50) return m.ser30 + (m.ser50 - m.ser30) * ((rate - 30) / 20);
    return m.ser50 * (1 - (rate - 50) / 100);
  }

  const desc = document.getElementById("model-desc");
  const msPrauc = document.getElementById("ms-prauc");
  const msR30 = document.getElementById("ms-r30");
  const msBrier = document.getElementById("ms-brier");
  document.querySelectorAll("#model-toggles .tog").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#model-toggles .tog").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const m = METRICS[btn.dataset.model];
      desc.innerHTML = "<h3>" + m.label + "</h3><p>" + m.blurb + "</p>";
      msPrauc.textContent = m.prauc.toFixed(3);
      msR30.textContent = (m.r30 * 100).toFixed(1) + "%";
      msBrier.textContent = m.brier.toFixed(3);
    });
  });

  const slider = document.getElementById("rate-slider");
  const rateLabel = document.getElementById("rate-label");
  const chips = document.querySelectorAll(".chip");

  function setRate(r) {
    slider.value = r;
    rateLabel.textContent = r + "%";
    chips.forEach((c) => c.classList.toggle("active", +c.dataset.rate === +r));
    renderBudget(+r);
  }
  slider.addEventListener("input", () => setRate(slider.value));
  chips.forEach((c) => c.addEventListener("click", () => setRate(c.dataset.rate)));

  function renderBudget(rate) {
    const b = interp(METRICS.baseline, rate);
    const l = interp(METRICS.logistic, rate);
    const f = interp(METRICS.forest, rate);
    document.getElementById("live-base").textContent = (b * 100).toFixed(1) + "%";
    document.getElementById("live-lr").textContent = (l * 100).toFixed(1) + "%";
    document.getElementById("live-rf").textContent = (f * 100).toFixed(1) + "%";
    document.getElementById("live-miss").textContent = ((1 - l) * 100).toFixed(0) + "%";
    document.getElementById("live-gain").textContent =
      (l - b >= 0 ? "+" : "") + ((l - b) * 100).toFixed(1) + " pp";
    document.getElementById("live-ser").textContent = interpSer(METRICS.logistic, rate).toFixed(3);
    document.getElementById("budget-note").textContent =
      "At " + rate + "% review, logistic catches about " + ((l - b) * 100).toFixed(1) +
      " percentage points more errors than the confidence baseline on the held-out test groups.";
    drawRecallChart(rate, b, l, f);
  }

  function drawRecallChart(rate, b, l, f) {
    const canvas = document.getElementById("recall-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || 520;
    const H = 280;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const pad = { l: 48, r: 16, t: 24, b: 40 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + plotH);
    ctx.lineTo(pad.l + plotW, pad.t + plotH);
    ctx.stroke();
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px system-ui";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + plotH - (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.l - 4, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.strokeStyle = "#e2e8f0";
      ctx.stroke();
      ctx.fillStyle = "#94a3b8";
      ctx.fillText(i * 25 + "%", pad.l - 8, y + 4);
    }
    ctx.textAlign = "center";
    ctx.fillText("share of true errors caught", pad.l + plotW / 2, H - 10);
    const vals = [
      { v: b, color: "#94a3b8", label: "Baseline" },
      { v: l, color: "#7c3aed", label: "Logistic" },
      { v: f, color: "#0d9488", label: "Forest" }
    ];
    const barW = plotW / 5;
    vals.forEach((d, i) => {
      const x = pad.l + (i + 0.7) * (plotW / 3);
      const h = d.v * plotH;
      const y = pad.t + plotH - h;
      ctx.fillStyle = d.color;
      roundRect(ctx, x - barW / 2, y, barW, h, 6);
      ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.font = "600 12px system-ui";
      ctx.textAlign = "center";
      ctx.fillText((d.v * 100).toFixed(1) + "%", x, y - 6);
      ctx.fillStyle = "#64748b";
      ctx.font = "11px system-ui";
      ctx.fillText(d.label, x, pad.t + plotH + 16);
    });
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawCompareChart() {
    const canvas = document.getElementById("compare-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || 720;
    const H = 220;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const pad = { l: 48, r: 16, t: 20, b: 36 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    const rates = [20, 30, 50];
    const series = [
      { key: "baseline", color: "#94a3b8", label: "Baseline" },
      { key: "logistic", color: "#7c3aed", label: "Logistic" },
      { key: "forest", color: "#0d9488", label: "Forest" }
    ];
    ctx.strokeStyle = "#e2e8f0";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px system-ui";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + plotH - (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
      ctx.fillText(i * 25 + "%", pad.l - 8, y + 4);
    }
    const groupW = plotW / rates.length;
    const barW = groupW / 5;
    rates.forEach((rate, gi) => {
      const cx = pad.l + (gi + 0.5) * groupW;
      series.forEach((s, si) => {
        const m = METRICS[s.key];
        const v = rate === 20 ? m.r20 : rate === 30 ? m.r30 : m.r50;
        const h = v * plotH;
        const x = cx + (si - 1) * (barW + 4) - barW / 2;
        const y = pad.t + plotH - h;
        ctx.fillStyle = s.color;
        roundRect(ctx, x, y, barW, h, 4);
        ctx.fill();
      });
      ctx.fillStyle = "#64748b";
      ctx.font = "11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(rate + "% review", cx, pad.t + plotH + 18);
    });
    series.forEach((s, i) => {
      const x = pad.l + i * 90;
      ctx.fillStyle = s.color;
      ctx.fillRect(x, 6, 12, 12);
      ctx.fillStyle = "#334155";
      ctx.font = "11px system-ui";
      ctx.textAlign = "left";
      ctx.fillText(s.label, x + 16, 16);
    });
  }

  setRate(30);
  drawCompareChart();
  window.addEventListener("resize", () => {
    setRate(slider.value);
    drawCompareChart();
  });
})();
