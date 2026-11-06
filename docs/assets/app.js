/* GridOps Console: reads data/console.json written by the hourly control loop. */
(() => {
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const utc = (s) => new Date(/Z|[+-]\d\d:\d\d$/.test(s) ? s : s + "Z");
  const hm = (s) => utc(s).toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit" });
  const ago = (s) => { const m = Math.round((Date.now() - utc(s)) / 60000); return m < 60 ? `${m} min ago` : m < 2880 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`; };
  // National Grid ESO intensity bands (gCO2/kWh)
  const BANDS = [[40, "Very low", "#1b7a4a"], [120, "Low", "#5c9d3b"], [200, "Moderate", "#c58a0b"], [290, "High", "#d0612b"], [1e9, "Very high", "#c2352b"]];
  const band = (v) => BANDS.find(([u]) => v < u);

  fetch("data/console.json", { cache: "no-store" })
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(render)
    .catch(() => { $("meta").textContent = "The control loop hasn't published yet. It runs every hour."; });

  function render(d) {
    $("meta").textContent = `Control loop last ran ${ago(d.generated_at)}`;
    const v = d.now.actual, [, name, color] = band(v);
    $("now-val").textContent = Math.round(v);
    document.documentElement.style.setProperty("--band", color);
    $("now-band").textContent = `${name} carbon`;
    const fc = d.forecast, lowIdx = fc.model.indexOf(Math.min(...fc.model));
    $("now-note").textContent = `Measured at ${hm(d.now.time)} UTC. Cleanest time in the next 24 h: around ${hm(fc.time[lowIdx])} (≈${Math.round(fc.model[lowIdx])}).`;
    forecastChart(d);
    status(d);
    registry(d);
    runs(d);
    live(d);
    drift(d);
  }

  function forecastChart(d) {
    const h = d.history, f = d.forecast, muted = css("--muted");
    const labels = [...h.time, ...f.time];
    const pad = (a, before, after = 0) => [...Array(before).fill(null), ...a, ...Array(after).fill(null)];
    const last = h.actual[h.actual.length - 1];
    new Chart($("fc"), {
      type: "line",
      data: { labels, datasets: [
        { label: "Actual", data: pad(h.actual, 0, f.time.length), borderColor: css("--obs"), borderWidth: 2, pointRadius: 0, tension: .2 },
        { label: "Champion", data: pad([last, ...f.model], h.time.length - 1), borderColor: css("--model"), borderWidth: 2.5, pointRadius: 0, tension: .2,
          fill: "origin", backgroundColor: css("--model") + "1c" },
        { label: "ESO", data: pad([last, ...f.eso], h.time.length - 1), borderColor: css("--eso"), borderDash: [4, 3], borderWidth: 1.8, pointRadius: 0, tension: .2 },
      ] },
      options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, animation: { duration: 500 },
        plugins: { legend: { display: false }, tooltip: { callbacks: { title: (i) => hm(labels[i[0].dataIndex]), label: (i) => i.raw == null ? null : `${i.dataset.label}: ${Math.round(i.raw)} gCO₂/kWh` } } },
        scales: { x: { grid: { display: false }, ticks: { color: muted, maxTicksLimit: 7, maxRotation: 0, callback: (i) => hm(labels[i]) } },
                  y: { beginAtZero: true, grid: { color: css("--rule") }, ticks: { color: muted } } } },
      plugins: [{ id: "now", afterDraw(c) {
        const x = c.scales.x.getPixelForValue(h.time.length - 1), { top, bottom } = c.chartArea, ctx = c.ctx;
        ctx.save(); ctx.strokeStyle = muted; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
        ctx.fillStyle = muted; ctx.font = "12px Barlow, sans-serif"; ctx.fillText("now", x + 5, top + 12); ctx.restore();
      } }],
    });
  }

  function status(d) {
    const c = d.champion, m = c.metrics, live = d.live || {};
    $("s-champ").textContent = `v${c.version}`;
    $("s-champ-d").textContent = `promoted ${ago(c.exported_at)}`;
    const skill = m.skill_vs_persistence;
    $("s-hold").innerHTML = `${m.mae.toFixed(1)} <span class="${skill > 0 ? "good" : "bad"}" style="font-size:16px">${skill > 0 ? "−" : "+"}${Math.abs(skill * 100).toFixed(0)}% error</span>`;
    $("s-hold-d").textContent = `MAE gCO₂/kWh over 24 h, against ${m.persistence_mae.toFixed(1)} for "no change"`;
    if (live.n_scored >= 48) {
      const better = live.mae_7d < live.eso_mae_7d;
      $("s-live").innerHTML = `${live.mae_7d.toFixed(1)} <span class="${better ? "good" : "warn"}" style="font-size:16px">vs ESO ${live.eso_mae_7d.toFixed(1)}</span>`;
      $("s-live-d").textContent = `beats ESO on ${Math.round(live.win_rate_7d * 100)}% of ${live.n_scored.toLocaleString()} scored half-hours`;
    } else {
      $("s-live").textContent = "warming up";
      $("s-live-d").textContent = `v${c.version}: ${live.n_scored || 0} half-hours scored so far; shown after the first full day`;
    }
    const alerts = d.drift.filter((x) => x.status === "alert"), warns = d.drift.filter((x) => x.status === "warn");
    $("s-drift").innerHTML = alerts.length ? `<span class="bad">${alerts.length} alert${alerts.length > 1 ? "s" : ""}</span>` : warns.length ? `<span class="warn">${warns.length} warning${warns.length > 1 ? "s" : ""}</span>` : `<span class="good">stable</span>`;
    $("s-drift-d").textContent = [...alerts, ...warns].map((x) => x.feature).join(", ") || `${d.drift.length} features within limits`;
    const r = d.runs[d.runs.length - 1];
    $("s-loop").textContent = `${d.runs.length} runs`;
    $("s-loop-d").textContent = r ? `last ${ago(r.at)}, ${r.seconds}s${r.retrained ? ", retrained" : ""}` : "";
  }

  function registry(d) {
    $("versions").innerHTML = d.registry.slice(0, 7).map((v) => `
      <li class="${v.status === "promoted" ? "promoted" : ""} ${v.champion ? "champion" : ""}">
        <div class="row1"><b>v${v.version}</b>${v.champion ? '<span class="badge champ">champion</span>' : v.status === "rejected" ? '<span class="badge rej">rejected</span>' : '<span class="badge">retired</span>'}</div>
        <div class="row2">holdout MAE ${v.mae.toFixed(2)}${v.skill != null ? `, ${(v.skill * 100).toFixed(0)}% better than persistence` : ""}. Registered ${ago(v.created)}</div>
      </li>`).join("");
  }

  function runs(d) {
    const rs = d.runs, max = Math.max(...rs.map((r) => r.seconds), 1);
    $("runs").innerHTML = rs.map((r) => `<span class="${r.retrained ? "retrain" : ""} ${r.promoted ? "promote" : ""}" style="height:${Math.max(6, (r.seconds / max) * 100)}%"
      title="${hm(r.at)}: ${r.seconds}s${r.retrained ? " (retrained" + (r.promoted ? ", promoted)" : ", rejected)") : ""}"></span>`).join("");
    const retrains = rs.filter((r) => r.retrained).slice(-4).reverse();
    $("reasons").innerHTML = retrains.map((r) => `<li><time>${hm(r.at)}</time><span>${r.promoted ? "Promoted" : "Kept champion"}: ${r.reasons.join("; ")}</span></li>`).join("");
  }

  function live(d) {
    const L = d.live, muted = css("--muted");
    if (!L.daily) { $("live").parentElement.innerHTML = `<p class="sub">Live scores appear once the first logged forecasts come due, within a day of launch.</p>`; return; }
    new Chart($("live"), {
      type: "line",
      data: { labels: L.daily.day.map((x) => x.slice(5)), datasets: [
        { label: "Champion", data: L.daily.model, borderColor: css("--model"), backgroundColor: css("--model"), borderWidth: 2.5, pointRadius: 3 },
        { label: "ESO", data: L.daily.eso, borderColor: css("--eso"), backgroundColor: css("--eso"), borderDash: [4, 3], borderWidth: 2, pointRadius: 3 },
      ] },
      options: { maintainAspectRatio: false, plugins: { legend: { labels: { color: muted, boxWidth: 12, boxHeight: 3 } } },
        scales: { x: { ticks: { color: muted }, grid: { display: false } }, y: { beginAtZero: true, ticks: { color: muted }, grid: { color: css("--rule") }, title: { display: true, text: "MAE, gCO₂/kWh", color: muted } } } },
    });
  }

  function drift(d) {
    const names = { wind_mean: "Wind speed (100 m)", solar_mean: "Solar radiation", temp_mean: "Temperature", cloud_mean: "Cloud cover", actual: "Carbon intensity (target)" };
    const col = { ok: css("--ok"), warn: css("--warn"), alert: css("--bad") };
    $("drift").querySelector("tbody").innerHTML = d.drift.map((x) => `<tr><td>${names[x.feature] || x.feature}</td>
      <td><span class="psi" style="width:${Math.min(120, (x.psi || 0) * 300)}px;background:${col[x.status]}"></span>${x.psi?.toFixed(3) ?? "–"}</td>
      <td>${x.ref_mean}</td><td>${x.cur_mean}</td></tr>`).join("");
  }
})();
