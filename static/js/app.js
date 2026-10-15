/* ==========================================================================
   Signal Monitor — dashboard shell

   Data source: the Flask backend, always in real historical inference mode.
     * Model failure probability: GET /api/historical/scores
     * Scaled sensor traces (generic labels): GET /api/historical/series
     * Model identity, aggregate metrics and limitations: GET /api/model/insights
     * Startup status (ready + models): GET /api/modes

   No system data is loaded and no model is executed in the browser, and no
   invented/demo data is ever substituted. If the required local artifacts are
   missing or the backend is unreachable, the dashboard shows an explicit
   unavailable state instead.
   ========================================================================== */
(function () {
  "use strict";

  var DEFAULT_RANGE = 30;
  var DAY_MS = 86400000;

  /* ---------------------------------------------------------------------- *
   * State
   * ---------------------------------------------------------------------- */
  var STATE = {
    mode: "loading",            // "loading" | "historical" | "unavailable"
    ready: false,
    error: null,                // safe, user-facing reason when unavailable
    source: null,
    models: [],
    modelId: null,
    probability: null,
    dates: [],
    sensors: [],
    series: {},
    episodes: [],
    events: [],
    summary: [],
    coverage: { start: null, days: 0 },
    range: DEFAULT_RANGE,
    hidden: {},
    insights: null,
  };

  /* ---------------------------------------------------------------------- *
   * Helpers
   * ---------------------------------------------------------------------- */
  function el(channel identifier, cls, text) {
    var n = document.createElement(channel identifier);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function byId(id) { return document.getElementById(id); }
  function fmtDate(d) { return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short" }); }
  function fmtFull(d) { return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }); }
  function parseISO(s) { return new Date(String(s) + "T00:00:00"); }
  function isNum(v) { return typeof v === "number" && isFinite(v); }

  function showNotice(id, text, isError) {
    var n = byId(id);
    if (!n) return;
    n.textContent = text;
    n.hidden = false;
    n.classList.toggle("is-error", !!isError);
    n.classList.toggle("is-ok", !isError);
  }
  function hideNotice(id) {
    var n = byId(id);
    if (n) n.hidden = true;
  }
  function emptyChart(container, text) {
    if (!container) return;
    container.innerHTML = "";
    container.appendChild(el("div", "chart-empty", text));
  }

  function getJSON(url) {
    if (typeof fetch !== "function") {
      return Promise.resolve({ ok: false, status: 0, data: null, message: "This browser cannot reach the local backend." });
    }
    return fetch(url, { headers: { Accept: "application/json" } })
      .then(function (r) {
        return r.json().catch(function () { return null; }).then(function (body) {
          var msg = body && body.message ? String(body.message) : null;
          return { ok: r.ok, status: r.status, data: body, message: msg };
        });
      })
      .catch(function () {
        return { ok: false, status: 0, data: null, message: "The local backend could not be reached." };
      });
  }

  function sourceLabel() { return STATE.source ? STATE.source.label : "Unknown"; }

  function pillFor(status) {
    var s = String(status == null ? "" : status).toLowerCase();
    if (s.indexOf("nominal") >= 0 || s === "ok" || s.indexOf("cleared") >= 0 || s.indexOf("resolved") >= 0) return "ok";
    if (s.indexOf("watch") >= 0 || s.indexOf("elevated") >= 0 || s.indexOf("warn") >= 0 ||
        s.indexOf("alarm") >= 0 || s.indexOf("high") >= 0 || s.indexOf("open") >= 0) return "warn";
    return "neutral";
  }
  function humanize(s) {
    return String(s == null ? "" : s).replace(/[-_]/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }
  function sevColor(sev) {
    return sev === "high" ? "#f0616d" : sev === "med" ? "#f0b429" : sev === "low" ? "#5aa9ff" : "#2dd4bf";
  }
  function sevClass(sev) {
    return sev === "high" ? "high" : sev === "med" ? "med" : sev === "low" ? "low" : "info";
  }
  function severityFromPeak(peak) {
    if (peak == null) return "low";
    if (peak >= 0.8) return "high";
    if (peak >= 0.65) return "med";
    return "low";
  }

  /* ---------------------------------------------------------------------- *
   * Data adapter — real historical scoring
   * ---------------------------------------------------------------------- */
  function applyHistorical(scores, series) {
    STATE.mode = "historical";
    STATE.ready = true;
    STATE.error = null;
    STATE.source = scores.source || { id: "historical", label: "Historical data", detail: "Offline local records", live: false };
    STATE.probability = {
      points: scores.points || [],
      model: scores.model || {},
      rows: scores.rows,
      scored: scores.scored,
      unscored: scores.unscored,
      alarms: scores.alarms,
      coverage: scores.coverage || {},
      last_processed: scores.last_processed,
      last_scored: scores.last_scored,
    };
    STATE.dates = (series && series.dates ? series.dates : []).map(parseISO);
    STATE.sensors = (series && series.sensors) || [];
    STATE.series = (series && series.values) || {};
    STATE.coverage = (series && series.coverage) || scores.coverage || { start: null, days: 0 };
    if (![7, 30, 90, 365].includes(STATE.range)) STATE.range = DEFAULT_RANGE;

    STATE.episodes = (scores.episodes || []).map(function (e) {
      return {
        id: e.id, start: e.start, end: e.end, days: e.days, peak: e.peak,
        model: e.model || "Supervised model", status: "Model alarm",
        severity: severityFromPeak(e.peak), kind: "model_alarm",
      };
    });

    STATE.events = STATE.episodes.slice().reverse().map(function (ep) {
      return {
        date: ep.end, type: "Alarm", signal: "Supervised probability",
        severity: ep.severity,
        detail: ep.id + " · peak score " + Number(ep.peak || 0).toFixed(2) + " (30d horizon).",
      };
    });

    var cov = STATE.coverage;
    STATE.summary = [
      { label: "Data mode", value: "Historical", unit: "", status: "demo", foot: "Local artifacts" },
      { label: "Scored days", value: String(scores.scored), unit: "of " + scores.rows, status: "demo", foot: "Supervised model" },
      { label: "Model alarms", value: String(scores.alarms), unit: "days", status: scores.alarms ? "watch" : "ok", foot: "Indication only" },
      { label: "Coverage", value: String(cov.days != null ? cov.days : ""), unit: "days", status: "demo", foot: "Offline history" },
    ];
  }

  /* ---------------------------------------------------------------------- *
   * Summary cards
   * ---------------------------------------------------------------------- */
  var ACCENTS = ["#2dd4bf", "#5aa9ff", "#f0b429", "#a78bfa"];

  function renderSummary() {
    var host = byId("summaryCards");
    if (!host) return;
    host.innerHTML = "";
    if (STATE.error) {
      var card = el("div", "card");
      card.style.setProperty("--accent", "#f0616d");
      card.appendChild(el("div", "card-label", "Data unavailable"));
      card.appendChild(el("div", "card-value", "—"));
      var foot = el("div", "card-foot");
      foot.appendChild(el("span", "pill pill-warn", "Unavailable"));
      foot.appendChild(document.createTextNode(" " + STATE.error));
      card.appendChild(foot);
      host.appendChild(card);
      return;
    }
    if (!STATE.summary.length) {
      var c = el("div", "card");
      c.appendChild(el("div", "card-label", "Loading"));
      c.appendChild(el("div", "card-value", "…"));
      host.appendChild(c);
      return;
    }
    STATE.summary.forEach(function (c, i) {
      var card = el("div", "card");
      card.style.setProperty("--accent", ACCENTS[i % ACCENTS.length]);
      card.appendChild(el("div", "card-label", c.label));
      var v = el("div", "card-value");
      v.appendChild(document.createTextNode(String(c.value)));
      if (c.unit) v.appendChild(el("small", null, c.unit));
      card.appendChild(v);
      var footEl = el("div", "card-foot");
      footEl.appendChild(el("span", "pill pill-" + pillFor(c.status), humanize(c.status)));
      footEl.appendChild(document.createTextNode(" " + (c.foot || "")));
      card.appendChild(footEl);
      host.appendChild(card);
    });
  }

  /* ---------------------------------------------------------------------- *
   * Charts (hand-written SVG, no libraries)
   * ---------------------------------------------------------------------- */
  function svgBase(container, label) {
    var W = 920;
    var H = container.classList.contains("tall") ? 380 : 300;
    var padL = 52, padR = 18, padT = 18, padB = 32;
    var NS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label);
    return { NS: NS, svg: svg, W: W, H: H, padL: padL, padR: padR, padT: padT, padB: padB,
             plotW: W - padL - padR, plotH: H - padT - padB };
  }
  function xOf(b, i, count) { return b.padL + (count <= 1 ? b.plotW / 2 : (i / (count - 1)) * b.plotW); }
  function yOf(b, v) { return b.padT + (1 - Math.max(0, Math.min(1, v))) * b.plotH; }

  function drawGrid(base, count, dateList) {
    var g, i;
    for (g = 0; g <= 4; g++) {
      var frac = g / 4;
      var y = base.padT + frac * base.plotH;
      var line = document.createElementNS(base.NS, "line");
      line.setAttribute("x1", base.padL); line.setAttribute("x2", base.W - base.padR);
      line.setAttribute("y1", y); line.setAttribute("y2", y);
      line.setAttribute("stroke", "rgba(255,255,255,0.06)");
      base.svg.appendChild(line);
      var lbl = document.createElementNS(base.NS, "text");
      lbl.setAttribute("x", base.padL - 8); lbl.setAttribute("y", y + 4);
      lbl.setAttribute("text-anchor", "end"); lbl.setAttribute("fill", "#64717d");
      lbl.setAttribute("font-size", "11"); lbl.textContent = (1 - frac).toFixed(2);
      base.svg.appendChild(lbl);
    }
    var ticks = Math.min(5, count);
    for (i = 0; i < ticks; i++) {
      var idx = ticks === 1 ? 0 : Math.round((i / (ticks - 1)) * (count - 1));
      var t = document.createElementNS(base.NS, "text");
      t.setAttribute("x", xOf(base, idx, count)); t.setAttribute("y", base.H - 10);
      t.setAttribute("text-anchor", "middle"); t.setAttribute("fill", "#64717d");
      t.setAttribute("font-size", "11"); t.textContent = fmtDate(dateList[idx]);
      base.svg.appendChild(t);
    }
  }

  function attachHover(base, container, count, dateList, fillTip) {
    var svg = base.svg;
    var vline = document.createElementNS(base.NS, "line");
    vline.setAttribute("y1", base.padT); vline.setAttribute("y2", base.H - base.padB);
    vline.setAttribute("stroke", "rgba(255,255,255,0.25)");
    vline.setAttribute("stroke-dasharray", "3 3");
    vline.setAttribute("opacity", "0");
    svg.appendChild(vline);

    var tip = el("div", "chart-tip");
    container.appendChild(tip);

    svg.addEventListener("mousemove", function (ev) {
      var rect = svg.getBoundingClientRect();
      if (!(rect.width > 0) || !(base.plotW > 0) || count < 1) return;
      var rel = (ev.clientX - rect.left) / rect.width;
      if (!isFinite(rel) || rel < 0 || rel > 1) return;
      var i = Math.max(0, Math.min(count - 1, Math.round(rel * (count - 1))));
      vline.setAttribute("x1", xOf(base, i, count));
      vline.setAttribute("x2", xOf(base, i, count));
      vline.setAttribute("opacity", "1");
      tip.innerHTML = "";
      tip.appendChild(el("div", "tip-row tip-date", fmtFull(dateList[i])));
      fillTip(i, tip);
      tip.style.opacity = "1";
      tip.style.left = Math.min(0.72, Math.max(0.02, xOf(base, i, count) / base.W)) * 100 + "%";
    });
    svg.addEventListener("mouseleave", function () {
      vline.setAttribute("opacity", "0");
      tip.style.opacity = "0";
    });
  }

  function renderProbabilityChart(container) {
    container.innerHTML = "";
    if (!STATE.probability || !STATE.probability.points.length) return false;
    var all = STATE.probability.points;
    var count = Math.min(STATE.range, all.length);
    var slice = all.slice(all.length - count);
    var dateList = slice.map(function (p) { return parseISO(p.timestamp); });

    var base = svgBase(container, "Failure probability chart");
    drawGrid(base, count, dateList);

    var thr = (STATE.probability.model && STATE.probability.model.threshold != null)
      ? Number(STATE.probability.model.threshold) : 0.5;
    var thY = yOf(base, thr);
    var th = document.createElementNS(base.NS, "line");
    th.setAttribute("x1", base.padL); th.setAttribute("x2", base.W - base.padR);
    th.setAttribute("y1", thY); th.setAttribute("y2", thY);
    th.setAttribute("stroke", "#f0616d"); th.setAttribute("stroke-dasharray", "4 4");
    th.setAttribute("opacity", "0.55");
    base.svg.appendChild(th);

    var i, d = "", started = false;
    var runStart = -1;
    for (i = 0; i <= count; i++) {
      var alarm = i < count && slice[i].alarm;
      if (alarm && runStart < 0) runStart = i;
      if (!alarm && runStart >= 0) {
        var rect = document.createElementNS(base.NS, "rect");
        rect.setAttribute("x", xOf(base, runStart, count));
        rect.setAttribute("width", Math.max(2, xOf(base, Math.min(i, count - 1), count) - xOf(base, runStart, count)));
        rect.setAttribute("y", base.padT);
        rect.setAttribute("height", base.plotH);
        rect.setAttribute("fill", "#f0616d");
        rect.setAttribute("opacity", "0.12");
        base.svg.appendChild(rect);
        runStart = -1;
      }
    }
    for (i = 0; i < count; i++) {
      var v = slice[i].probability;
      if (v == null) { started = false; continue; }
      d += (started ? " L" : " M") + xOf(base, i, count).toFixed(1) + " " + yOf(base, v).toFixed(1);
      started = true;
      if (slice[i].alarm) {
        var c = document.createElementNS(base.NS, "circle");
        c.setAttribute("cx", xOf(base, i, count)); c.setAttribute("cy", yOf(base, v));
        c.setAttribute("r", "2.6"); c.setAttribute("fill", "#f0616d");
        base.svg.appendChild(c);
      }
    }
    if (d) {
      var path = document.createElementNS(base.NS, "path");
      path.setAttribute("d", d.trim());
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#5aa9ff");
      path.setAttribute("stroke-width", "2");
      path.setAttribute("stroke-linejoin", "round");
      base.svg.appendChild(path);
    }
    container.appendChild(base.svg);
    attachHover(base, container, count, dateList, function (i2, tip) {
      var p = slice[i2];
      var row = el("div", "tip-row");
      row.appendChild(el("span", null, "Probability"));
      row.appendChild(el("span", null, p.probability == null ? "–" : Number(p.probability).toFixed(3)));
      tip.appendChild(row);
      var row2 = el("div", "tip-row");
      row2.appendChild(el("span", null, "Status"));
      row2.appendChild(el("span", null, p.alarm ? "ALARM" : "normal"));
      tip.appendChild(row2);
    });
    return true;
  }

  function renderSeriesChart(container, legendEl) {
    if (!container) return false;
    var sensors = STATE.sensors;
    if (!sensors.length) {
      container.innerHTML = "";
      if (legendEl) legendEl.innerHTML = "";
      return false;
    }
    var nDays = STATE.dates.length;
    var count = Math.min(STATE.range, nDays);
    var start = nDays - count;
    var visible = sensors.filter(function (s) { return !STATE.hidden[s.key]; });

    if (legendEl) {
      legendEl.innerHTML = "";
      sensors.forEach(function (s) {
        var item = el("span", "legend-item" + (STATE.hidden[s.key] ? " is-off" : ""));
        var sw = el("span", "legend-swatch");
        sw.style.background = s.color;
        item.appendChild(sw);
        item.appendChild(document.createTextNode(s.label));
        item.addEventListener("click", function () {
          if (STATE.hidden[s.key]) delete STATE.hidden[s.key]; else STATE.hidden[s.key] = true;
          renderSeriesChart(container, legendEl);
          renderTraceStats();
        });
        legendEl.appendChild(item);
      });
    }

    container.innerHTML = "";
    if (!visible.length) {
      container.appendChild(el("div", "chart-empty", "No traces selected."));
      return true;
    }
    var dateList = STATE.dates.slice(start);
    var base = svgBase(container, "Sensor trend chart");
    var normalized = visible.map(function (s) {
      var slice = STATE.series[s.key].slice(start);
      var mn = Infinity, mx = -Infinity;
      slice.forEach(function (v) { if (isNum(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } });
      if (mn === Infinity) { mn = 0; mx = 1; }
      var span = (mx - mn) || 1;
      return { sensor: s, slice: slice, pts: slice.map(function (v) { return isNum(v) ? (v - mn) / span : null; }) };
    });
    drawGrid(base, count, dateList);
    normalized.forEach(function (ser) {
      var d = "", started = false;
      for (var i = 0; i < count; i++) {
        var v = ser.pts[i];
        if (v == null) { started = false; continue; }
        d += (started ? " L" : " M") + xOf(base, i, count).toFixed(1) + " " + yOf(base, v).toFixed(1);
        started = true;
      }
      if (!d) return;
      var path = document.createElementNS(base.NS, "path");
      path.setAttribute("d", d.trim());
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", ser.sensor.color);
      path.setAttribute("stroke-width", "2");
      path.setAttribute("stroke-linejoin", "round");
      base.svg.appendChild(path);
    });
    container.appendChild(base.svg);
    attachHover(base, container, count, dateList, function (i, tip) {
      normalized.forEach(function (ser) {
        var row = el("div", "tip-row");
        row.appendChild(el("span", null, ser.sensor.label));
        var v = ser.slice[i];
        row.appendChild(el("span", null, isNum(v)
          ? Number(v).toFixed(Math.abs(v) < 10 ? 2 : 1) + " " + ser.sensor.unit
          : "–"));
        tip.appendChild(row);
      });
    });
    return true;
  }

  function renderOverviewChart() {
    var container = byId("probChart");
    var legend = byId("trendLegend");
    if (!container) return;
    if (STATE.probability && STATE.probability.points.length) {
      renderProbabilityChart(container);
      if (legend) {
        legend.innerHTML = "";
        var a = el("span", "legend-item");
        var sw = el("span", "legend-swatch"); sw.style.background = "#5aa9ff";
        a.appendChild(sw); a.appendChild(document.createTextNode("Failure probability"));
        var b = el("span", "legend-item");
        var sw2 = el("span", "legend-swatch"); sw2.style.background = "#f0616d";
        b.appendChild(sw2);
        var thr = (STATE.probability.model && STATE.probability.model.threshold != null)
          ? Number(STATE.probability.model.threshold).toFixed(2) : "0.50";
        b.appendChild(document.createTextNode("Alarm threshold " + thr));
        legend.appendChild(a); legend.appendChild(b);
      }
      hideNotice("trendNotice");
      return;
    }
    if (STATE.error) {
      emptyChart(container, STATE.error);
      if (legend) legend.innerHTML = "";
      showNotice("trendNotice", STATE.error, true);
      return;
    }
    if (STATE.mode === "loading") {
      emptyChart(container, "Loading historical model scores…");
      if (legend) legend.innerHTML = "";
      hideNotice("trendNotice");
      return;
    }
    emptyChart(container, "No model scores are available for this history.");
    if (legend) legend.innerHTML = "";
    hideNotice("trendNotice");
  }

  function renderTrendsChart() {
    var container = byId("trendChart2");
    if (!container) return;
    if (STATE.error) {
      emptyChart(container, STATE.error);
      var lg = byId("trendLegend2"); if (lg) lg.innerHTML = "";
      showNotice("trendNotice2", STATE.error, true);
      return;
    }
    if (STATE.mode === "loading") {
      emptyChart(container, "Loading sensor traces…");
      hideNotice("trendNotice2");
      return;
    }
    if (renderSeriesChart(container, byId("trendLegend2"))) hideNotice("trendNotice2");
    else showNotice("trendNotice2", "No sensor traces are available for this history.", true);
  }

  /* ---------------------------------------------------------------------- *
   * Window statistics
   * ---------------------------------------------------------------------- */
  function renderTraceStats() {
    var host = byId("traceStats");
    if (!host) return;
    host.innerHTML = "";
    if (!STATE.sensors.length || !STATE.series[STATE.sensors[0].key]) {
      host.appendChild(el("li", "empty", "No signal statistics available."));
      return;
    }
    var nDays = STATE.dates.length;
    var count = Math.min(STATE.range, nDays);
    var start = nDays - count;
    var sub = byId("traceStatsSubtitle");
    if (sub) {
      sub.textContent = "Min / max over " + count + " days (" +
        fmtFull(STATE.dates[start]) + " – " + fmtFull(STATE.dates[STATE.dates.length - 1]) + ")";
    }
    STATE.sensors.forEach(function (s) {
      var seen = false, mn = Infinity, mx = -Infinity;
      STATE.series[s.key].slice(start).forEach(function (v) {
        if (isNum(v)) { seen = true; if (v < mn) mn = v; if (v > mx) mx = v; }
      });
      var li = el("li");
      li.appendChild(el("span", "k", s.label));
      li.appendChild(el("span", "v", seen ? Number(mn).toFixed(2) + " – " + Number(mx).toFixed(2) + " " + s.unit : "–"));
      host.appendChild(li);
    });
  }

  /* ---------------------------------------------------------------------- *
   * Timeline / warning table / episode track / events
   * ---------------------------------------------------------------------- */
  function renderTimeline() {
    var host = byId("warningTimeline");
    if (!host) return;
    host.innerHTML = "";
    if (!STATE.episodes.length) {
      host.appendChild(el("li", "empty", STATE.error ? "Warning history is unavailable." : "No alarm episodes for this history."));
      return;
    }
    STATE.episodes.slice().reverse().forEach(function (ep) {
      var li = el("li");
      li.style.setProperty("--accent", sevColor(ep.severity));
      li.appendChild(el("div", "tl-time", fmtFull(parseISO(ep.end))));
      li.appendChild(el("div", "tl-title", ep.id + (ep.peak != null ? " — peak score " + Number(ep.peak).toFixed(2) : "")));
      li.appendChild(el("div", "tl-desc", (ep.model || "-") + " · " + ep.status));
      host.appendChild(li);
    });
  }

  function renderWarningTable() {
    var tbody = document.querySelector("#warningTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!STATE.episodes.length) {
      var msg = STATE.error ? "Warning history is unavailable." : "No warning episodes for this history.";
      tbody.innerHTML = '<tr><td colspan="7" class="empty">' + msg + "</td></tr>";
      return;
    }
    STATE.episodes.slice().reverse().forEach(function (ep) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + ep.id + "</td>" +
        "<td>" + fmtFull(parseISO(ep.start)) + "</td>" +
        "<td>" + fmtFull(parseISO(ep.end)) + "</td>" +
        "<td>" + (ep.days != null ? ep.days : "–") + "</td>" +
        "<td>" + (ep.peak != null ? Number(ep.peak).toFixed(2) : "–") + "</td>" +
        "<td>" + (ep.model || "–") + "</td>" +
        '<td class="sev sev-' + sevClass(ep.severity) + '">Indication</td>';
      tbody.appendChild(tr);
    });
  }

  function renderEpisodeTrack() {
    var host = byId("episodeTrack");
    if (!host) return;
    host.innerHTML = "";
    if (!STATE.episodes.length || !STATE.coverage || !STATE.coverage.start) {
      showNotice("episodeNotice", STATE.error ? STATE.error : "No alarm episodes to place on the timeline.", !!STATE.error);
      return;
    }
    hideNotice("episodeNotice");
    var startMs = parseISO(STATE.coverage.start).getTime();
    var days = Math.max(1, STATE.coverage.days || 1);
    STATE.episodes.forEach(function (ep) {
      var s = Math.round((parseISO(ep.start).getTime() - startMs) / DAY_MS);
      var e = Math.round((parseISO(ep.end).getTime() - startMs) / DAY_MS);
      var left = Math.max(0, Math.min(100, (s / days) * 100));
      var width = Math.max(0.7, ((e - s + 1) / days) * 100);
      var bar = el("div", "episode-bar" + (ep.severity === "high" ? " is-high" : ep.severity === "med" ? " is-med" : ""));
      bar.style.left = left + "%";
      bar.style.width = Math.min(100 - left, width) + "%";
      bar.title = ep.id + " · " + fmtFull(parseISO(ep.start)) + " → " + fmtFull(parseISO(ep.end)) +
        (ep.peak != null ? " · peak " + Number(ep.peak).toFixed(2) : "");
      host.appendChild(bar);
    });
    var axis = el("div", "episode-axis");
    axis.appendChild(el("span", null, fmtFull(parseISO(STATE.coverage.start))));
    axis.appendChild(el("span", null, fmtFull(new Date(startMs + (days - 1) * DAY_MS))));
    host.appendChild(axis);
  }

  function renderEvents() {
    var tbody = document.querySelector("#recentEvents tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!STATE.events.length) {
      var msg = STATE.error ? "Warning records are unavailable." : "No warning records for this history.";
      tbody.innerHTML = '<tr><td colspan="5" class="empty">' + msg + "</td></tr>";
      return;
    }
    STATE.events.forEach(function (e) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + fmtFull(parseISO(e.date)) + "</td>" +
        "<td>" + e.type + "</td>" +
        "<td>" + e.signal + "</td>" +
        '<td class="sev sev-' + sevClass(e.severity) + '">' + String(e.severity).toUpperCase() + "</td>" +
        "<td>" + e.detail + "</td>";
      tbody.appendChild(tr);
    });
  }

  /* ---------------------------------------------------------------------- *
   * Model insights
   * ---------------------------------------------------------------------- */
  function renderInsights() {
    var ins = STATE.insights || { models: [], metrics: [], limitations: [], bundles_present: false };
    var host = byId("modelCards");
    if (host) {
      host.innerHTML = "";
      var p = STATE.probability;
      if (p && p.model) {
        var card = el("div", "card");
        card.style.setProperty("--accent", ACCENTS[0]);
        card.appendChild(el("div", "card-label", "Active scoring model"));
        card.appendChild(el("div", "card-value", p.model.label || p.model.id || "Selected model"));
        var v = el("div", "card-value");
        v.appendChild(el("small", null, (p.scored != null ? p.scored + " days scored · " : "") +
          "horizon " + p.model.horizon_days + "d · threshold " + Number(p.model.threshold).toFixed(2)));
        card.appendChild(v);
        var cf = el("div", "card-foot");
        cf.appendChild(el("span", "pill pill-ok", "Scoring active"));
        cf.appendChild(document.createTextNode(" Real historical inference"));
        card.appendChild(cf);
        host.appendChild(card);
      }
      (ins.models || []).forEach(function (m, i) {
        var mc = el("div", "card");
        mc.style.setProperty("--accent", ACCENTS[(i + 1) % ACCENTS.length]);
        mc.appendChild(el("div", "card-label", m.id === STATE.modelId ? "Selected model" : "Model bundle"));
        mc.appendChild(el("div", "card-value", m.name));
        var mv = el("div", "card-value");
        mv.appendChild(el("small", null, "horizon " + m.horizon_days + "d · threshold " + Number(m.threshold).toFixed(2) +
          " · " + m.n_core_sensors + " core sensors"));
        mc.appendChild(mv);
        var f = el("div", "card-foot");
        f.appendChild(el("span", "pill pill-" + (m.id === STATE.modelId ? "ok" : "neutral"),
          m.id === STATE.modelId ? "In use" : "Available"));
        mc.appendChild(f);
        host.appendChild(mc);
      });
    }
    renderInsightsStatus();
    renderMetrics((ins.metrics || []));
    renderLimitations((ins.limitations || []));
  }

  function renderInsightsStatus() {
    var host = byId("modelStatusList");
    if (!host) return;
    host.innerHTML = "";
    var ins = STATE.insights || {};
    var modelLabel = (STATE.probability && STATE.probability.model)
      ? (STATE.probability.model.label + " (" + STATE.probability.model.id + ")")
      : (STATE.modelId || "none");
    var inference = STATE.probability ? "Active (real historical scoring)"
      : STATE.mode === "unavailable" ? "Unavailable" : "Loading";
    var rows = [
      ["Scoring model", modelLabel],
      ["Model bundles", ins.bundles_present ? "Present (local, read-only)" : "Not present"],
      ["Real inference", inference],
      ["Model horizon", STATE.probability && STATE.probability.model ? STATE.probability.model.horizon_days + " days" : "–"],
      ["Alarm threshold", STATE.probability && STATE.probability.model ? Number(STATE.probability.model.threshold).toFixed(2) : "–"],
      ["Metrics source", (ins.metrics && ins.metrics.length) ? "Aggregate local reports" : "Not available"],
      ["Data source", sourceLabel()],
    ];
    rows.forEach(function (r) {
      var li = el("li");
      li.appendChild(el("span", "k", r[0]));
      li.appendChild(el("span", "v", r[1]));
      host.appendChild(li);
    });
  }

  function renderMetrics(metrics) {
    var tbody = document.querySelector("#metricsTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!metrics.length) {
      tbody.innerHTML = '<tr><td colspan="4" class="empty">No aggregate metrics available.</td></tr>';
      showNotice("metricsNotice", "No aggregate metric artifacts are present on this machine.", true);
      return;
    }
    hideNotice("metricsNotice");
    metrics.forEach(function (rec) {
      var group = rec.group || "metric";
      var model = rec.model || rec.approach || "–";
      Object.keys(rec).forEach(function (k) {
        if (k === "group" || k === "model" || k === "approach") return;
        if (rec[k] === null || rec[k] === undefined) return;
        var val = typeof rec[k] === "number" ? Math.round(rec[k] * 1000) / 1000 : rec[k];
        var tr = document.createElement("tr");
        tr.innerHTML = "<td>" + group + "</td><td>" + model + "</td><td>" + k + "</td><td>" + val + "</td>";
        tbody.appendChild(tr);
      });
    });
  }

  function renderLimitations(items) {
    var host = byId("limitationsList");
    if (!host) return;
    host.innerHTML = "";
    (items || []).forEach(function (t) {
      var li = el("li");
      li.appendChild(el("span", "v", t));
      host.appendChild(li);
    });
    if (!items || !items.length) host.appendChild(el("li", "empty", "No limitations recorded."));
  }

  /* ---------------------------------------------------------------------- *
   * Chrome
   * ---------------------------------------------------------------------- */
  function setBadges(text, cls) {
    ["topBadge", "sidebarBadge", "overviewBadge", "warningBadge"].forEach(function (id) {
      var n = byId(id);
      if (!n) return;
      n.textContent = text;
      n.className = "pill " + cls;
    });
  }

  function updateChrome() {
    var hist = STATE.mode === "historical";
    var chip = byId("sourceChip");
    if (chip) chip.textContent = sourceLabel();
    var line = byId("pageSourceLine");
    if (line) line.textContent = STATE.source ? STATE.source.label + " — " + STATE.source.detail : "…";

    if (hist) {
      setBadges("HISTORICAL", "pill-demo");
    } else if (STATE.mode === "loading") {
      setBadges("LOADING", "pill-neutral");
    } else {
      setBadges("NO DATA", "pill-neutral");
    }

    var note = byId("sidebarNote");
    if (note) {
      note.textContent = hist
        ? "Offline historical records, scored locally. Not a live context."
        : "Real historical scoring is not available on this machine.";
    }
    var sub = byId("trendSubtitle");
    if (sub) {
      sub.textContent = (hist && STATE.probability)
        ? "Model failure probability (horizon " + STATE.probability.model.horizon_days + "d, threshold " + Number(STATE.probability.model.threshold).toFixed(2) + ")"
        : "Model failure probability";
    }
    var sub2 = byId("trendSubtitle2");
    if (sub2) sub2.textContent = "Scaled core sensors under generic labels";
    var ev = byId("eventsSubtitle");
    if (ev) ev.textContent = "Model alarms (indication only)";
    var tl = byId("timelineSubtitle");
    if (tl) tl.textContent = "Model-generated alarm periods";
    var ws = byId("warningSubtitle");
    if (ws) ws.textContent = "Model-generated alarm episodes (offline history)";

    if (hist) {
      showNotice("warningDisclaimer",
        "These are model-generated alarm indications over offline history, not confirmed failure events. " +
        "The single documented failure label is provisional; treat alarms as a prompt to investigate.", false);
    } else if (STATE.mode === "loading") {
      showNotice("warningDisclaimer", "Loading historical model scores…", false);
    } else {
      showNotice("warningDisclaimer", STATE.error || "Historical scoring is unavailable.", true);
    }

    var ms = byId("metricsSubtitle");
    if (ms) {
      ms.textContent = (STATE.insights && STATE.insights.metrics && STATE.insights.metrics.length)
        ? "Aggregate results from local reports"
        : "No local report artifacts";
    }
    var mss = byId("modelSelectSubtitle");
    if (mss) mss.textContent = "Select the model used for historical scoring";

    setModelSelectorEnabled(hist);

    window.__signal_monitor = {
      mode: STATE.mode,
      ready: STATE.ready === true,
      source: STATE.source ? STATE.source.id : null,
      model: STATE.modelId,
      days: STATE.dates.length,
      range: STATE.range,
      coverage: STATE.coverage || { start: null, end: null, days: 0 },
      label: STATE.source ? STATE.source.label : null,
      hasProbability: !!STATE.probability,
      error: STATE.error || null,
    };
  }

  /* ---------------------------------------------------------------------- *
   * Paint
   * ---------------------------------------------------------------------- */
  function paint() {
    renderSummary();
    renderOverviewChart();
    renderTrendsChart();
    renderTimeline();
    renderWarningTable();
    renderEpisodeTrack();
    renderEvents();
    renderTraceStats();
    renderInsights();
    updateChrome();
  }

  /* ---------------------------------------------------------------------- *
   * Range controls (synchronized)
   * ---------------------------------------------------------------------- */
  function maxDays() {
    var a = STATE.dates.length;
    var b = (STATE.probability && STATE.probability.points) ? STATE.probability.points.length : 0;
    return Math.max(a, b, 1);
  }

  function setRange(n) {
    var cap = maxDays();
    if (n > cap) n = cap;
    STATE.range = n;
    document.querySelectorAll("#rangeSelector button, #rangeSelector2 button").forEach(function (b) {
      var r = Number(b.dataset.range);
      b.classList.toggle("is-active", r === n);
      b.disabled = r > cap;
    });
    renderOverviewChart();
    renderTrendsChart();
    renderTraceStats();
    if (window.__signal_monitor) window.__signal_monitor.range = n;
  }

  function wireRanges() {
    ["rangeSelector", "rangeSelector2"].forEach(function (id) {
      var group = byId(id);
      if (!group) return;
      group.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () { if (!b.disabled) setRange(Number(b.dataset.range)); });
      });
    });
  }

  /* ---------------------------------------------------------------------- *
   * Model selector
   * ---------------------------------------------------------------------- */
  function buildModelSelector(models, defaultModel) {
    var host = byId("modelSelector");
    if (!STATE.modelId) STATE.modelId = defaultModel || (models && models[0] && models[0].id) || null;
    if (!host) return;
    host.innerHTML = "";
    (models || []).forEach(function (m) {
      var btn = el("button", "mode-btn" + (STATE.modelId === m.id ? " is-active" : ""), m.label || m.id);
      btn.type = "button";
      btn.dataset.model = m.id;
      btn.addEventListener("click", function () { selectModel(m.id); });
      host.appendChild(btn);
    });
    setModelSelectorEnabled(STATE.mode === "historical");
  }

  function setModelSelectorEnabled(enabled) {
    document.querySelectorAll("#modelSelector .mode-btn").forEach(function (b) {
      b.disabled = !enabled;
    });
  }

  function syncModelSelector() {
    document.querySelectorAll("#modelSelector .mode-btn").forEach(function (b) {
      b.classList.toggle("is-active", b.dataset.model === STATE.modelId);
    });
  }

  function selectModel(id) {
    if (!id || STATE.mode !== "historical") return;
    STATE.modelId = id;
    syncModelSelector();
    loadHistorical(id);
  }

  /* ---------------------------------------------------------------------- *
   * Data loading
   * ---------------------------------------------------------------------- */
  function loadInsights() {
    return getJSON("/api/model/insights").then(function (res) {
      if (res.ok && res.data) STATE.insights = res.data;
    });
  }

  function showUnavailable(reason) {
    STATE.mode = "unavailable";
    STATE.ready = false;
    STATE.error = reason || "Real historical scoring data is not available on this machine.";
    STATE.probability = null;
    STATE.dates = [];
    STATE.sensors = [];
    STATE.series = {};
    STATE.episodes = [];
    STATE.events = [];
    STATE.summary = [];
    STATE.coverage = { start: null, days: 0 };
    paint();
    setRange(STATE.range);
  }

  function loadHistorical(modelId) {
    var mid = modelId || STATE.modelId || "logreg";
    STATE.mode = "loading";
    STATE.error = null;
    paint();
    return Promise.all([
      getJSON("/api/historical/scores?model=" + encodeURIComponent(mid)),
      getJSON("/api/historical/series?model=" + encodeURIComponent(mid)),
      getJSON("/api/model/insights"),
    ]).then(function (res) {
      var scores = res[0], series = res[1], ins = res[2];
      if (ins.ok && ins.data) STATE.insights = ins.data;
      if (!scores.ok || !scores.data || !scores.data.points || !scores.data.points.length) {
        var reason = scores.message || (scores.data && scores.data.message) ||
          "Real historical model scores could not be produced from the local artifacts.";
        showUnavailable(reason);
        renderInsights();
        return;
      }
      STATE.modelId = scores.data.model ? scores.data.model.id : STATE.modelId;
      applyHistorical(scores.data, series.ok && series.data ? series.data : {});
      STATE.hidden = {};
      paint();
      setRange(STATE.range);
      syncModelSelector();
    });
  }

  /* ---------------------------------------------------------------------- *
   * Navigation + clock
   * ---------------------------------------------------------------------- */
  function setupNav() {
    var items = document.querySelectorAll(".nav-item");
    var views = document.querySelectorAll(".view");
    var sidebar = document.querySelector(".sidebar");
    items.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var view = btn.dataset.view;
        items.forEach(function (x) { x.classList.toggle("is-active", x === btn); });
        views.forEach(function (v) { v.hidden = v.id !== "view-" + view; });
        if (window.matchMedia("(max-width: 860px)").matches) sidebar.classList.remove("is-open");
      });
    });
    var toggle = byId("navToggle");
    if (toggle) toggle.addEventListener("click", function () { sidebar.classList.toggle("is-open"); });
  }

  function setupClock() {
    var node = byId("clock");
    if (node) node.textContent = "Prototype";
  }

  /* ---------------------------------------------------------------------- *
   * Boot
   * ---------------------------------------------------------------------- */
  function init() {
    setupNav();
    setupClock();
    wireRanges();
    STATE.mode = "loading";
    paint();
    setRange(DEFAULT_RANGE);

    loadInsights().then(function () { renderInsights(); updateChrome(); });

    getJSON("/api/modes").then(function (res) {
      var m = res.ok ? res.data : null;
      if (!m) {
        buildModelSelector([]);
        showUnavailable(res.message || "The local backend could not be reached.");
        return;
      }
      STATE.models = m.models || [];
      STATE.modelId = m.default_model || STATE.modelId;
      buildModelSelector(STATE.models, m.default_model);
      if (m.ready && STATE.models.length) {
        loadHistorical(STATE.modelId);
      } else {
        showUnavailable(m.detail || "Real historical scoring is not available on this machine.");
      }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
