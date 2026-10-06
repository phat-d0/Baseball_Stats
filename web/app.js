"use strict";

// ---------- state ----------
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const state = {
  data: null,
  tab: "games", // the app always opens on today's games
  day: 0, // index into data.slates
  batLine: store.get("batLine", 1.5),
  pitLine: store.get("pitLine", 5.5),
  recKind: store.get("recKind", "batter"),
  minEdge: store.get("minEdge", 0.05),
  calLine: null, // calibration chart: null = all lines pooled
  query: "",
  sheetLine: null, // line picked inside an open player sheet
  sheetPlayer: null,
};

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (x, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const fix = (x, d = 1) => (x == null ? "–" : x.toFixed(d));
const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][(n % 100 > 10 && n % 100 < 14) || n % 10 > 3 ? 0 : n % 10]}`;
const handName = (h) => ({ L: "LHP", R: "RHP" })[h] || "";
const batsName = (h) => ({ L: "bats L", R: "bats R", S: "switch" })[h] || "";

// P(over line) from a pmf whose last entry is the "or more" tail.
const pOver = (pmf, line) => pmf.slice(Math.floor(line) + 1).reduce((a, b) => a + b, 0);

// Keep the first n-1 counts and fold the rest into an "n-1 or more" bar.
const collapse = (pmf, n) => pmf.slice(0, n - 1).concat([pmf.slice(n - 1).reduce((a, b) => a + b, 0)]);

// Fair (no-vig) American odds for a probability: compare with your sportsbook's price.
function fairOdds(p) {
  if (p == null || p <= 0 || p >= 1) return "–";
  const a = p >= 0.5 ? Math.round((-100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p);
  return a > 0 ? `+${a}` : `−${Math.abs(a)}`;
}

function gameTime(iso) {
  if (!iso) return "TBD";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

// ---------- DraftKings props ----------
const EDGE_STEPS = [0.02, 0.05, 0.08, 0.12];
const american = (a) => (a == null ? "–" : a > 0 ? `+${a}` : `−${Math.abs(a)}`);
const signedPct = (x, d = 0) => (x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)}%`);
const CHECK = '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M3.5 8.5l3 3 6-7"/></svg>';

// Best-paying side across a player's DraftKings lines: {line, side, price, ev, p}.
function bestBet(player) {
  let best = null;
  for (const b of player.book || []) {
    for (const side of ["over", "under"]) {
      const ev = b[`ev_${side}`];
      if (ev == null || b[side] == null) continue;
      if (!best || ev > best.ev) {
        best = { line: b.line, side, price: b[side], ev, p: side === "over" ? b.p_model : 1 - b.p_model, fetchedAt: b.fetched_at };
      }
    }
  }
  return best;
}
function mainBook(player, line) {
  const book = player.book || [];
  return book.find((b) => b.line === line) || book[Math.floor(book.length / 2)] || null;
}
// "DK 5.5: O −120 / U +100" for the line shown (or the player's main DK line).
function bookMetaAt(player, line) {
  const b = mainBook(player, line);
  return b ? ` · DK ${b.line}: O ${american(b.over)} / U ${american(b.under)}` : "";
}
function valueBadge(player) {
  const bb = bestBet(player);
  if (!bb || bb.ev < state.minEdge) return "";
  return `<span class="badge">${CHECK}${bb.side === "over" ? "Over" : "Under"} ${bb.line} ${american(bb.price)} · ${signedPct(bb.ev)}</span>`;
}
function oddsNote() {
  const s = state.data?.odds_source;
  if (!s || (s.error && s.error.startsWith("no ODDS_API_KEY"))) return "";
  if (s.error === "no games today") return "";
  const bits = [];
  if (s.fetched_at) {
    const mins = Math.round((Date.now() - Date.parse(s.fetched_at)) / 60000);
    bits.push(`DraftKings props updated ${mins < 60 ? `${mins} min` : `${Math.round(mins / 60)} h`} ago for ${s.events_priced} of ${s.events_total} games.`);
  } else {
    bits.push("No DraftKings props downloaded yet today.");
  }
  if (s.daily_allowance != null) bits.push(`Budget ${s.spent_today}/${s.daily_allowance} credits today${s.credits_left != null ? `, ${s.credits_left} left this month` : ""}.`);
  if (s.error) bits.push(esc(s.error));
  return `<p class="note">${bits.join(" ")}</p>`;
}
function edgeControl() {
  return `<div class="control-label"><span>Minimum edge</span><span>${pct(state.minEdge)}</span></div>
    <div class="segmented">${EDGE_STEPS.map((e) => `<button data-edge="${e}" class="${e === state.minEdge ? "on" : ""}">${pct(e)}</button>`).join("")}</div>`;
}
function valuePicks() {
  if (slate()?.label !== "Today") return "";
  const all = [...allPitchers().map((p) => ({ ...p, kind: "pitcher" })), ...allBatters().map((b) => ({ ...b, kind: "batter" }))];
  if (!all.some((p) => p.book?.length)) return oddsNote();
  // Batters from a projected lineup stay out: they may not play (they're still in Hitters).
  const picks = all.filter((p) => p.kind === "pitcher" || p.confirmed !== false)
    .map((p) => ({ p, bb: bestBet(p) })).filter((x) => x.bb && x.bb.ev >= state.minEdge)
    .sort((a, b) => b.bb.ev - a.bb.ev).slice(0, 12);
  const rows = picks.map(({ p, bb }) => {
    const mins = bb.fetchedAt ? Math.round((Date.now() - Date.parse(bb.fetchedAt)) / 60000) : null;
    const age = mins == null ? "" : `<span class="meta">Price ${mins < 60 ? `${mins} min` : `${(mins / 60).toFixed(1)} h`} old</span>`;
    return `
    <button class="row-btn ${mins != null && mins > 90 ? "stale" : ""}" data-player="${p.kind}" data-game="${p.game.game_pk}" data-id="${p.id}">
      <span class="who"><b>${esc(p.name)}</b>
        <span class="meta">${bb.side === "over" ? "Over" : "Under"} ${bb.line} ${p.kind === "pitcher" ? "K" : "H+R+RBI"} · ${esc(teamAbbr(p.team))} ${p.side === "home" ? "vs" : "@"} ${esc(teamAbbr(p.opp))}</span>${age}</span>
      <span class="vals"><b class="pos-text">${signedPct(bb.ev)}</b><small>${american(bb.price)} · model ${pct(bb.p)}</small></span>
    </button>`;
  }).join("");
  return `<h2 class="section-title">Value picks · DraftKings</h2>
    ${edgeControl()}
    ${picks.length ? `<div class="card list">${rows}</div>` : `<div class="card"><span class="muted">Nothing above a ${pct(state.minEdge)} edge right now.</span></div>`}
    <p class="note">Edge = how much the model expects a $1 bet to return above your stake at DraftKings' price. Small edges are mostly noise; check Record → vs DraftKings before trusting them. Greyed = price over 90 minutes old. Hitters from projected lineups are left out until their lineup posts.</p>
    ${oddsNote()}`;
}
function bookTable(player) {
  if (!player.book?.length) return "";
  const cell = (ev) => `<td class="${ev != null && ev >= state.minEdge ? "edge-pos" : ""}">${signedPct(ev)}</td>`;
  const rows = player.book.map((b) => `<tr><td>${b.line}</td><td>${american(b.over)}<br><span class="muted small">${american(b.under)}</span></td><td>${pct(b.p_model)}</td><td>${pct(b.p_book)}</td>${cell(b.ev_over)}${cell(b.ev_under)}</tr>`).join("");
  return `<h3 class="section-title">DraftKings</h3>
    <table><thead><tr><th>Line</th><th>O / U</th><th>Model</th><th>DK</th><th>Edge O</th><th>Edge U</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="note">Model and DK = chance of the over; DK's is its price with the margin taken out.</p>`;
}

// ---------- data access ----------
const slates = () => state.data?.slates || [];
const slate = () => slates()[Math.min(state.day, Math.max(slates().length - 1, 0))];
const lines = (kind) => state.data?.lines?.[kind] || (kind === "batter" ? [0.5, 1.5, 2.5] : [3.5, 4.5, 5.5, 6.5, 7.5]);
const teamAbbr = (t) => t?.abbr || t?.name || "";

function allPitchers() {
  const out = [];
  for (const g of slate()?.games || []) {
    for (const side of ["away", "home"]) {
      const p = g.pitchers[side];
      if (p) out.push({ ...p, game: g, side, team: g[side], opp: g[side === "home" ? "away" : "home"] });
    }
  }
  return out;
}
function allBatters() {
  const out = [];
  for (const g of slate()?.games || []) {
    for (const side of ["away", "home"]) {
      const oppSide = side === "home" ? "away" : "home";
      for (const b of g.lineups[side]) {
        out.push({ ...b, game: g, side, team: g[side], opp: g[oppSide], oppSp: g.pitchers[oppSide] });
      }
    }
  }
  return out;
}
function findPlayer(kind, gamePk, id) {
  const list = kind === "pitcher" ? allPitchers() : allBatters();
  return list.find((p) => p.game.game_pk === gamePk && p.id === id);
}

// ---------- shared controls ----------
function daySwitch() {
  if (slates().length < 2) return slate() ? `<p class="note">${esc(slate().label)} · ${esc(longDate(slate().date))}</p>` : "";
  return `<div class="segmented" role="tablist" aria-label="Day">${slates()
    .map((s, i) => `<button data-day="${i}" class="${i === state.day ? "on" : ""}" role="tab" aria-selected="${i === state.day}">${esc(s.label)}</button>`)
    .join("")}</div>`;
}
function lineSwitch(kind, current, attr) {
  return `<div class="segmented" aria-label="Line">${lines(kind)
    .map((l) => `<button ${attr}="${l}" class="${l === current ? "on" : ""}">Over ${l}</button>`)
    .join("")}</div>`;
}
function longDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" });
}
function shortDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString([], { month: "short", day: "numeric" });
}
function noGames() {
  return `<div class="empty">No games scheduled in the next week.<br><span class="muted">The Record tab still shows how the model did.</span></div>`;
}

// ---------- charts ----------
// Bars for P(exactly k); bars above the line are highlighted.
function distChart(pmf, line, unit) {
  const n = pmf.length;
  const W = 340, H = 130, top = 16, bottom = 18, gap = 2;
  const bw = (W - gap * (n - 1)) / n;
  const max = Math.max(...pmf, 0.01);
  let bars = "";
  pmf.forEach((p, k) => {
    const h = ((H - top - bottom) * p) / max;
    const x = k * (bw + gap);
    const y = H - bottom - h;
    const over = k > line;
    const lab = k === n - 1 ? `${k}+` : `${k}`;
    const r = Math.min(4, bw / 2, h);
    const path = h > 0
      ? `M${x},${H - bottom} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${H - bottom} Z`
      : "";
    bars += `<path class="bar ${over ? "over" : ""}" d="${path}"/>`;
    if (p >= 0.04) bars += `<text class="val" x="${x + bw / 2}" y="${y - 4}">${Math.round(p * 100)}</text>`;
    bars += `<text class="lbl" x="${x + bw / 2}" y="${H - 4}">${lab}</text>`;
    bars += `<rect class="hit" x="${x - gap / 2}" y="0" width="${bw + gap}" height="${H}" data-tip="${esc(`${lab} ${unit}: ${(p * 100).toFixed(1)}%`)}" data-x="${((x + bw / 2) / W) * 100}"/>`;
  });
  return `
    <div class="chart dist">
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Chance of each ${esc(unit)} total">${bars}</svg>
      <div class="tooltip" hidden></div>
    </div>
    <div class="legend"><span><i style="background:var(--accent)"></i>Over ${line}</span><span><i style="background:var(--track)"></i>Under</span><span class="muted">Bar labels are %</span></div>`;
}

// Predicted probability vs how often it happened. On the dashed line = well calibrated.
function calibrationChart(points) {
  const W = 320, H = 220, L = 30, R = 8, T = 8, B = 26;
  const sx = (v) => L + v * (W - L - R);
  const sy = (v) => T + (1 - v) * (H - T - B);
  let grid = "";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    grid += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L - 4}" y="${sy(v) + 3}" text-anchor="end">${v * 100}</text>`;
    grid += `<text x="${sx(v)}" y="${H - 8}" text-anchor="middle">${v * 100}</text>`;
  }
  const maxN = Math.max(...points.map((p) => p.n), 1);
  const path = points.map((p, i) => `${i ? "L" : "M"}${sx(p.p)},${sy(p.actual)}`).join(" ");
  const dots = points.map((p) => {
    const r = 4 + 4 * Math.sqrt(p.n / maxN);
    return `<circle class="dot" cx="${sx(p.p)}" cy="${sy(p.actual)}" r="${r}"/>` +
      `<circle class="hit" cx="${sx(p.p)}" cy="${sy(p.actual)}" r="${r + 10}" fill="transparent" data-tip="${esc(`Model ${pct(p.p)} → happened ${pct(p.actual)} (${p.n.toLocaleString()} picks)`)}" data-x="${(sx(p.p) / W) * 100}"/>`;
  }).join("");
  return `
    <div class="chart">
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration: predicted chance against actual frequency">
        <g class="grid">${grid}</g>
        <line class="diag" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}"/>
        <path class="line" d="${path}"/>
        ${dots}
      </svg>
      <div class="tooltip" hidden></div>
    </div>
    <p class="note">Across: the model's chance of going over (%). Up: how often it actually went over (%). Dots on the dashed line mean the chances can be taken at face value; bigger dots = more picks.</p>`;
}

function bindCharts(root = document) {
  root.querySelectorAll(".chart").forEach((chart) => {
    const tip = $(".tooltip", chart);
    if (!tip) return;
    const show = (ev) => {
      const t = ev.target.closest("[data-tip]");
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.dataset.tip;
      tip.style.left = `${Math.min(Math.max(Number(t.dataset.x), 18), 82)}%`;
      tip.hidden = false;
    };
    chart.addEventListener("pointermove", show);
    chart.addEventListener("pointerdown", show);
    chart.addEventListener("pointerleave", () => { tip.hidden = true; });
  });
}

// ---------- games ----------
function pitcherCell(p) {
  if (!p) return `<div class="kproj"><span class="muted small">Starter TBD</span></div>`;
  return `<div class="kproj"><b>${fix(p.mu)}</b><small>proj K</small></div>`;
}
function gameCard(g) {
  const conf = (side) => g.lineups[side].length && g.lineups[side].every((b) => b.confirmed);
  const lineupChip = (side) => {
    if (!g.lineups[side].length) return "";
    return conf(side)
      ? `<span class="chip ok">${esc(teamAbbr(g[side]))} lineup posted</span>`
      : `<span class="chip">${esc(teamAbbr(g[side]))} lineup projected</span>`;
  };
  const weather = [g.temp_f != null ? `${Math.round(g.temp_f)}°F` : "", g.wind || ""].filter(Boolean).join(", ");
  const sp = (side) => g.pitchers[side] ? `${esc(g.pitchers[side].name)} <span class="muted">${handName(g.pitchers[side].hand)}</span>` : "TBD";
  return `
    <button class="card game" data-game="${g.game_pk}">
      <div class="game-head"><span>${gameTime(g.time)}${g.venue ? ` · ${esc(g.venue)}` : ""}</span><span>${esc(weather)}</span></div>
      <div class="matchup">
        <div class="team-line"><b>${esc(g.away.name)}</b><span>${sp("away")}</span></div>${pitcherCell(g.pitchers.away)}
        <div class="team-line"><b>@ ${esc(g.home.name)}</b><span>${sp("home")}</span></div>${pitcherCell(g.pitchers.home)}
      </div>
      <div class="chips">${lineupChip("away")}${lineupChip("home")}</div>
    </button>`;
}
function viewGames() {
  if (!slate()) return noGames();
  const games = slate().games;
  return `${daySwitch()}
    ${valuePicks()}
    <h2 class="section-title">${games.length} game${games.length === 1 ? "" : "s"}</h2>
    ${games.map(gameCard).join("")}
    <p class="note">Tap a game for both lineups. Until a team posts its lineup, its last starting nine stands in.</p>`;
}

function batterRow(b, line, showTeam = true) {
  const p = pOver(b.pmf, line);
  const where = showTeam ? `${esc(teamAbbr(b.team))} ${b.side === "home" ? "vs" : "@"} ${esc(teamAbbr(b.opp))} · ` : "";
  const vs = b.oppSp ? `vs ${esc(b.oppSp.name)} (${b.oppSp.hand || "?"})` : "starter TBD";
  return `
    <button class="row-btn" data-player="batter" data-game="${b.game.game_pk}" data-id="${b.id}">
      <span class="who">${showTeam ? "" : `<span class="order">${b.order ?? ""}</span>`}<b>${esc(b.name)}</b>${b.confirmed ? "" : '<span class="tag">proj</span>'}
        <span class="meta">${where}${b.order ? `bats ${ordinal(b.order)} · ` : ""}${vs}${bookMetaAt(b, line)}</span>${valueBadge(b)}</span>
      <span class="vals"><b>${pct(p)}</b><small>proj ${fix(b.mu)}</small></span>
      <span class="meter" aria-hidden="true"><span style="width:${(p * 100).toFixed(1)}%"></span></span>
    </button>`;
}
function pitcherRow(p, line) {
  const po = pOver(p.pmf, line);
  return `
    <button class="row-btn" data-player="pitcher" data-game="${p.game.game_pk}" data-id="${p.id}">
      <span class="who"><b>${esc(p.name)}</b>
        <span class="meta">${esc(teamAbbr(p.team))} ${p.side === "home" ? "vs" : "@"} ${esc(teamAbbr(p.opp))} · ${gameTime(p.game.time)} · ${handName(p.hand)}${bookMetaAt(p, null)}</span>${valueBadge(p)}</span>
      <span class="vals"><b>${fix(p.mu)} K</b><small>over ${line}: ${pct(po)}</small></span>
      <span class="meter" aria-hidden="true"><span style="width:${(po * 100).toFixed(1)}%"></span></span>
    </button>`;
}

function gameSheet(g) {
  const line = state.batLine;
  const lineup = (side) => {
    const bs = g.lineups[side];
    const oppSide = side === "home" ? "away" : "home";
    if (!bs.length) return `<p class="note">No lineup yet.</p>`;
    const rows = bs.map((b) => batterRow({ ...b, game: g, side, team: g[side], opp: g[oppSide], oppSp: g.pitchers[oppSide] }, line, false));
    const posted = bs.every((b) => b.confirmed);
    return `<div class="card list">${rows.join("")}</div><p class="note">${posted ? "Official lineup." : "Projected from the last game; not posted yet."}</p>`;
  };
  const sps = ["away", "home"].map((side) => g.pitchers[side]
    ? pitcherRow({ ...g.pitchers[side], game: g, side, team: g[side], opp: g[side === "home" ? "away" : "home"] }, state.pitLine)
    : `<div class="row-btn"><span class="who"><b>${esc(teamAbbr(g[side]))} starter TBD</b></span></div>`).join("");
  const facts = [gameTime(g.time), g.venue, g.umpire ? `HP umpire ${g.umpire}` : "",
    g.temp_f != null ? `${Math.round(g.temp_f)}°F` : "", g.wind || ""].filter(Boolean).map(esc).join(" · ");
  return `
    <div class="detail">
      <h2 id="sheet-title">${esc(g.away.name)} @ ${esc(g.home.name)}</h2>
      <p class="sub">${facts}</p>
      <h3 class="section-title">Starting pitchers</h3>
      <div class="card list">${sps}</div>
      <h3 class="section-title">${esc(g.away.name)} · H+R+RBI over ${line}</h3>
      ${lineup("away")}
      <h3 class="section-title">${esc(g.home.name)} · H+R+RBI over ${line}</h3>
      ${lineup("home")}
    </div>`;
}

// ---------- player sheet ----------
function statGrid(pairs) {
  return `<div class="stat-grid">${pairs
    .filter(([, v]) => v != null && v !== "–")
    .map(([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("")}</div>`;
}
function overTable(pmf, kind) {
  const rows = lines(kind).map((l) => {
    const p = pOver(pmf, l);
    return `<tr><td>${l}</td><td>${pct(p)}</td><td>${fairOdds(p)}</td><td>${pct(1 - p)}</td><td>${fairOdds(1 - p)}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>Line</th><th>Over</th><th>Fair</th><th>Under</th><th>Fair</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="note">Fair = the model's chance as American odds, with no bookmaker margin. A bet is worth a look only when your sportsbook pays more than this.</p>`;
}
function playerSheet(kind, p) {
  const dk = mainBook(p, null);
  const fallback = kind === "pitcher" ? state.pitLine : state.batLine;
  const line = state.sheetLine ?? (dk && lines(kind).includes(dk.line) ? dk.line : fallback);
  const where = `${esc(teamAbbr(p.team))} ${p.side === "home" ? "vs" : "@"} ${esc(teamAbbr(p.opp))} · ${gameTime(p.game.time)}`;
  const s = p.stats || {};
  const r1 = (x) => (x == null ? null : x.toFixed(1));
  const r2 = (x) => (x == null ? null : x.toFixed(2));
  const factor = (x) => (x == null ? null : `${x >= 1 ? "+" : "−"}${Math.abs((x - 1) * 100).toFixed(0)}%`);
  if (kind === "pitcher") {
    return `
      <div class="detail">
        <h2 id="sheet-title">${esc(p.name)}</h2>
        <p class="sub">${where} · ${handName(p.hand)}</p>
        <div class="hero"><b>${fix(p.mu)}</b><span>projected strikeouts</span></div>
        ${lineSwitch("pitcher", line, "data-sheet-line")}
        ${distChart(collapse(p.pmf, 13), line, "K")}
        ${bookTable(p)}
        ${overTable(p.pmf, "pitcher")}
        <h3 class="section-title">Why</h3>
        ${statGrid([
          ["K per start, last 5", r1(s.k_per_start_l5)], ["K per start, season", r1(s.k_per_start_szn)],
          ["K%, season", s.k_pct_szn == null ? null : pct(s.k_pct_szn, 1)], ["Whiff%, season", s.whiff_pct_szn == null ? null : pct(s.whiff_pct_szn, 1)],
          ["Outs per start, last 5", r1(s.outs_per_start_l5)], ["Pitches, last 5", r1(s.pitches_per_start_l5)],
          ["ERA, season", r2(s.era_szn)], ["Starts, season", s.starts_szn == null ? null : String(s.starts_szn)],
          ["Opp. lineup K% vs hand", s.lineup_k_pct == null ? null : pct(s.lineup_k_pct, 1)], ["Opp. team K%, season", s.opp_team_k_pct == null ? null : pct(s.opp_team_k_pct, 1)],
          ["Umpire K effect", factor(s.ump_k_factor)], ["Park K effect", factor(s.park_k_factor)],
          ["Days rest", s.days_rest == null ? null : String(s.days_rest)],
          ["Expected batters faced", r1(s.exp_bf)],
        ])}
      </div>`;
  }
  const vs = p.oppSp ? `vs ${esc(p.oppSp.name)} (${handName(p.oppSp.hand)})` : "starter TBD";
  return `
    <div class="detail">
      <h2 id="sheet-title">${esc(p.name)}</h2>
      <p class="sub">${where} · ${p.order ? `bats ${ordinal(p.order)}` : ""} ${p.confirmed ? "" : "(projected)"} · ${vs}</p>
      <div class="hero"><b>${fix(p.mu, 2)}</b><span>projected H+R+RBI</span></div>
      ${lineSwitch("batter", line, "data-sheet-line")}
      ${distChart(collapse(p.pmf, 7), line, "H+R+RBI")}
      ${bookTable(p)}
      ${overTable(p.pmf, "batter")}
      <h3 class="section-title">Why</h3>
      ${statGrid([
        ["H+R+RBI per game, last 15", r2(s.hrr_per_g_l15)], ["H+R+RBI per game, season", r2(s.hrr_per_g_szn)],
        [`Per PA vs ${p.vs === "L" ? "lefties" : "righties"}`, r2(s.hrr_per_pa_vs_hand)], ["PA per game, last 15", r1(s.pa_per_g_l15)],
        ["xwOBA, season", s.xwoba_szn == null ? null : s.xwoba_szn.toFixed(3)], ["Games, season", s.games_szn == null ? null : String(s.games_szn)],
        ["Opp. starter K%", s.opp_sp_k_pct == null ? null : pct(s.opp_sp_k_pct, 1)], ["Opp. starter ERA", r2(s.opp_sp_era)],
        ["Park run effect", factor(s.park_r_factor)], ["Bats", batsName(p.bats)],
      ])}
    </div>`;
}

// ---------- hitters / pitchers ----------
function matchesQuery(p) {
  const q = state.query.trim().toLowerCase();
  if (!q) return true;
  return [p.name, p.team?.name, p.team?.abbr].some((x) => (x || "").toLowerCase().includes(q));
}
function viewHitters() {
  if (!slate()) return noGames();
  const line = state.batLine;
  const list = allBatters().filter(matchesQuery).sort((a, b) => pOver(b.pmf, line) - pOver(a.pmf, line));
  return `${daySwitch()}
    ${lineSwitch("batter", line, "data-bat-line")}
    <input class="search" id="search" type="search" placeholder="Search player or team" value="${esc(state.query)}" autocomplete="off">
    <h2 class="section-title">Chance of H+R+RBI over ${line}</h2>
    ${list.length ? `<div class="card list">${list.slice(0, 150).map((b) => batterRow(b, line)).join("")}</div>` : '<div class="empty">No hitters yet for this day.</div>'}
    <p class="note">"proj" = lineup not posted yet; based on the team's last game.</p>`;
}
function viewPitchers() {
  if (!slate()) return noGames();
  const line = state.pitLine;
  const list = allPitchers().filter(matchesQuery).sort((a, b) => b.mu - a.mu);
  return `${daySwitch()}
    ${lineSwitch("pitcher", line, "data-pit-line")}
    <input class="search" id="search" type="search" placeholder="Search pitcher or team" value="${esc(state.query)}" autocomplete="off">
    <h2 class="section-title">Projected strikeouts</h2>
    ${list.length ? `<div class="card list">${list.map((p) => pitcherRow(p, line)).join("")}</div>` : '<div class="empty">No probable starters announced yet.</div>'}`;
}

// ---------- record ----------
const MIN_PICKS = 200; // below this, results are mostly luck
const MODEL_NAMES = {
  current: "game-level model",
  pa_simple: "plate-appearance model (each hitter's strikeout chance × batters faced)",
  pa_seq: "batter-by-batter simulation",
};
function modelNote(kind) {
  const m = state.data?.model?.[kind];
  if (!m?.name) return "";
  const shadow = m.shadow ? ` · ${esc(MODEL_NAMES[m.shadow] || m.shadow)} runs in shadow for comparison` : "";
  return `<p class="note">Active ${kind === "pitcher" ? "strikeout" : "H+R+RBI"} model: ${esc(MODEL_NAMES[m.name] || m.name)}${shadow}.</p>`;
}

function marketCard() {
  const m = state.data?.record?.market;
  const k = m?.[state.recKind];
  const head = `<h2 class="section-title">vs DraftKings</h2>`;
  if (!k || !k.by_threshold) {
    return `${head}<div class="card"><span class="muted">No graded DraftKings picks yet. Every price the app downloads is logged and graded after the game${m?.first_snapshot ? `; logging since ${shortDate(m.first_snapshot)}` : ""}.</span></div>`;
  }
  const t = k.by_threshold[String(state.minEdge)] || { n: 0 };
  const few = (t.n || 0) < MIN_PICKS;
  const pts = (x) => (x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(1)} pts`);
  const tiles = `
    <div class="tiles ${few ? "dim" : ""}">
      <div class="tile"><div class="label">Picks graded</div><div class="value">${(t.n || 0).toLocaleString()}</div><div class="sub">${t.n_void || 0} void · ${t.n_pending || 0} pending</div></div>
      <div class="tile"><div class="label">Return per $1</div><div class="value ${t.roi > 0 && !few ? "pos-text" : ""}">${signedPct(t.roi, 1)}</div><div class="sub">90% range ${signedPct(t.roi_lo, 0)} to ${signedPct(t.roi_hi, 0)}</div></div>
      <div class="tile"><div class="label">Win rate</div><div class="value">${pct(t.win, 1)}</div><div class="sub">break-even ${pct(t.breakeven, 1)} · model said ${pct(t.expected, 1)}</div></div>
      <div class="tile"><div class="label">Closing line value</div><div class="value">${pts(t.clv)}</div><div class="sub">${t.clv_pos == null ? "no closing prices yet" : `${pct(t.clv_pos)} beat the close`}</div></div>
    </div>`;
  const ll = k.logloss_model != null
    ? `<p class="note">Model log loss <b class="${k.logloss_model < k.logloss_book ? "pos-text" : ""}">${k.logloss_model.toFixed(4)}</b> vs DraftKings ${k.logloss_book.toFixed(4)} on ${k.n_lines.toLocaleString()} closing lines. Lower is better; if the model isn't lower, it knows nothing the price doesn't.</p>`
    : "";
  const sh = k.shadow;
  const shadowLine = sh
    ? `<p class="note">Shadow (${esc(MODEL_NAMES[sh.name] || sh.name)}): log loss ${sh.logloss_shadow.toFixed(4)} vs active ${sh.logloss_model.toFixed(4)} and DraftKings ${sh.logloss_book.toFixed(4)} on the same ${sh.n_lines.toLocaleString()} closing lines.</p>`
    : "";
  const edgeRows = (k.by_edge || []).map((b) => `<tr><td>${pct(b.lo)}${b.hi ? `–${pct(b.hi)}` : "+"}</td><td>${b.n || 0}</td><td>${pct(b.win)}</td><td>${signedPct(b.roi, 1)}</td><td>${b.clv == null ? "–" : pts(b.clv)}</td></tr>`).join("");
  const lineup = k.by_lineup
    ? `<p class="note">Hitters at a 2%+ edge: confirmed lineups ${signedPct(k.by_lineup.confirmed?.roi, 1)} on ${k.by_lineup.confirmed?.n || 0}, projected ${signedPct(k.by_lineup.projected?.roi, 1)} on ${k.by_lineup.projected?.n || 0}.</p>`
    : "";
  return `${head}
    ${edgeControl()}
    ${tiles}
    ${few ? `<p class="note">Too few picks to judge yet (${t.n || 0} of ${MIN_PICKS}). Until then these numbers are mostly luck.</p>` : ""}
    ${ll}
    ${shadowLine}
    <div class="card"><table><thead><tr><th>Edge</th><th>Picks</th><th>Win</th><th>Return</th><th>CLV</th></tr></thead><tbody>${edgeRows}</tbody></table></div>
    ${lineup}
    <p class="note">A pick is the first DraftKings price at or above the edge, $1 flat, graded after the game. Closing line value: how far DraftKings' own chance moved toward the pick by first pitch, in percentage points; beating the close consistently is the surest sign of real value.</p>`;
}

function viewRecord() {
  const rec = state.data?.record?.[state.recKind];
  const toggle = `<div class="segmented">${[["batter", "Hitters · H+R+RBI"], ["pitcher", "Pitchers · K"]]
    .map(([k, l]) => `<button data-rec="${k}" class="${k === state.recKind ? "on" : ""}">${l}</button>`).join("")}</div>`;
  if (!rec || !rec.n) return `${toggle}${marketCard()}<div class="empty">Not enough finished games yet to grade the model.</div>`;
  const unit = state.recKind === "batter" ? "H+R+RBI" : "K";
  // Log loss of the over/under chances: how much better than the player's season average.
  const llGain = rec.logloss_baseline ? (rec.logloss_baseline - rec.logloss_model) / rec.logloss_baseline : null;
  const tp = rec.top_picks || {};
  const byLine = rec.calibration_by_line || {};
  const calKey = state.calLine != null && byLine[String(state.calLine)] ? String(state.calLine) : null;
  const calSwitch = Object.keys(byLine).length
    ? `<div class="segmented">${[[null, "All lines"], ...Object.keys(byLine).map((l) => [l, `O ${l}`])]
        .map(([v, l]) => `<button data-cal-line="${v ?? ""}" class="${(v ?? null) === calKey ? "on" : ""}">${l}</button>`).join("")}</div>`
    : "";
  return `${toggle}
    ${modelNote(state.recKind)}
    ${marketCard()}
    <h2 class="section-title">Last ${Math.round((Date.parse(rec.test_to) - Date.parse(rec.test_from)) / 864e5) + 1} days, not used in training</h2>
    <div class="tiles">
      <div class="tile"><div class="label">Over/under chances</div><div class="value">${llGain == null ? "–" : `${llGain >= 0 ? "+" : "−"}${Math.abs(llGain * 100).toFixed(1)}%`}</div><div class="sub">${llGain == null ? "" : llGain >= 0 ? "better than season average" : "worse than season average"}</div></div>
      <div class="tile"><div class="label">Average miss</div><div class="value">${fix(rec.mae_model, 2)}</div><div class="sub">${unit} per game · season avg ${fix(rec.mae_baseline, 2)}</div></div>
      <div class="tile"><div class="label">Daily top-10 over ${tp.line}</div><div class="value">${pct(tp.actual)}</div><div class="sub">hit · model said ${pct(tp.predicted)}</div></div>
      <div class="tile"><div class="label">${state.recKind === "batter" ? "Hitter" : "Starter"} games graded</div><div class="value">${rec.n.toLocaleString()}</div><div class="sub">${shortDate(rec.test_from)} – ${shortDate(rec.test_to)}</div></div>
    </div>
    <h2 class="section-title">Are the chances honest?</h2>
    ${calSwitch}
    <div class="card">${calibrationChart(calKey ? byLine[calKey] : rec.calibration || [])}</div>
    <p class="note">"Season average" is the player's own ${unit} per game this season, the obvious guess without a model. "Over/under chances" scores the chance of going over every line (log loss); it's what matters for betting, and it can improve even when the average miss barely moves. The model retrains every run on all finished games; this check holds out the most recent ${Math.round((Date.parse(rec.test_to) - Date.parse(rec.test_from)) / 864e5) + 1} days.</p>`;
}

// ---------- paper portfolio ----------
const decimalOdds = (a) => (a > 0 ? 1 + a / 100 : 1 + 100 / Math.abs(a));
const money = (x, d = 2) => (x == null ? "–" : `${x >= 0 ? "+" : "−"}$${Math.abs(x).toFixed(d)}`);

function profitChart(curve) {
  if (!curve || curve.length < 2) return "";
  const W = 320, H = 170, L = 40, R = 8, T = 10, B = 22;
  const ys = curve.map((c) => c.cum);
  const lo = Math.min(0, ...ys), hi = Math.max(0, ...ys);
  const span = hi - lo || 1;
  const sx = (i) => L + (i / (curve.length - 1)) * (W - L - R);
  const sy = (v) => T + (1 - (v - lo) / span) * (H - T - B);
  const ticks = [lo, (lo + hi) / 2, hi];
  const grid = ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L - 4}" y="${sy(v) + 3}" text-anchor="end">${money(v, 0)}</text>`).join("");
  const path = curve.map((c, i) => `${i ? "L" : "M"}${sx(i)},${sy(c.cum)}`).join(" ");
  const hits = curve.map((c, i) => `<circle class="hit" cx="${sx(i)}" cy="${sy(c.cum)}" r="12" fill="transparent" data-tip="${esc(`${shortDate(c.date)}: ${money(c.profit)} · total ${money(c.cum)}`)}" data-x="${(sx(i) / W) * 100}"/>`).join("");
  const last = curve.length - 1;
  return `
    <div class="chart">
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cumulative paper profit by day">
        <g class="grid">${grid}</g>
        <line class="zero" x1="${L}" x2="${W - R}" y1="${sy(0)}" y2="${sy(0)}"/>
        <path class="line" d="${path}"/>
        <circle class="dot" cx="${sx(last)}" cy="${sy(curve[last].cum)}" r="4"/>
        <text class="xlab" x="${L}" y="${H - 6}">${shortDate(curve[0].date)}</text>
        <text class="xlab" x="${W - R}" y="${H - 6}" text-anchor="end">${shortDate(curve[last].date)}</text>
        ${hits}
      </svg>
      <div class="tooltip" hidden></div>
    </div>`;
}

function tradeRow(t) {
  const what = `${t.side === "over" ? "Over" : "Under"} ${t.line} ${t.kind === "pitcher" ? "K" : "H+R+RBI"}`;
  const placed = new Date(t.fetched_at);
  const when = `${placed.toLocaleDateString([], { month: "short", day: "numeric" })} ${placed.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
  const right = t.result === "open"
    ? `<b>$${t.stake.toFixed(0)}</b><small>to win $${(t.stake * (decimalOdds(t.price) - 1)).toFixed(2)}</small>`
    : `<b class="${t.result === "won" ? "pos-text" : t.result === "lost" ? "neg-text" : ""}">${t.result === "void" ? "void" : money(t.profit)}</b><small>${t.result === "void" ? "refunded" : t.result}${t.actual != null ? ` · actual ${t.actual}` : ""}</small>`;
  return `
    <div class="row-btn">
      <span class="who"><b>${esc(t.player_name || "")}</b>
        <span class="meta">${what} · ${american(t.price)} · edge ${signedPct(t.ev)}</span>
        <span class="meta">Placed ${when}${t.clv != null ? ` · CLV ${t.clv >= 0 ? "+" : "−"}${Math.abs(t.clv * 100).toFixed(1)} pts` : ""}</span></span>
      <span class="vals">${right}</span>
    </div>`;
}

function viewPortfolio() {
  const pp = state.data?.paper;
  const rules = `<p class="note">Paper trading: $${pp?.stake ?? 10} on every DraftKings price at or above a ${pct(pp?.threshold ?? 0.12)} model edge, at the first price that clears it (one trade per player and prop per game). Hitters from projected lineups are skipped. Trades are recorded automatically every run and settled from the box score; void = refunded.</p>`;
  if (!pp || !pp.summary?.n) {
    return `<h2 class="section-title">Paper Portfolio</h2><div class="empty">No paper trades yet.<br><span class="muted">One is recorded the first time a DraftKings price shows a ${pct(pp?.threshold ?? 0.12)}+ edge.</span></div>${rules}`;
  }
  const s = pp.summary;
  const open = pp.trades.filter((t) => t.result === "open");
  const settled = pp.trades.filter((t) => t.result !== "open");
  return `
    <h2 class="section-title">Paper Portfolio</h2>
    <div class="tiles">
      <div class="tile"><div class="label">Profit</div><div class="value ${s.profit > 0 ? "pos-text" : s.profit < 0 ? "neg-text" : ""}">${money(s.profit)}</div><div class="sub">on $${s.staked.toFixed(0)} settled</div></div>
      <div class="tile"><div class="label">Return</div><div class="value">${signedPct(s.roi, 1)}</div><div class="sub">avg edge ${pct(s.avg_edge, 0)} at placement</div></div>
      <div class="tile"><div class="label">Record</div><div class="value">${s.won}–${s.lost}</div><div class="sub">${s.void} void${s.clv != null ? ` · CLV ${s.clv >= 0 ? "+" : "−"}${Math.abs(s.clv).toFixed(1)} pts` : ""}</div></div>
      <div class="tile"><div class="label">Open</div><div class="value">${s.open}</div><div class="sub">$${s.at_risk.toFixed(0)} at risk</div></div>
    </div>
    ${pp.curve?.length > 1 ? `<h2 class="section-title">Profit over time</h2><div class="card">${profitChart(pp.curve)}</div>` : ""}
    ${open.length ? `<h2 class="section-title">Open · ${open.length}</h2><div class="card list">${open.map(tradeRow).join("")}</div>` : ""}
    ${settled.length ? `<h2 class="section-title">Settled · ${settled.length}</h2><div class="card list">${settled.slice(0, 100).map(tradeRow).join("")}</div>` : ""}
    ${s.won + s.lost < 200 ? `<p class="note">Only ${s.won + s.lost} settled so far; until a few hundred, results are mostly luck.</p>` : ""}
    ${rules}`;
}

// ---------- shell ----------
const VIEWS = { games: viewGames, hitters: viewHitters, pitchers: viewPitchers, portfolio: viewPortfolio, record: viewRecord };

function render() {
  $("#view").innerHTML = VIEWS[state.tab]();
  document.querySelectorAll(".tabbar button").forEach((b) => {
    const on = b.dataset.tab === state.tab;
    b.classList.toggle("active", on);
    b.setAttribute("aria-current", on ? "page" : "false");
  });
  bindCharts($("#view"));
}
function setTab(tab) {
  state.tab = tab;
  state.query = "";
  render();
  window.scrollTo(0, 0);
  $("#view").focus({ preventScroll: true });
}
function setUpdated() {
  const d = state.data;
  const mins = Math.round((Date.now() - Date.parse(d.generated_at)) / 60000);
  const ago = mins < 1 ? "just now" : mins < 60 ? `${mins} min ago` : mins < 1440 ? `${Math.round(mins / 60)} h ago` : `${Math.round(mins / 1440)} days ago`;
  $("#updated").textContent = `Updated ${ago}`;
}

function openSheet(html) {
  $("#sheet-body").innerHTML = html;
  const sheet = $("#sheet");
  sheet.hidden = false;
  document.body.style.overflow = "hidden";
  bindCharts($("#sheet-body"));
  $(".sheet-close", sheet).focus();
}
function closeSheet() {
  $("#sheet").hidden = true;
  document.body.style.overflow = "";
  state.sheetPlayer = null;
  state.sheetLine = null;
}
function showPlayer(kind, gamePk, id) {
  const p = findPlayer(kind, gamePk, id);
  if (!p) return;
  state.sheetPlayer = { kind, gamePk, id };
  openSheet(playerSheet(kind, p));
}

document.addEventListener("click", (ev) => {
  const t = ev.target.closest("button, [data-close]");
  if (!t) return;
  if (t.dataset.tab) setTab(t.dataset.tab);
  else if (t.dataset.day) { state.day = Number(t.dataset.day); render(); }
  else if (t.dataset.batLine) { state.batLine = Number(t.dataset.batLine); store.set("batLine", state.batLine); render(); }
  else if (t.dataset.pitLine) { state.pitLine = Number(t.dataset.pitLine); store.set("pitLine", state.pitLine); render(); }
  else if (t.dataset.edge) { state.minEdge = Number(t.dataset.edge); store.set("minEdge", state.minEdge); render(); }
  else if (t.dataset.calLine !== undefined) { state.calLine = t.dataset.calLine === "" ? null : t.dataset.calLine; render(); }
  else if (t.dataset.rec) { state.recKind = t.dataset.rec; state.calLine = null; store.set("recKind", state.recKind); render(); }
  else if (t.dataset.sheetLine) {
    state.sheetLine = Number(t.dataset.sheetLine);
    const sp = state.sheetPlayer;
    if (sp) { const p = findPlayer(sp.kind, sp.gamePk, sp.id); $("#sheet-body").innerHTML = playerSheet(sp.kind, p); bindCharts($("#sheet-body")); }
  } else if (t.dataset.player) showPlayer(t.dataset.player, Number(t.dataset.game), Number(t.dataset.id));
  else if (t.dataset.game) {
    const g = slate().games.find((x) => x.game_pk === Number(t.dataset.game));
    if (g) openSheet(gameSheet(g));
  } else if (t.hasAttribute("data-close")) closeSheet();
  else if (t.id === "refresh") load(true);
});
document.addEventListener("input", (ev) => {
  if (ev.target.id !== "search") return;
  state.query = ev.target.value;
  const pos = ev.target.selectionStart;
  render();
  const s = $("#search");
  s.focus();
  s.setSelectionRange(pos, pos);
});
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeSheet(); });

// ---------- load ----------
async function load(force = false) {
  const btn = $("#refresh");
  btn.classList.add("spin");
  try {
    const res = await fetch("data.json", { cache: force ? "reload" : "no-cache" });
    if (!res.ok) throw new Error(res.statusText);
    state.data = await res.json();
    state.day = Math.min(state.day, Math.max(slates().length - 1, 0));
    setUpdated();
    render();
  } catch (err) {
    if (!state.data) $("#view").innerHTML = `<div class="empty">Couldn't load predictions.<br><span class="muted">${esc(err.message || err)}</span></div>`;
  } finally {
    btn.classList.remove("spin");
  }
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}
load();
