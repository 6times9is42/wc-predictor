const state = {
  simulation: [],
  teams: [],
  groups: {},
  matchups: [],
  matchupMap: new Map(),
  teamMap: new Map(),
  simMap: new Map(),
  explanations: {},
  featureImportance: [],
  meta: {}
};

const titles = {
  tournament: "Tournament probabilities",
  team: "Team deep dive",
  matchup: "Head-to-head predictor",
  group: "Group simulator",
  explain: "Model explainability"
};

const colors = {
  home: "#0f766e",
  draw: "#b7791f",
  away: "#be123c"
};

function byId(id) {
  return document.getElementById(id);
}

function pct(value, digits = 1) {
  return `${((Number(value) || 0) * 100).toFixed(digits)}%`;
}

function fixed(value, digits = 2) {
  return (Number(value) || 0).toFixed(digits);
}

function intNumber(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(Number(value) || 0);
}

function money(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "EUR",
    notation: "compact",
    maximumFractionDigits: 1
  }).format(Number(value) || 0);
}

function featureLabel(name) {
  return name.replaceAll("_", " ").replace(/\b\w/g, char => char.toUpperCase());
}

function matchupKey(home, away) {
  return `${home}|||${away}`;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

async function loadData() {
  const [
    simulation,
    teams,
    groups,
    matchups,
    explanations,
    featureImportance,
    meta
  ] = await Promise.all([
    loadJson("/data/simulation_results.json"),
    loadJson("/data/teams.json"),
    loadJson("/data/groups.json"),
    loadJson("/data/match_predictions.json"),
    loadJson("/data/explanations.json"),
    loadJson("/data/feature_importance.json"),
    loadJson("/data/meta.json")
  ]);

  state.simulation = simulation;
  state.teams = teams.slice().sort((a, b) => a.team_name.localeCompare(b.team_name));
  state.groups = groups;
  state.matchups = matchups;
  state.explanations = explanations;
  state.featureImportance = featureImportance;
  state.meta = meta;
  state.teamMap = new Map(state.teams.map(team => [team.team_name, team]));
  state.simMap = new Map(state.simulation.map(row => [row.team, row]));
  state.matchupMap = new Map(state.matchups.map(row => [matchupKey(row.home_team, row.away_team), row]));

  populateControls();
  renderAll();
  byId("status-pill").textContent = "Ready";
  byId("data-stamp").textContent = `${meta.teams} teams · ${intNumber(meta.simulations)} simulations`;
}

function populateTeamSelect(select, selectedTeam) {
  select.innerHTML = state.teams
    .map(team => `<option value="${team.team_name}">${team.team_name}</option>`)
    .join("");
  if (selectedTeam) {
    select.value = selectedTeam;
  }
}

function populateControls() {
  populateTeamSelect(byId("team-select"), "France");
  populateTeamSelect(byId("match-home"), "Spain");
  populateTeamSelect(byId("explain-team-select"), "Spain");
  refreshAwayOptions("France");

  byId("group-select").innerHTML = Object.keys(state.groups)
    .map(group => `<option value="${group}">${group}</option>`)
    .join("");
  byId("group-select").value = "Group F";
}

function renderMetricCards(container, cards) {
  container.innerHTML = cards.map(card => `
    <article class="metric-card">
      <div class="metric-label">${card.label}</div>
      <div class="metric-value">${card.value}</div>
      <div class="metric-note">${card.note || ""}</div>
    </article>
  `).join("");
}

function renderBarStack(container, rows, options = {}) {
  const max = options.max ?? Math.max(...rows.map(row => Number(row.value) || 0), 0.01);
  const color = options.color || colors.home;
  container.innerHTML = rows.map(row => {
    const width = Math.max(0, Math.min(100, ((Number(row.value) || 0) / max) * 100));
    const value = options.rawValue ? row.display : pct(row.value, options.digits ?? 1);
    return `
      <div class="bar-row">
        <div class="bar-label" title="${row.label}">${row.label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${row.color || color}"></div></div>
        <div class="bar-value">${value}</div>
      </div>
    `;
  }).join("");
}

function renderTournament() {
  const top = state.simulation[0];
  const second = state.simulation[1];
  const averageR32 = state.simulation.reduce((sum, row) => sum + Number(row.r32_probability || 0), 0) / state.simulation.length;
  const strongestSquad = state.teams.slice().sort((a, b) => b.squad_avg_rating - a.squad_avg_rating)[0];

  renderMetricCards(byId("overview-metrics"), [
    { label: "Favorite", value: top.team, note: `${pct(top.win_probability, 2)} title probability` },
    { label: "Challenger", value: second.team, note: `${pct(second.win_probability, 2)} title probability` },
    { label: "Teams", value: state.teams.length, note: "Qualified field" },
    { label: "Avg R32", value: pct(averageR32, 1), note: `${strongestSquad.team_name} leads squad rating` }
  ]);

  renderBarStack(
    byId("win-bars"),
    state.simulation.slice(0, 16).map(row => ({ label: row.team, value: row.win_probability })),
    { max: state.simulation[0].win_probability, digits: 2 }
  );

  byId("ranking-table").innerHTML = state.simulation.map(row => `
    <tr>
      <td>${row.rank}</td>
      <td>${row.team}</td>
      <td>${pct(row.win_probability, 2)}</td>
      <td>${pct(Number(row.win_probability) + Number(row.runner_up_probability), 2)}</td>
      <td>${pct(row.semifinal_probability, 1)}</td>
      <td>${pct(row.r32_probability, 1)}</td>
    </tr>
  `).join("");
}

function renderTeamView() {
  const teamName = byId("team-select").value;
  const team = state.teamMap.get(teamName);
  const sim = state.simMap.get(teamName);
  const explanation = state.explanations[teamName];

  renderMetricCards(byId("team-metrics"), [
    { label: "Win Probability", value: pct(sim.win_probability, 2), note: `Rank ${sim.rank}` },
    { label: "ELO", value: intNumber(team.team_elo), note: `FIFA rank ${team.fifa_ranking || "n/a"}` },
    { label: "Squad Rating", value: fixed(team.squad_avg_rating, 2), note: `${money(team.market_value_top11_eur)} top XI` },
    { label: "Recent Win Rate", value: pct(team.weighted_win_rate, 1), note: `${team.confederation_name || "Unknown"} · ${team.wc_appearances_total || 0} WC apps` }
  ]);

  byId("team-rank-label").textContent = `Rank ${sim.rank}`;
  renderBarStack(byId("team-path-bars"), [
    { label: "Win", value: sim.win_probability },
    { label: "Final", value: Number(sim.win_probability) + Number(sim.runner_up_probability) },
    { label: "Semifinal", value: sim.semifinal_probability },
    { label: "Quarterfinal", value: sim.quarterfinal_probability },
    { label: "Round of 16", value: sim.r16_probability },
    { label: "Round of 32", value: sim.r32_probability }
  ], { max: 1 });

  renderExplanation(explanation, "team");
}

function renderExplanation(explanation, prefix) {
  const positive = byId(`${prefix}-positive-drivers`);
  const negative = byId(`${prefix}-negative-drivers`);
  const opponent = byId(`${prefix}-shap-opponent`);
  if (opponent) {
    opponent.textContent = explanation ? `vs ${explanation.opponent_name}` : "";
  }

  renderDrivers(positive, explanation?.positive_features || {}, "positive");
  renderDrivers(negative, explanation?.negative_features || {}, "negative");
}

function renderDrivers(container, features, type) {
  const entries = Object.entries(features || {});
  if (!entries.length) {
    container.innerHTML = `<div class="empty-note">No material ${type} drivers</div>`;
    return;
  }

  container.innerHTML = entries.map(([feature, value]) => `
    <div class="driver-item ${type}">
      <span>${featureLabel(feature)}</span>
      <strong>${Number(value).toFixed(3)}</strong>
    </div>
  `).join("");
}

function refreshAwayOptions(fallback) {
  const home = byId("match-home").value;
  const awaySelect = byId("match-away");
  const current = awaySelect.value;
  const options = state.teams.filter(team => team.team_name !== home);
  awaySelect.innerHTML = options
    .map(team => `<option value="${team.team_name}">${team.team_name}</option>`)
    .join("");

  if (current && current !== home && options.some(team => team.team_name === current)) {
    awaySelect.value = current;
  } else if (fallback && fallback !== home && options.some(team => team.team_name === fallback)) {
    awaySelect.value = fallback;
  } else {
    awaySelect.value = options[0]?.team_name || "";
  }
}

function renderMatchup() {
  const home = byId("match-home").value;
  const away = byId("match-away").value;
  const pred = state.matchupMap.get(matchupKey(home, away));

  if (!pred) {
    byId("match-factor").textContent = "Prediction unavailable";
    return;
  }

  const homeProb = Number(pred.home_win_prob);
  const drawProb = Number(pred.draw_prob);
  const awayProb = Number(pred.away_win_prob);
  const homeDeg = homeProb * 360;
  const drawDeg = drawProb * 360;

  byId("match-donut").style.background = `conic-gradient(${colors.home} 0 ${homeDeg}deg, ${colors.draw} ${homeDeg}deg ${homeDeg + drawDeg}deg, ${colors.away} ${homeDeg + drawDeg}deg 360deg)`;
  byId("match-confidence").textContent = pred.confidence || "precomputed";
  byId("match-factor").textContent = matchupFactor(home, away, homeProb, awayProb);

  byId("match-legend").innerHTML = [
    { label: `${home} win`, value: homeProb, color: colors.home },
    { label: "Draw", value: drawProb, color: colors.draw },
    { label: `${away} win`, value: awayProb, color: colors.away }
  ].map(item => `
    <div class="legend-item">
      <div class="legend-left"><span class="legend-color" style="background:${item.color}"></span><span>${item.label}</span></div>
      <strong>${pct(item.value, 1)}</strong>
    </div>
  `).join("");

  renderMetricCards(byId("match-metrics"), [
    { label: `${home} xG`, value: fixed(pred.expected_home_goals, 2), note: "Expected goals" },
    { label: `${away} xG`, value: fixed(pred.expected_away_goals, 2), note: "Expected goals" }
  ]);

  renderBarStack(byId("match-prob-bars"), [
    { label: `${home} win`, value: homeProb, color: colors.home },
    { label: "Draw", value: drawProb, color: colors.draw },
    { label: `${away} win`, value: awayProb, color: colors.away }
  ], { max: 1 });
}

function matchupFactor(home, away, homeProb, awayProb) {
  if (homeProb > awayProb + 0.1) return `${home} favored`;
  if (awayProb > homeProb + 0.1) return `${away} favored`;
  return "Even matchup";
}

function poisson(lambda) {
  const safeLambda = Math.max(0.05, Number(lambda) || 1);
  const limit = Math.exp(-safeLambda);
  let k = 0;
  let probability = 1;
  do {
    k += 1;
    probability *= Math.random();
  } while (probability > limit);
  return k - 1;
}

function simulateMatch(teamA, teamB) {
  const pred = state.matchupMap.get(matchupKey(teamA, teamB)) || {
    home_win_prob: 0.33,
    draw_prob: 0.34,
    away_win_prob: 0.33,
    expected_home_goals: 1,
    expected_away_goals: 1
  };
  const total = Number(pred.home_win_prob) + Number(pred.draw_prob) + Number(pred.away_win_prob);
  const homeProb = Number(pred.home_win_prob) / total;
  const drawProb = Number(pred.draw_prob) / total;
  const roll = Math.random();
  let outcome = "A";
  if (roll < homeProb) outcome = "H";
  else if (roll < homeProb + drawProb) outcome = "D";

  let homeGoals = poisson(pred.expected_home_goals);
  let awayGoals = poisson(pred.expected_away_goals);
  if (outcome === "H" && homeGoals <= awayGoals) homeGoals = awayGoals + 1;
  if (outcome === "A" && awayGoals <= homeGoals) awayGoals = homeGoals + 1;
  if (outcome === "D") {
    const level = Math.floor((homeGoals + awayGoals) / 2);
    homeGoals = level;
    awayGoals = level;
  }

  return { outcome, homeGoals, awayGoals };
}

function simulateGroupStage() {
  const standings = {};
  for (const [groupName, teams] of Object.entries(state.groups)) {
    const table = new Map(teams.map(team => [team, { pts: 0, gd: 0, gf: 0 }]));
    for (let i = 0; i < teams.length; i += 1) {
      for (let j = i + 1; j < teams.length; j += 1) {
        const teamA = teams[i];
        const teamB = teams[j];
        const result = simulateMatch(teamA, teamB);
        const rowA = table.get(teamA);
        const rowB = table.get(teamB);
        rowA.gf += result.homeGoals;
        rowB.gf += result.awayGoals;
        rowA.gd += result.homeGoals - result.awayGoals;
        rowB.gd += result.awayGoals - result.homeGoals;
        if (result.outcome === "H") rowA.pts += 3;
        else if (result.outcome === "A") rowB.pts += 3;
        else {
          rowA.pts += 1;
          rowB.pts += 1;
        }
      }
    }

    standings[groupName] = Array.from(table.entries()).sort((a, b) => {
      const statsA = a[1];
      const statsB = b[1];
      return (
        statsB.pts - statsA.pts ||
        statsB.gd - statsA.gd ||
        statsB.gf - statsA.gf ||
        Math.random() - 0.5
      );
    });
  }
  return standings;
}

function thirdPlaceQualifiers(standings) {
  return Object.entries(standings)
    .map(([group, rows]) => ({ team: rows[2][0], group, stats: rows[2][1] }))
    .sort((a, b) => (
      b.stats.pts - a.stats.pts ||
      b.stats.gd - a.stats.gd ||
      b.stats.gf - a.stats.gf ||
      Math.random() - 0.5
    ))
    .slice(0, 8);
}

function runGroupSimulation() {
  const groupName = byId("group-select").value;
  const teams = state.groups[groupName];
  const runs = 2500;
  const counts = Object.fromEntries(teams.map(team => [team, { advance: 0, topTwo: 0, third: 0, points: 0, rank: 0 }]));

  for (let run = 0; run < runs; run += 1) {
    const standings = simulateGroupStage();
    const qualifiedThirds = new Set(thirdPlaceQualifiers(standings).map(row => row.team));
    const groupRows = standings[groupName];

    groupRows.forEach(([team, stats], index) => {
      const rank = index + 1;
      counts[team].points += stats.pts;
      counts[team].rank += rank;
      if (rank <= 2) {
        counts[team].topTwo += 1;
        counts[team].advance += 1;
      } else if (rank === 3) {
        counts[team].third += 1;
        if (qualifiedThirds.has(team)) counts[team].advance += 1;
      }
    });
  }

  const rows = teams.map(team => ({
    team,
    averagePoints: counts[team].points / runs,
    averageRank: counts[team].rank / runs,
    qualification: counts[team].advance / runs,
    topTwo: counts[team].topTwo / runs,
    third: counts[team].third / runs
  })).sort((a, b) => a.averageRank - b.averageRank || b.averagePoints - a.averagePoints);

  byId("group-standings").innerHTML = rows.map(row => `
    <li>
      <div class="standings-main"><span>${row.team}</span><span>${fixed(row.averagePoints, 2)} pts</span></div>
      <div class="standings-sub">Avg rank ${fixed(row.averageRank, 2)} · third-place rate ${pct(row.third, 1)}</div>
    </li>
  `).join("");

  renderBarStack(
    byId("group-qual-bars"),
    rows.slice().sort((a, b) => b.qualification - a.qualification).map(row => ({ label: row.team, value: row.qualification })),
    { max: 1 }
  );
  renderBarStack(
    byId("group-top-two-bars"),
    rows.slice().sort((a, b) => b.topTwo - a.topTwo).map(row => ({ label: row.team, value: row.topTwo })),
    { max: 1 }
  );
}

function renderExplainability() {
  renderBarStack(
    byId("importance-bars"),
    state.featureImportance.slice(0, 15).map(row => ({
      label: featureLabel(row.feature),
      value: row.importance,
      display: fixed(row.importance, 3)
    })),
    { rawValue: true }
  );

  const teamName = byId("explain-team-select").value;
  const explanation = state.explanations[teamName];
  byId("explain-shap-opponent").textContent = explanation ? `vs ${explanation.opponent_name}` : "";
  renderExplanation(explanation, "explain");
}

function renderAll() {
  renderTournament();
  renderTeamView();
  renderMatchup();
  runGroupSimulation();
  renderExplainability();
}

function setView(viewName) {
  document.querySelectorAll(".nav-button").forEach(button => {
    button.classList.toggle("is-active", button.dataset.view === viewName);
  });
  document.querySelectorAll(".view").forEach(view => {
    view.classList.toggle("is-active", view.id === `view-${viewName}`);
  });
  byId("view-title").textContent = titles[viewName];
}

function bindEvents() {
  document.querySelectorAll(".nav-button").forEach(button => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });

  byId("team-select").addEventListener("change", renderTeamView);
  byId("explain-team-select").addEventListener("change", renderExplainability);
  byId("match-home").addEventListener("change", () => {
    refreshAwayOptions();
    renderMatchup();
  });
  byId("match-away").addEventListener("change", renderMatchup);
  byId("simulate-group-button").addEventListener("click", runGroupSimulation);
  byId("group-select").addEventListener("change", runGroupSimulation);
}

bindEvents();
loadData().catch(error => {
  byId("status-pill").textContent = "Data error";
  document.querySelector(".main-panel").insertAdjacentHTML(
    "afterbegin",
    `<div class="error-box">${error.message}</div>`
  );
});
