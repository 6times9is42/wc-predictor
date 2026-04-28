import argparse
import json
import os
import re
import shutil
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


BASE_URL = "https://www.transfermarkt.com"
TM_API_BASE_URL = "https://tmapi-alpha.transfermarkt.technology"
TM_API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://www.transfermarkt.co.uk",
    "referer": "https://www.transfermarkt.co.uk",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
}

TEAM_ALIASES = {
    "USA": ["United States", "USA", "United States of America"],
    "Czechia": ["Czech Republic", "Czechia"],
    "Türkiye": ["Turkey", "Turkiye", "Türkiye"],
    "Ivory Coast": ["Cote d'Ivoire", "Ivory Coast"],
    "Curaçao": ["Curacao", "Curaçao"],
    "South Korea": ["Korea, South", "South Korea", "Korea Republic"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia and Herzegovina"],
    "Cape Verde": ["Cape Verde", "Cabo Verde"],
    "England": ["England"],
}

POSITION_MAP = {
    "goalkeeper": "GK",
    "keeper": "GK",
    "centre-back": "DEF",
    "center-back": "DEF",
    "left-back": "DEF",
    "right-back": "DEF",
    "defender": "DEF",
    "back": "DEF",
    "sweeper": "DEF",
    "defensive midfield": "MID",
    "central midfield": "MID",
    "attacking midfield": "MID",
    "midfield": "MID",
    "midfielder": "MID",
    "left winger": "FWD",
    "right winger": "FWD",
    "winger": "FWD",
    "centre-forward": "FWD",
    "center-forward": "FWD",
    "second striker": "FWD",
    "forward": "FWD",
    "striker": "FWD",
    "attack": "FWD",
    "gk": "GK",
    "cb": "DEF",
    "lb": "DEF",
    "rb": "DEF",
    "dm": "MID",
    "cm": "MID",
    "am": "MID",
    "lm": "MID",
    "rm": "MID",
    "lw": "FWD",
    "rw": "FWD",
    "cf": "FWD",
    "ss": "FWD",
}

CONFEDERATION_MAP = {
    "uefa": "UEFA",
    "europe": "UEFA",
    "conmebol": "CONMEBOL",
    "south america": "CONMEBOL",
    "caf": "CAF",
    "africa": "CAF",
    "afc": "AFC",
    "asia": "AFC",
    "concacaf": "CONCACAF",
    "north america": "CONCACAF",
    "ofc": "OFC",
    "oceania": "OFC",
}

CLUB_COMPETITION_FALLBACKS = {
    "Violette Athletic Club": ("Ligue Haïtienne", 1),
    "Paris Saint-Germain Espoirs": ("Challenge Espoirs", 4),
    "Atlético Levante UD": ("Tercera Federación - Grupo 6", 4),
}

NATIONAL_PLAYER_STATES = {"CURRENT_NATIONAL_PLAYER", "RECENT_NATIONAL_PLAYER"}

SUPPLEMENTAL_TRANSFERMARKT_PLAYERS = {
    "Portugal": [
        {
            "player_id": "8198",
            "player_name": "Cristiano Ronaldo",
            "position": "FWD",
            "position_detail": "Centre-Forward",
            "jersey_number": 7,
        }
    ],
}

OUTPUT_COLUMNS = [
    "player_name",
    "team_name",
    "position",
    "position_detail",
    "age",
    "caps",
    "club_name",
    "club_league",
    "club_league_tier",
    "market_value_eur",
    "injured",
    "suspended",
    "jersey_number",
    "transfermarkt_player_id",
    "transfermarkt_player_url",
    "transfermarkt_team_url",
    "team_confederation",
    "team_fifa_ranking",
    "team_market_value_eur",
]


@dataclass
class SearchCandidate:
    name: str
    url: str
    squad_url: str
    country: str
    competition: str
    squad_size: int
    market_value_eur: int
    score: int = 0


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def _scraper_config() -> dict:
    config = load_config()
    return config.get("scraping", {}).get("transfermarkt", {})


def _absolute_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("http"):
        return href
    return urllib.parse.urljoin(BASE_URL, href)


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def team_search_terms(team_name: str) -> list[str]:
    aliases = TEAM_ALIASES.get(team_name, [team_name])
    terms = []
    for term in [team_name, *aliases]:
        if term and term not in terms:
            terms.append(term)
    return terms


def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US,en")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver_path = os.getenv("CHROMEDRIVER_PATH") or shutil.which("chromedriver")
    service = Service(driver_path) if driver_path else Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def fetch_soup(driver, url: str, wait_seconds: float | None = None) -> BeautifulSoup:
    wait = _scraper_config().get("request_delay_seconds", 2.0) if wait_seconds is None else wait_seconds
    driver.get(url)
    time.sleep(wait)
    return BeautifulSoup(driver.page_source, "html.parser")


def parse_market_value(mv_str: str) -> int:
    """Converts Transfermarkt market values such as EUR 50.00m or EUR 1.35bn."""
    if not mv_str:
        return 0

    value = (
        mv_str.replace("€", "")
        .replace("EUR", "")
        .replace("\xa0", "")
        .replace(",", "")
        .strip()
        .lower()
    )
    if value in {"", "-", "n/a"}:
        return 0

    match = re.search(r"(\d+(?:\.\d+)?)\s*(bn|m|k|th\.?|)?", value)
    if not match:
        return 0
    suffix = match.group(2) or ""
    multiplier = {
        "bn": 1_000_000_000,
        "m": 1_000_000,
        "k": 1_000,
        "th.": 1_000,
        "th": 1_000,
    }.get(suffix, 1)
    return int(float(match.group(1)) * multiplier)


def parse_league_tier(value: str) -> int:
    value = (value or "").lower()
    if "first" in value or value == "1":
        return 1
    if "second" in value or value == "2":
        return 2
    if "third" in value or value == "3":
        return 3
    if "fourth" in value or value == "4":
        return 4
    return 0


def infer_competition_tier(competition_id: str | None, competition_name: str | None) -> int:
    """Infer league tier from Transfermarkt competition ids/names.

    The profile HTML exposes "League level", but the JSON API returns the club's
    primary competition. Transfermarkt competition ids/names are regular enough
    to infer the domestic pyramid level for rating purposes.
    """
    if not competition_id and not competition_name:
        return 0

    comp_id = str(competition_id or "").upper()
    name = str(competition_name or "").lower()

    fourth_tier_markers = [
        "4.",
        "fourth",
        "league two",
        "national 2",
        "national 3",
        "regionalliga",
        "segunda federación",
        "serie d",
        "liga 4",
    ]
    third_tier_markers = [
        "3.",
        "third",
        "league one",
        "championnat national",
        "primera federación",
        "serie c",
        "liga 3",
        "j3 league",
    ]
    second_tier_markers = [
        "2.",
        "second",
        "championship",
        "liga 2",
        "ligue 2",
        "laliga2",
        "serie b",
        "segunda división",
        "segunda division",
        "eerste divisie",
        "j2 league",
        "k league 2",
        "usl championship",
        "liga de expansión",
        "liga de expansion",
        "primera nacional",
    ]

    if any(marker in name for marker in fourth_tier_markers):
        return 4
    if any(marker in name for marker in third_tier_markers):
        return 3
    if any(marker in name for marker in second_tier_markers):
        return 2

    if re.search(r"(?:^|[A-Z])4[A-Z]*$", comp_id):
        return 4
    if re.search(r"(?:^|[A-Z])3[A-Z]*$", comp_id):
        return 3
    if re.search(r"(?:^|[A-Z])2[A-Z]*$", comp_id):
        return 2

    return 1


def map_position(position_text: str) -> str:
    normalized = _normalize_name(position_text).replace(" ", "-")
    text = (position_text or "").lower()
    for key, value in POSITION_MAP.items():
        if key in text or key == normalized:
            return value
    return "MID"


def _first_int(value: str, default: int = 0) -> int:
    match = re.search(r"\d+", value or "")
    return int(match.group(0)) if match else default


def parse_team_metadata(soup: BeautifulSoup) -> dict:
    text = soup.get_text(" ", strip=True)
    conf_match = re.search(r"Confederation:\s*([A-Za-z ]+?)(?:\s+FIFA|\s+World ranking|$)", text)
    ranking_match = re.search(r"FIFA World ranking:\s*Pos\s*(\d+)", text)
    conf_value = conf_match.group(1).strip() if conf_match else ""
    confederation = CONFEDERATION_MAP.get(_normalize_name(conf_value), conf_value.upper())

    return {
        "team_confederation": confederation,
        "team_fifa_ranking": int(ranking_match.group(1)) if ranking_match else 0,
        "team_market_value_eur": parse_market_value(
            soup.select_one(".data-header__market-value-wrapper").get_text(" ", strip=True)
            if soup.select_one(".data-header__market-value-wrapper")
            else ""
        ),
    }


def parse_search_candidates(soup: BeautifulSoup, team_name: str) -> list[SearchCandidate]:
    club_table = soup.select_one("#club-grid table.items")
    if not club_table:
        return []

    targets = {_normalize_name(term) for term in team_search_terms(team_name)}
    candidates = []

    for row in club_table.select("tbody tr"):
        link = row.select_one("td.hauptlink a[href*='/startseite/verein/']")
        if not link:
            link = row.select_one("a[href*='/startseite/verein/']")
        if not link:
            continue

        name = link.get_text(" ", strip=True) or link.get("title", "")
        href = link.get("href", "")
        squad_link = row.select_one("a[href*='/kader/verein/']")
        squad_href = squad_link.get("href", "") if squad_link else href.replace("/startseite/", "/kader/")
        detail_row = row.select_one("table.inline-table tr:nth-of-type(2) td")
        country_img = row.select_one("td.zentriert img.flaggenrahmen")
        market_cell = row.select_one("td.rechts")

        candidate = SearchCandidate(
            name=name,
            url=_absolute_url(href),
            squad_url=_absolute_url(squad_href),
            country=country_img.get("title", "") if country_img else "",
            competition=detail_row.get_text(" ", strip=True) if detail_row else "",
            squad_size=_first_int(squad_link.get_text(" ", strip=True) if squad_link else "0"),
            market_value_eur=parse_market_value(market_cell.get_text(" ", strip=True) if market_cell else ""),
        )
        candidate.score = score_search_candidate(candidate, targets)
        candidates.append(candidate)

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def score_search_candidate(candidate: SearchCandidate, targets: set[str]) -> int:
    name = _normalize_name(candidate.name)
    country = _normalize_name(candidate.country)
    competition = _normalize_name(candidate.competition)

    score = 0
    if name in targets:
        score += 120
    elif any(name.startswith(target) or target in name for target in targets):
        score += 55
    if country in targets:
        score += 25
    if "world cup" in competition:
        score += 15
    if candidate.squad_size > 0:
        score += 10
    if re.search(r"\bu\s*\d{2}\b", name) or " women " in f" {name} ":
        score -= 80
    if name.endswith(" b"):
        score -= 35
    return score


def discover_transfermarkt_squad_url(driver, team_name: str) -> str:
    for query in team_search_terms(team_name):
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche?query={encoded_query}"
        logger.debug(f"Searching Transfermarkt for {team_name}: {search_url}")
        soup = fetch_soup(driver, search_url)
        candidates = parse_search_candidates(soup, team_name)
        if candidates and candidates[0].score >= 50:
            logger.info(f"Matched {team_name} to Transfermarkt page {candidates[0].squad_url}")
            return candidates[0].squad_url

    return discover_transfermarkt_url_via_duckduckgo(driver, team_name)


def discover_transfermarkt_url_via_duckduckgo(driver, team_name: str) -> str:
    query = f"{team_search_terms(team_name)[0]} national football team squad site:transfermarkt.com"
    search_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    soup = fetch_soup(driver, search_url)

    for link in soup.select("a.result__a, a.result__snippet"):
        href = link.get("href", "")
        if not href:
            continue
        parsed = urllib.parse.urlparse(href)
        query_params = urllib.parse.parse_qs(parsed.query)
        extracted = query_params.get("uddg", [href])[0]
        if "transfermarkt" in extracted and "/verein/" in extracted:
            return extracted.replace("/startseite/", "/kader/")

    logger.warning(f"Could not find Transfermarkt squad URL for {team_name}")
    return ""


def parse_squad_html(html: str, team_name: str, team_url: str = "") -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")
    if not table:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    team_metadata = parse_team_metadata(soup)
    rows = table.select("tbody tr.odd, tbody tr.even")
    players = []

    for row in rows:
        direct_cells = row.find_all("td", recursive=False)
        if len(direct_cells) < 4:
            continue

        player_cell = direct_cells[1] if len(direct_cells) > 1 else direct_cells[0]
        player_link = player_cell.select_one("a[href*='/profil/spieler/']")
        if not player_link:
            continue

        player_name = player_link.get_text(" ", strip=True) or player_link.get("title", "")
        position_detail = ""
        inline_rows = player_cell.select("table.inline-table tr")
        if len(inline_rows) > 1:
            position_detail = inline_rows[1].get_text(" ", strip=True)
        if not position_detail:
            position_detail = direct_cells[0].get("title", "")

        club_img = direct_cells[3].select_one("img") if len(direct_cells) > 3 else None
        club_name = ""
        if club_img:
            club_name = club_img.get("alt") or club_img.get("title") or ""

        player_url = _absolute_url(player_link.get("href", ""))
        player_id_match = re.search(r"/spieler/(\d+)", player_url)

        players.append(
            {
                "player_name": player_name,
                "team_name": team_name,
                "position": map_position(position_detail),
                "position_detail": position_detail,
                "age": _first_int(direct_cells[2].get_text(" ", strip=True), default=0),
                "caps": 0,
                "club_name": club_name or "Unknown Club",
                "club_league": "Unknown",
                "club_league_tier": 0,
                "market_value_eur": parse_market_value(direct_cells[-1].get_text(" ", strip=True)),
                "injured": "injury" in _normalize_name(row.get_text(" ", strip=True)),
                "suspended": "suspension" in _normalize_name(row.get_text(" ", strip=True)),
                "jersey_number": _first_int(direct_cells[0].get_text(" ", strip=True), default=0),
                "transfermarkt_player_id": player_id_match.group(1) if player_id_match else "",
                "transfermarkt_player_url": player_url,
                "transfermarkt_team_url": team_url,
                **team_metadata,
            }
        )

    return pd.DataFrame(players, columns=OUTPUT_COLUMNS)


def parse_player_profile_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]
    joined = " ".join(lines)

    caps = 0
    caps_match = re.search(r"Caps/Goals:\s*(\d+)\s*/", joined)
    if caps_match:
        caps = int(caps_match.group(1))

    league = "Unknown"
    tier = 0
    for idx, line in enumerate(lines):
        if line == "League level:":
            if idx > 0:
                league = lines[idx - 1]
            if idx + 1 < len(lines):
                tier = parse_league_tier(lines[idx + 1])
            break

    return {
        "caps": caps,
        "club_league": league,
        "club_league_tier": tier,
    }


def _tm_api_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(TM_API_HEADERS)
    return session


def _tm_api_get(session: requests.Session, endpoint: str, delay_seconds: float = 0.0) -> dict:
    url = f"{TM_API_BASE_URL}{endpoint}"
    for attempt in range(3):
        try:
            response = session.get(url, timeout=20)
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            payload = response.json()
            if delay_seconds:
                time.sleep(delay_seconds)
            return payload.get("data", {}) if payload.get("success", True) else {}
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                logger.warning(f"Transfermarkt API request failed for {endpoint}: {exc}")
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


def _player_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def _int_or_none(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_transfermarkt_api_player(player_data: dict, national_career_data: dict | None = None) -> dict:
    current_club_id = ""
    for assignment in player_data.get("clubAssignments", []) or []:
        if assignment.get("type") == "current" and assignment.get("clubId"):
            current_club_id = str(assignment["clubId"])
            break

    caps = None
    if national_career_data:
        current_histories = [
            history
            for history in national_career_data.get("history", []) or []
            if history.get("careerState") in NATIONAL_PLAYER_STATES
        ]
        if current_histories:
            caps = max(_int_or_none(history.get("gamesPlayed")) or 0 for history in current_histories)

    market_value = (
        (player_data.get("marketValueDetails") or {})
        .get("current", {})
        .get("value")
    )

    return {
        "age": _int_or_none((player_data.get("lifeDates") or {}).get("age")),
        "caps": caps,
        "market_value_eur": _int_or_none(market_value),
        "transfermarkt_current_club_id": current_club_id,
    }


def enrich_player_profiles_from_api(squad_df: pd.DataFrame) -> pd.DataFrame:
    if squad_df.empty:
        return squad_df

    api_delay = _scraper_config().get("api_delay_seconds", 0.02)
    session = _tm_api_session()
    records = squad_df.to_dict("records")

    player_cache: dict[str, dict] = {}
    national_cache: dict[str, dict] = {}
    player_enrichment: dict[str, dict] = {}

    player_ids = sorted({_player_id(record.get("transfermarkt_player_id")) for record in records if _player_id(record.get("transfermarkt_player_id"))})
    logger.info(f"Enriching {len(player_ids)} unique players from Transfermarkt JSON API")
    for idx, player_id in enumerate(player_ids, start=1):
        logger.debug(f"Transfermarkt API player {idx}/{len(player_ids)}: {player_id}")
        player_cache[player_id] = _tm_api_get(session, f"/player/{player_id}", delay_seconds=api_delay)
        national_cache[player_id] = _tm_api_get(
            session,
            f"/player/{player_id}/national-career-history",
            delay_seconds=api_delay,
        )
        player_enrichment[player_id] = parse_transfermarkt_api_player(
            player_cache[player_id],
            national_cache[player_id],
        )

    club_ids = sorted({
        enrichment["transfermarkt_current_club_id"]
        for enrichment in player_enrichment.values()
        if enrichment.get("transfermarkt_current_club_id")
    })
    club_cache: dict[str, dict] = {}
    competition_ids: set[str] = set()
    logger.info(f"Fetching {len(club_ids)} current clubs from Transfermarkt JSON API")
    for idx, club_id in enumerate(club_ids, start=1):
        logger.debug(f"Transfermarkt API club {idx}/{len(club_ids)}: {club_id}")
        club_data = _tm_api_get(session, f"/club/{club_id}", delay_seconds=api_delay)
        club_cache[club_id] = club_data
        competition_id = (club_data.get("baseDetails") or {}).get("primaryCompetitionId")
        if competition_id:
            competition_ids.add(str(competition_id))

    competition_cache: dict[str, dict] = {}
    logger.info(f"Fetching {len(competition_ids)} club competitions from Transfermarkt JSON API")
    for idx, competition_id in enumerate(sorted(competition_ids), start=1):
        logger.debug(f"Transfermarkt API competition {idx}/{len(competition_ids)}: {competition_id}")
        competition_cache[competition_id] = _tm_api_get(
            session,
            f"/competition/{competition_id}",
            delay_seconds=api_delay,
        )

    for record in records:
        player_id = _player_id(record.get("transfermarkt_player_id"))
        enrichment = player_enrichment.get(player_id, {})
        if enrichment.get("age") is not None:
            record["age"] = enrichment["age"]
        if enrichment.get("caps") is not None:
            record["caps"] = enrichment["caps"]
        if enrichment.get("market_value_eur") is not None:
            record["market_value_eur"] = enrichment["market_value_eur"]

        club_id = enrichment.get("transfermarkt_current_club_id")
        club_data = club_cache.get(club_id, {}) if club_id else {}
        if club_data.get("name"):
            record["club_name"] = club_data["name"]

        competition_id = (club_data.get("baseDetails") or {}).get("primaryCompetitionId")
        competition_data = competition_cache.get(str(competition_id), {}) if competition_id else {}
        league_name = competition_data.get("name") or competition_data.get("shortName")
        if league_name:
            record["club_league"] = league_name
            record["club_league_tier"] = infer_competition_tier(str(competition_id), league_name)
        elif record.get("club_name") in CLUB_COMPETITION_FALLBACKS:
            league_name, tier = CLUB_COMPETITION_FALLBACKS[record["club_name"]]
            record["club_league"] = league_name
            record["club_league_tier"] = tier

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def add_supplemental_players_from_api(squad_df: pd.DataFrame) -> pd.DataFrame:
    if squad_df.empty or not _scraper_config().get("include_supplemental_players", True):
        return squad_df

    api_delay = _scraper_config().get("api_delay_seconds", 0.02)
    session = _tm_api_session()
    records = squad_df.to_dict("records")
    additions = []

    for team_name, player_specs in SUPPLEMENTAL_TRANSFERMARKT_PLAYERS.items():
        team_rows = squad_df[squad_df["team_name"] == team_name]
        if team_rows.empty:
            continue

        existing_player_ids = {
            _player_id(player_id)
            for player_id in team_rows["transfermarkt_player_id"].tolist()
        }
        template = team_rows.iloc[0].to_dict()

        for spec in player_specs:
            player_id = _player_id(spec["player_id"])
            if not player_id or player_id in existing_player_ids:
                continue

            logger.info(f"Adding supplemental Transfermarkt player {spec['player_name']} to {team_name}")
            player_data = _tm_api_get(session, f"/player/{player_id}", delay_seconds=api_delay)
            national_data = _tm_api_get(
                session,
                f"/player/{player_id}/national-career-history",
                delay_seconds=api_delay,
            )
            enrichment = parse_transfermarkt_api_player(player_data, national_data)

            club_id = enrichment.get("transfermarkt_current_club_id")
            club_data = _tm_api_get(session, f"/club/{club_id}", delay_seconds=api_delay) if club_id else {}
            competition_id = (club_data.get("baseDetails") or {}).get("primaryCompetitionId")
            competition_data = (
                _tm_api_get(session, f"/competition/{competition_id}", delay_seconds=api_delay)
                if competition_id
                else {}
            )
            league_name = competition_data.get("name") or competition_data.get("shortName")
            league_tier = infer_competition_tier(str(competition_id), league_name) if league_name else 0
            if not league_name and club_data.get("name") in CLUB_COMPETITION_FALLBACKS:
                league_name, league_tier = CLUB_COMPETITION_FALLBACKS[club_data["name"]]

            attrs = player_data.get("attributes") or {}
            api_position = attrs.get("position") or {}
            position_detail = spec.get("position_detail") or api_position.get("name") or "Forward"

            row = {column: template.get(column, "") for column in OUTPUT_COLUMNS}
            row.update(
                {
                    "player_name": spec.get("player_name") or player_data.get("shortName") or player_data.get("name"),
                    "team_name": team_name,
                    "position": spec.get("position") or map_position(position_detail),
                    "position_detail": position_detail,
                    "age": enrichment.get("age") or 0,
                    "caps": enrichment.get("caps") or 0,
                    "club_name": club_data.get("name") or "Unknown",
                    "club_league": league_name or "Unknown",
                    "club_league_tier": league_tier,
                    "market_value_eur": enrichment.get("market_value_eur") or 0,
                    "injured": False,
                    "suspended": False,
                    "jersey_number": spec.get("jersey_number", 0),
                    "transfermarkt_player_id": player_id,
                    "transfermarkt_player_url": _absolute_url(player_data.get("relativeUrl", "")),
                }
            )
            additions.append(row)

    if not additions:
        return squad_df

    return pd.DataFrame([*records, *additions], columns=OUTPUT_COLUMNS).drop_duplicates(
        ["team_name", "transfermarkt_player_id", "player_name"],
        keep="last",
    )


def enrich_player_profiles(driver, squad_df: pd.DataFrame) -> pd.DataFrame:
    if squad_df.empty:
        return squad_df

    wait = _scraper_config().get("profile_delay_seconds", 1.0)
    records = squad_df.to_dict("records")
    for idx, record in enumerate(records, start=1):
        url = record.get("transfermarkt_player_url", "")
        if not url:
            continue
        try:
            logger.debug(f"Enriching player profile {idx}/{len(records)}: {record['player_name']}")
            soup = fetch_soup(driver, url, wait_seconds=wait)
            record.update(parse_player_profile_html(str(soup)))
        except WebDriverException as exc:
            logger.warning(f"Could not enrich {record.get('player_name', '')}: {exc}")

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def enrich_existing_squads(
    input_path: str = "data/raw/squads.csv",
    output_path: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    if output_path is None:
        output_path = input_path
    if not os.path.exists(input_path):
        logger.error(f"Squads data not found at {input_path}. Run squad_data.py first.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    squad_df = pd.read_csv(input_path)
    if squad_df.empty:
        logger.warning(f"Squads data at {input_path} is empty.")
        return squad_df

    source = (source or _scraper_config().get("profile_enrichment_source", "api")).lower()
    original_row_count = len(squad_df)
    if source == "api":
        squad_df = add_supplemental_players_from_api(squad_df)

    needs_enrichment = (
        squad_df.get("club_league", pd.Series("", index=squad_df.index)).fillna("").eq("Unknown")
        | pd.to_numeric(squad_df.get("club_league_tier", pd.Series(0, index=squad_df.index)), errors="coerce").fillna(0).eq(0)
    )
    logger.info(f"Enriching {int(needs_enrichment.sum())} player profiles from {input_path} using {source}")
    if not needs_enrichment.any():
        if len(squad_df) != original_row_count:
            squad_df.to_csv(output_path, index=False)
            logger.info(f"Saved enriched squads to {output_path}")
        return squad_df

    if source == "api":
        enriched_subset = enrich_player_profiles_from_api(squad_df.loc[needs_enrichment].copy())
    elif source == "profile":
        driver = get_driver()
        try:
            enriched_subset = enrich_player_profiles(driver, squad_df.loc[needs_enrichment].copy())
        finally:
            driver.quit()
    else:
        raise ValueError(f"Unknown Transfermarkt enrichment source: {source}")

    for column in ["age", "caps", "club_name", "club_league", "club_league_tier", "market_value_eur"]:
        squad_df.loc[needs_enrichment, column] = enriched_subset[column].values

    squad_df.to_csv(output_path, index=False)
    logger.info(f"Saved enriched squads to {output_path}")
    return squad_df


def scrape_squad_data(driver, team_name: str, enrich_profiles: bool | None = None) -> pd.DataFrame:
    logger.info(f"Scraping Transfermarkt squad data for {team_name}")
    team_url = discover_transfermarkt_squad_url(driver, team_name)
    if not team_url:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    soup = fetch_soup(driver, team_url)
    squad_df = parse_squad_html(str(soup), team_name=team_name, team_url=team_url)
    if squad_df.empty:
        logger.error(f"Could not parse squad table for {team_name}: {team_url}")
        return squad_df

    should_enrich = _scraper_config().get("enrich_player_profiles", False) if enrich_profiles is None else enrich_profiles
    if should_enrich:
        source = _scraper_config().get("profile_enrichment_source", "api").lower()
        if source == "api":
            squad_df = enrich_player_profiles_from_api(squad_df)
        elif source == "profile":
            squad_df = enrich_player_profiles(driver, squad_df)
        else:
            raise ValueError(f"Unknown Transfermarkt enrichment source: {source}")

    logger.info(f"Scraped {len(squad_df)} players for {team_name}")
    return squad_df


def get_all_squads(teams: Iterable[str], enrich_profiles: bool | None = None) -> pd.DataFrame:
    teams = list(teams)
    logger.info(f"Starting Transfermarkt squad scrape for {len(teams)} teams")
    driver = get_driver()

    try:
        all_squads = []
        for team in teams:
            df = scrape_squad_data(driver, team, enrich_profiles=enrich_profiles)
            if not df.empty:
                all_squads.append(df)
            time.sleep(_scraper_config().get("team_delay_seconds", 1.0))
    finally:
        driver.quit()

    if not all_squads:
        logger.error("No squad data scraped")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    final_df = pd.concat(all_squads, ignore_index=True)
    final_df = final_df.drop_duplicates(["team_name", "transfermarkt_player_id", "player_name"])
    if _scraper_config().get("profile_enrichment_source", "api").lower() == "api":
        final_df = add_supplemental_players_from_api(final_df)

    config = load_config()
    out_dir = config["paths"]["raw_data"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "squads.csv")
    final_df.to_csv(out_path, index=False)
    logger.info(f"Saved complete squad data to {out_path}")

    return final_df


def load_teams(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape national-team squads from Transfermarkt.")
    parser.add_argument("--team-file", default="data/external/qualified_teams.json")
    parser.add_argument("--teams", nargs="*", help="Optional explicit team list for a focused scrape.")
    parser.add_argument("--enrich-profiles", action="store_true", help="Scrape player profile pages for caps and club league.")
    parser.add_argument("--no-enrich-profiles", action="store_true", help="Disable profile enrichment even if config enables it.")
    parser.add_argument("--enrich-existing", action="store_true", help="Only enrich data/raw/squads.csv from player profiles.")
    parser.add_argument(
        "--enrichment-source",
        choices=["api", "profile"],
        help="Use Transfermarkt JSON API or profile HTML pages for player enrichment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.enrich_existing:
        enrich_existing_squads(source=args.enrichment_source)
    else:
        try:
            selected_teams = args.teams if args.teams else load_teams(args.team_file)
        except Exception as exc:
            logger.error(f"Could not load qualified teams: {exc}")
            selected_teams = ["Brazil", "France", "USA"]

        enrich = None
        if args.enrich_profiles:
            enrich = True
        elif args.no_enrich_profiles:
            enrich = False

        get_all_squads(selected_teams, enrich_profiles=enrich)
