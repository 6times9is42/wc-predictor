import pandas as pd

from src.scraper.player_ratings import build_transfermarkt_ratings
from src.scraper.squad_data import (
    infer_competition_tier,
    parse_market_value,
    parse_player_profile_html,
    parse_squad_html,
    parse_transfermarkt_api_player,
)


def test_parse_market_value_common_units():
    assert parse_market_value("\u20ac30.00m") == 30_000_000
    assert parse_market_value("\u20ac1.35bn Total market value") == 1_350_000_000
    assert parse_market_value("\u20ac125k") == 125_000
    assert parse_market_value("-") == 0


def test_parse_squad_html_extracts_transfermarkt_rows():
    html = """
    <html>
      <body>
        <div class="data-header__market-value-wrapper">EUR 1.35bn Total market value</div>
        <main>
          <span>Confederation: UEFA</span>
          <span>FIFA World ranking: Pos 3</span>
        </main>
        <table class="items">
          <tbody>
            <tr class="odd">
              <td class="zentriert rueckennummer bg_Torwart" title="Goalkeeper"><div>23</div></td>
              <td>
                <table class="inline-table">
                  <tr>
                    <td rowspan="2"><img alt="Lucas Chevalier" /></td>
                    <td class="hauptlink"><a href="/lucas-chevalier/profil/spieler/463600">Lucas Chevalier</a></td>
                  </tr>
                  <tr><td>Goalkeeper</td></tr>
                </table>
              </td>
              <td class="zentriert">24</td>
              <td class="zentriert"><img alt="Paris Saint-Germain" title="Paris Saint-Germain" /></td>
              <td class="rechts hauptlink"><a>EUR 30.00m</a></td>
            </tr>
            <tr class="even">
              <td class="zentriert rueckennummer bg_Abwehr" title="Centre-Back"><div>4</div></td>
              <td>
                <table class="inline-table">
                  <tr>
                    <td rowspan="2"><img alt="Dayot Upamecano" /></td>
                    <td class="hauptlink"><a href="/dayot-upamecano/profil/spieler/344695">Dayot Upamecano</a></td>
                  </tr>
                  <tr><td>Centre-Back</td></tr>
                </table>
              </td>
              <td class="zentriert">27</td>
              <td class="zentriert"><img alt="Bayern Munich" title="Bayern Munich" /></td>
              <td class="rechts hauptlink"><a>EUR 70.00m</a></td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    df = parse_squad_html(html, team_name="France", team_url="https://example.test/france")

    assert len(df) == 2
    assert df.loc[0, "player_name"] == "Lucas Chevalier"
    assert df.loc[0, "position"] == "GK"
    assert df.loc[1, "position"] == "DEF"
    assert df.loc[1, "market_value_eur"] == 70_000_000
    assert df.loc[0, "team_confederation"] == "UEFA"
    assert df.loc[0, "team_fifa_ranking"] == 3


def test_parse_player_profile_html_extracts_caps_and_league():
    html = """
    <html><body>
      <div>Current international:</div><div>France</div>
      <div>Caps/Goals: 12 / 0</div>
      <div>Paris Saint-Germain</div>
      <div>Ligue 1</div>
      <div>League level:</div>
      <div>First Tier</div>
    </body></html>
    """

    parsed = parse_player_profile_html(html)

    assert parsed["caps"] == 12
    assert parsed["club_league"] == "Ligue 1"
    assert parsed["club_league_tier"] == 1


def test_parse_transfermarkt_api_player_extracts_current_club_caps_and_value():
    parsed = parse_transfermarkt_api_player(
        {
            "lifeDates": {"age": 27},
            "marketValueDetails": {"current": {"value": 200_000_000}},
            "clubAssignments": [
                {"type": "nationalTeam", "clubId": "3377"},
                {"type": "current", "clubId": "418"},
            ],
        },
        {
            "history": [
                {"careerState": "CURRENT_NATIONAL_PLAYER", "gamesPlayed": 96},
                {"careerState": "RECENT_NATIONAL_PLAYER", "gamesPlayed": 92},
                {"careerState": "FORMER_NATIONAL_PLAYER", "gamesPlayed": 11},
            ]
        },
    )

    assert parsed["age"] == 27
    assert parsed["caps"] == 96
    assert parsed["market_value_eur"] == 200_000_000
    assert parsed["transfermarkt_current_club_id"] == "418"


def test_parse_transfermarkt_api_player_accepts_recent_national_player_caps():
    parsed = parse_transfermarkt_api_player(
        {"lifeDates": {"age": 41}, "clubAssignments": [{"type": "current", "clubId": "18544"}]},
        {"history": [{"careerState": "RECENT_NATIONAL_PLAYER", "gamesPlayed": 226}]},
    )

    assert parsed["caps"] == 226


def test_infer_competition_tier_from_transfermarkt_competition():
    assert infer_competition_tier("ES1", "LaLiga") == 1
    assert infer_competition_tier("GB2", "Championship") == 2
    assert infer_competition_tier("L3", "3. Liga") == 3


def test_transfermarkt_ratings_are_deterministic_and_value_driven():
    squads = pd.DataFrame(
        [
            {
                "player_name": "High Value",
                "team_name": "France",
                "position": "FWD",
                "age": 27,
                "caps": 60,
                "club_name": "Real Madrid",
                "club_league": "LaLiga",
                "club_league_tier": 1,
                "market_value_eur": 200_000_000,
                "injured": False,
                "suspended": False,
            },
            {
                "player_name": "Lower Value",
                "team_name": "France",
                "position": "DEF",
                "age": 31,
                "caps": 5,
                "club_name": "Unknown Club",
                "club_league": "Unknown",
                "club_league_tier": 0,
                "market_value_eur": 1_000_000,
                "injured": False,
                "suspended": False,
            },
        ]
    )

    first = build_transfermarkt_ratings(squads)
    second = build_transfermarkt_ratings(squads)

    assert first["rating_composite"].tolist() == second["rating_composite"].tolist()
    assert first.loc[0, "rating_composite"] > first.loc[1, "rating_composite"]
    assert set(first["rating_source"]) == {"transfermarkt_market_value"}
