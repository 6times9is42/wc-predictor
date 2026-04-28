import json
import numpy as np
import pandas as pd
from loguru import logger
from src.model.predict import predict_match

R32_MATCH_SLOTS = [
    (73, ("second", "A"), ("second", "B")),
    (74, ("winner", "E"), ("third", ("A", "B", "C", "D", "F"))),
    (75, ("winner", "F"), ("second", "C")),
    (76, ("winner", "C"), ("second", "F")),
    (77, ("winner", "I"), ("third", ("C", "D", "F", "G", "H"))),
    (78, ("second", "E"), ("second", "I")),
    (79, ("winner", "A"), ("third", ("C", "E", "F", "H", "I"))),
    (80, ("winner", "L"), ("third", ("E", "H", "I", "J", "K"))),
    (81, ("winner", "D"), ("third", ("B", "E", "F", "I", "J"))),
    (82, ("winner", "G"), ("third", ("A", "E", "H", "I", "J"))),
    (83, ("second", "K"), ("second", "L")),
    (84, ("winner", "H"), ("second", "J")),
    (85, ("winner", "B"), ("third", ("E", "F", "G", "I", "J"))),
    (86, ("winner", "J"), ("second", "H")),
    (87, ("winner", "K"), ("third", ("D", "E", "I", "J", "L"))),
    (88, ("second", "D"), ("second", "G")),
]

R16_MATCH_SLOTS = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QF_MATCH_SLOTS = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

SF_MATCH_SLOTS = {
    101: (97, 98),
    102: (99, 100),
}


class BracketSimulator:
    def __init__(self, team_features_df: pd.DataFrame = None, prediction_lookup: dict | None = None):
        with open("data/external/wc2026_groups.json", "r") as f:
            self.groups = json.load(f)
            
        self.team_features = team_features_df if team_features_df is not None else pd.DataFrame()
        self.prediction_lookup = prediction_lookup or {}
        
    def _get_elo(self, team: str) -> float:
        if not self.team_features.empty:
            match = self.team_features[self.team_features["team_name"] == team]
            if not match.empty:
                return float(match.iloc[0]["team_elo"])
        return 1500.0
        
    def _get_gk_rating(self, team: str) -> float:
        if not self.team_features.empty:
            match = self.team_features[self.team_features["team_name"] == team]
            if not match.empty:
                return float(match.iloc[0].get("gk_rating", 5.0))
        return 5.0

    def simulate_match(self, team_a: str, team_b: str, is_knockout: bool = False) -> dict:
        """
        Simulates a single match and returns the outcome.
        For group stage, can return draw.
        For knockout, resolves draws via extra time and penalties.
        """
        pred = self.prediction_lookup.get((team_a, team_b))
        if pred is None:
            try:
                pred = predict_match(team_a, team_b, neutral=True)
            except Exception as exc:
                logger.debug(f"Prediction fallback for {team_a} vs {team_b}: {exc}")
                pred = None
        if not pred:
            # Fallback
            pred = {"home_win_prob": 0.33, "draw_prob": 0.34, "away_win_prob": 0.33,
                    "expected_home_goals": 1.0, "expected_away_goals": 1.0}
                    
        probs = [pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]]
        # Normalize just in case
        probs = [p / sum(probs) for p in probs]
        
        outcome = np.random.choice(["H", "D", "A"], p=probs)
        
        # Simulate goals based on expected goals using Poisson distribution
        home_goals = np.random.poisson(pred["expected_home_goals"])
        away_goals = np.random.poisson(pred["expected_away_goals"])
        
        # Adjust goals if they contradict the outcome
        if outcome == "H" and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif outcome == "A" and away_goals <= home_goals:
            away_goals = home_goals + 1
        elif outcome == "D":
            home_goals = away_goals = (home_goals + away_goals) // 2
            
        result = {
            "winner": team_a if outcome == "H" else (team_b if outcome == "A" else "Draw"),
            "team_a_goals": home_goals,
            "team_b_goals": away_goals,
            "outcome": outcome
        }
        
        if is_knockout and outcome == "D":
            # Extra time simulation
            elo_a = self._get_elo(team_a)
            elo_b = self._get_elo(team_b)
            
            # Base 50/50, adjusted by ELO diff (max 60/40)
            elo_diff = elo_a - elo_b
            prob_a_et = 0.5 + np.clip(elo_diff / 1000.0, -0.1, 0.1)
            
            # 30% chance match ends in Extra Time
            if np.random.random() < 0.3:
                et_winner = team_a if np.random.random() < prob_a_et else team_b
                result["winner"] = et_winner
                if et_winner == team_a:
                    result["team_a_goals"] += 1
                else:
                    result["team_b_goals"] += 1
                result["resolved_in"] = "ET"
            else:
                # Penalties
                gk_a = self._get_gk_rating(team_a)
                gk_b = self._get_gk_rating(team_b)
                
                # GK rating diff shifts penalty win probability
                prob_a_pen = 0.5 + np.clip((gk_a - gk_b) / 20.0, -0.1, 0.1)
                
                pen_winner = team_a if np.random.random() < prob_a_pen else team_b
                result["winner"] = pen_winner
                result["resolved_in"] = "PEN"
                
        return result

    def simulate_group_stage(self) -> dict:
        """Simulates all 12 groups."""
        standings = {}
        for group_name, teams in self.groups.items():
            group_standings = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams}
            
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a = teams[i]
                    team_b = teams[j]
                    
                    res = self.simulate_match(team_a, team_b, is_knockout=False)
                    
                    group_standings[team_a]["gf"] += res["team_a_goals"]
                    group_standings[team_b]["gf"] += res["team_b_goals"]
                    
                    gd = res["team_a_goals"] - res["team_b_goals"]
                    group_standings[team_a]["gd"] += gd
                    group_standings[team_b]["gd"] -= gd
                    
                    if res["outcome"] == "H":
                        group_standings[team_a]["pts"] += 3
                    elif res["outcome"] == "A":
                        group_standings[team_b]["pts"] += 3
                    else:
                        group_standings[team_a]["pts"] += 1
                        group_standings[team_b]["pts"] += 1
                        
            # Sort group: PTS -> GD -> GF -> Random
            sorted_group = sorted(
                group_standings.items(),
                key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], np.random.random()),
                reverse=True
            )
            standings[group_name] = sorted_group
            
        return standings

    def get_third_place_qualifiers(self, group_standings: dict) -> list:
        return [record["team"] for record in self.get_third_place_qualifier_records(group_standings)]

    def get_third_place_qualifier_records(self, group_standings: dict) -> list[dict]:
        third_places = []
        for group_name, sorted_teams in group_standings.items():
            team_data = sorted_teams[2]
            team_name = team_data[0]
            stats = team_data[1]
            third_places.append({"team": team_name, "stats": stats, "group": self._group_code(group_name)})

        # Sort best 8: PTS -> GD -> GF -> Random
        sorted_thirds = sorted(
            third_places,
            key=lambda x: (x["stats"]["pts"], x["stats"]["gd"], x["stats"]["gf"], np.random.random()),
            reverse=True
        )
        return sorted_thirds[:8]

    def _group_code(self, group_name: str) -> str:
        return group_name.replace("Group ", "").strip()

    def _assign_third_place_slots(self, third_place_records: list[dict]) -> dict[int, str]:
        third_by_group = {record["group"]: record for record in third_place_records}
        ranked_groups = [record["group"] for record in third_place_records]
        third_slots = [
            (match_id, slot[1])
            for match_id, left_slot, right_slot in R32_MATCH_SLOTS
            for slot in (left_slot, right_slot)
            if slot[0] == "third"
        ]

        def can_complete(slot_index: int, remaining_groups: set[str]) -> bool:
            for _, eligible_groups in third_slots[slot_index:]:
                if not any(group in remaining_groups and group in eligible_groups for group in ranked_groups):
                    return False
            return True

        def backtrack(slot_index: int, remaining_groups: set[str], assignment: dict[int, str]) -> dict[int, str] | None:
            if slot_index == len(third_slots):
                return assignment

            match_id, eligible_groups = third_slots[slot_index]
            options = [
                group
                for group in ranked_groups
                if group in remaining_groups and group in eligible_groups and group in third_by_group
            ]
            for group in options:
                next_remaining = set(remaining_groups)
                next_remaining.remove(group)
                if not can_complete(slot_index + 1, next_remaining):
                    continue
                next_assignment = dict(assignment)
                next_assignment[match_id] = group
                solved = backtrack(slot_index + 1, next_remaining, next_assignment)
                if solved is not None:
                    return solved
            return None

        assignment = backtrack(0, set(third_by_group), {})
        if assignment is None:
            logger.warning("Could not satisfy third-place bracket constraints; falling back to ranked assignment.")
            assignment = {}
            remaining_groups = set(third_by_group)
            for match_id, eligible_groups in third_slots:
                options = [group for group in ranked_groups if group in remaining_groups and group in eligible_groups]
                if options:
                    assignment[match_id] = options[0]
                    remaining_groups.remove(options[0])

        return assignment

    def build_r32_bracket(self, group_standings: dict, third_place_records: list[dict]) -> dict[int, tuple[str, str]]:
        qualifiers = {}
        for group_name, sorted_teams in group_standings.items():
            group = self._group_code(group_name)
            qualifiers[group] = {
                "winner": sorted_teams[0][0],
                "second": sorted_teams[1][0],
            }

        third_by_group = {record["group"]: record["team"] for record in third_place_records}
        third_assignment = self._assign_third_place_slots(third_place_records)

        def resolve(slot: tuple) -> str:
            slot_type = slot[0]
            if slot_type in {"winner", "second"}:
                return qualifiers[slot[1]][slot_type]
            if slot_type == "third":
                group = third_assignment.get(current_match_id)
                return third_by_group[group]
            raise ValueError(f"Unknown bracket slot type: {slot_type}")

        matches = {}
        for current_match_id, left_slot, right_slot in R32_MATCH_SLOTS:
            matches[current_match_id] = (resolve(left_slot), resolve(right_slot))

        return matches

    def simulate_knockout_round(self, pairs: list) -> list:
        winners = []
        next_pairs = []
        
        for i in range(0, len(pairs), 2):
            if i + 1 >= len(pairs):
                # Should not happen in standard bracket
                break
                
            match1 = pairs[i]
            match2 = pairs[i+1]
            
            w1 = self.simulate_match(match1[0], match1[1], is_knockout=True)["winner"]
            w2 = self.simulate_match(match2[0], match2[1], is_knockout=True)["winner"]
            
            winners.extend([w1, w2])
            next_pairs.append((w1, w2))
            
        return next_pairs

    def _simulate_match_map(self, match_slots: dict[int, tuple[int, int]], previous_winners: dict[int, str]) -> tuple[dict[int, tuple[str, str]], dict[int, str]]:
        matches = {
            match_id: (previous_winners[left_match_id], previous_winners[right_match_id])
            for match_id, (left_match_id, right_match_id) in match_slots.items()
        }
        winners = {
            match_id: self.simulate_match(team_a, team_b, is_knockout=True)["winner"]
            for match_id, (team_a, team_b) in matches.items()
        }
        return matches, winners

    def simulate_full_tournament(self) -> dict:
        result_log = {
            "group_stage": {},
            "r32": [],
            "r16": [],
            "qf": [],
            "sf": [],
            "final": "",
            "runner_up": "",
            "team_stats": {} # points, goals
        }
        
        # 1. Group Stage
        group_standings = self.simulate_group_stage()
        
        # Record stats
        for group, teams in group_standings.items():
            for team, stats in teams:
                result_log["team_stats"][team] = {"pts": stats["pts"], "gf": stats["gf"], "gc": stats["gf"] - stats["gd"]}
                
        # 2. Build the official 2026 knockout skeleton.
        thirds = self.get_third_place_qualifier_records(group_standings)
        r32_matches = self.build_r32_bracket(group_standings, thirds)
        result_log["r32"] = [team for match_id in sorted(r32_matches) for team in r32_matches[match_id]]
        r32_winners = {
            match_id: self.simulate_match(team_a, team_b, is_knockout=True)["winner"]
            for match_id, (team_a, team_b) in r32_matches.items()
        }

        r16_matches, r16_winners = self._simulate_match_map(R16_MATCH_SLOTS, r32_winners)
        result_log["r16"] = [team for match_id in sorted(r16_matches) for team in r16_matches[match_id]]

        qf_matches, qf_winners = self._simulate_match_map(QF_MATCH_SLOTS, r16_winners)
        result_log["qf"] = [team for match_id in sorted(qf_matches) for team in qf_matches[match_id]]

        sf_matches, sf_winners = self._simulate_match_map(SF_MATCH_SLOTS, qf_winners)
        result_log["sf"] = [team for match_id in sorted(sf_matches) for team in sf_matches[match_id]]

        final_match = (sf_winners[101], sf_winners[102])
        winner = self.simulate_match(final_match[0], final_match[1], is_knockout=True)["winner"]
        runner_up = final_match[1] if winner == final_match[0] else final_match[0]

        result_log["final"] = winner
        result_log["runner_up"] = runner_up
            
        return result_log
