import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguegamefinder,
    leaguedashplayerstats,
    leaguedashteamstats,
)
from nba_api.stats.static import teams


SEASON = '2025-26'
RANDOM_SEED = 42
SIMULATOR_VERSION = 1
SIMULATOR_PICKLE_FILE = Path(__file__).resolve().parent / 'data' / 'nba_team_simulator.pkl'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def find_team(team_name):
    query = team_name.strip().lower()
    all_teams = teams.get_teams()

    matches = [
        team
        for team in all_teams
        if query in team['full_name'].lower()
        or query == team['abbreviation'].lower()
        or query in team['nickname'].lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        names = [team['full_name'] for team in matches]
        raise ValueError(f'Multiple teams matched "{team_name}": {names}')

    raise ValueError(f'No NBA team found for "{team_name}"')


def clean_player_name(player_name):
    return str(player_name).replace('*', '').strip().lower()


def is_injured_player(player_name, injured_names):
    cleaned_player = clean_player_name(player_name)
    return any(injured == cleaned_player or injured in cleaned_player for injured in injured_names)


def fetch_with_pause(endpoint_factory):
    time.sleep(0.6)
    return endpoint_factory().get_data_frames()[0]


def get_roster(team_id):
    return fetch_with_pause(
        lambda: commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=SEASON,
            timeout=30,
        )
    )


def get_team_stats():
    return fetch_with_pause(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            per_mode_detailed='PerGame',
            timeout=30,
        )
    )


def get_player_stats():
    return fetch_with_pause(
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=SEASON,
            per_mode_detailed='PerGame',
            timeout=30,
        )
    )


def get_games(team_id):
    return fetch_with_pause(
        lambda: leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=SEASON,
            timeout=30,
        )
    )


def team_recent_form(games, count=10):
    recent_games = games.head(count)

    if recent_games.empty:
        return {'recent_points': 110, 'recent_win_pct': 0.5}

    return {
        'recent_points': recent_games['PTS'].mean(),
        'recent_win_pct': (recent_games['WL'] == 'W').mean(),
    }


def head_to_head_boost(team_games, opponent_abbreviation):
    h2h = team_games[
        team_games['MATCHUP'].astype(str).str.contains(opponent_abbreviation, na=False)
    ]

    if h2h.empty:
        return 0

    wins = (h2h['WL'] == 'W').mean()
    margin = h2h['PLUS_MINUS'].mean() if 'PLUS_MINUS' in h2h else 0
    return (wins - 0.5) * 4 + margin * 0.12


def injury_score_penalty(roster, player_stats, team_id, injured_players=None):
    injured_players = injured_players or []
    injured_names = {clean_player_name(player) for player in injured_players if player.strip()}

    if not injured_names:
        return 0

    roster_ids = set(roster['PLAYER_ID'])
    players = player_stats[
        (player_stats['TEAM_ID'] == team_id) | (player_stats['PLAYER_ID'].isin(roster_ids))
    ].copy()
    injured_stats = players[
        players['PLAYER_NAME'].apply(lambda name: is_injured_player(name, injured_names))
    ].copy()

    if injured_stats.empty:
        return 0

    injured_stats['PTS'] = pd.to_numeric(injured_stats['PTS'], errors='coerce').fillna(0)
    injured_stats['MIN'] = pd.to_numeric(injured_stats['MIN'], errors='coerce').fillna(0)
    return min(18, (injured_stats['PTS'] * 0.45 + injured_stats['MIN'] * 0.08).sum())


def get_team_row(team_stats, team_id, team_name):
    match = team_stats[team_stats['TEAM_ID'] == team_id]

    if match.empty:
        raise ValueError(f'Missing team stats for {team_name}')

    return match.iloc[0]


def estimate_team_score(team, opponent, team_stats, games, opponent_games):
    team_row = get_team_row(team_stats, team['id'], team['full_name'])
    opponent_row = get_team_row(team_stats, opponent['id'], opponent['full_name'])
    recent = team_recent_form(games)
    opponent_recent = team_recent_form(opponent_games)

    base_score = (
        team_row['PTS'] * 0.45
        + recent['recent_points'] * 0.25
        + (120 - opponent_row['PLUS_MINUS']) * 0.08
        + (120 - opponent_recent['recent_points']) * 0.12
        + 110 * 0.10
    )
    form_boost = (recent['recent_win_pct'] - 0.5) * 6
    h2h_boost = head_to_head_boost(games, opponent['abbreviation'])
    random_noise = random.normalvariate(0, 5)

    return int(round(base_score + form_boost + h2h_boost + random_noise))


def get_team_players(roster, player_stats, team_id, injured_players=None):
    injured_players = injured_players or []
    injured_names = {clean_player_name(player) for player in injured_players if player.strip()}
    roster_ids = set(roster['PLAYER_ID'])
    players = player_stats[
        (player_stats['TEAM_ID'] == team_id) | (player_stats['PLAYER_ID'].isin(roster_ids))
    ].copy()

    if players.empty:
        players = pd.DataFrame(
            {
                'PLAYER_NAME': roster['PLAYER'],
                'PLAYER_ID': roster['PLAYER_ID'],
                'PTS': 4,
                'REB': 2,
                'AST': 1,
                'MIN': 12,
                'NBA_FANTASY_PTS': 10,
            }
        )

    if injured_names:
        players = players[
            ~players['PLAYER_NAME'].apply(lambda name: is_injured_player(name, injured_names))
        ].copy()

    if players.empty:
        raise ValueError('All available players were marked injured for one team.')

    for column in ['PTS', 'REB', 'AST', 'MIN', 'NBA_FANTASY_PTS']:
        players[column] = pd.to_numeric(players[column], errors='coerce').fillna(0)

    players['impact'] = (
        players['PTS'] * 1.0
        + players['REB'] * 0.8
        + players['AST'] * 1.1
        + players['MIN'] * 0.35
        + players['NBA_FANTASY_PTS'] * 0.15
    )
    players = players.sort_values('impact', ascending=False).head(10)

    return players


def assign_minutes(players):
    total_impact = players['impact'].sum()

    if total_impact == 0:
        players['SIM_MIN'] = 24
    else:
        players['SIM_MIN'] = 8 + (players['impact'] / total_impact * 160)

    players['SIM_MIN'] = players['SIM_MIN'].clip(lower=8, upper=38)
    players['SIM_MIN'] = players['SIM_MIN'] / players['SIM_MIN'].sum() * 240
    return players


def simulate_box_score(team, players, team_score):
    players = assign_minutes(players.copy())
    total_scoring_weight = (players['PTS'] * players['SIM_MIN']).sum()

    if total_scoring_weight == 0:
        players['PTS_WEIGHT'] = 1 / len(players)
    else:
        players['PTS_WEIGHT'] = players['PTS'] * players['SIM_MIN'] / total_scoring_weight

    rows = []
    for _, player in players.iterrows():
        minutes_ratio = player['SIM_MIN'] / max(player['MIN'], 1)
        points = max(0, team_score * player['PTS_WEIGHT'] + random.normalvariate(0, 2.2))
        rebounds = max(0, player['REB'] * minutes_ratio + random.normalvariate(0, 1.3))
        assists = max(0, player['AST'] * minutes_ratio + random.normalvariate(0, 1.1))

        rows.append(
            {
                'Player': player['PLAYER_NAME'],
                'Player ID': int(player['PLAYER_ID']),
                'Team': team['abbreviation'],
                'MIN': round(player['SIM_MIN'], 1),
                'PTS': int(round(points)),
                'REB': int(round(rebounds)),
                'AST': int(round(assists)),
            }
        )

    box_score = pd.DataFrame(rows)
    point_difference = team_score - box_score['PTS'].sum()
    if not box_score.empty:
        best_scorer_index = box_score['PTS'].idxmax()
        box_score.loc[best_scorer_index, 'PTS'] += point_difference

    return box_score.sort_values(['PTS', 'REB', 'AST'], ascending=False)


def choose_mvp(winning_box_score):
    mvp_scores = (
        winning_box_score['PTS']
        + winning_box_score['REB'] * 1.2
        + winning_box_score['AST'] * 1.5
    )
    return winning_box_score.loc[mvp_scores.idxmax()]


def format_table(df):
    return df.to_string(index=False)


def build_simulation_result(team_one_name, team_two_name, team_one_injuries=None, team_two_injuries=None):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    team_one = find_team(team_one_name)
    team_two = find_team(team_two_name)

    if team_one['id'] == team_two['id']:
        raise ValueError('Choose two different teams.')

    team_stats = get_team_stats()
    player_stats = get_player_stats()
    team_one_roster = get_roster(team_one['id'])
    team_two_roster = get_roster(team_two['id'])
    team_one_games = get_games(team_one['id'])
    team_two_games = get_games(team_two['id'])

    team_one_score = estimate_team_score(
        team_one,
        team_two,
        team_stats,
        team_one_games,
        team_two_games,
    )
    team_two_score = estimate_team_score(
        team_two,
        team_one,
        team_stats,
        team_two_games,
        team_one_games,
    )

    team_one_score -= int(round(injury_score_penalty(
        team_one_roster,
        player_stats,
        team_one['id'],
        team_one_injuries,
    )))
    team_two_score -= int(round(injury_score_penalty(
        team_two_roster,
        player_stats,
        team_two['id'],
        team_two_injuries,
    )))

    if team_one_score == team_two_score:
        team_one_score += 1

    team_one_players = get_team_players(
        team_one_roster,
        player_stats,
        team_one['id'],
        team_one_injuries,
    )
    team_two_players = get_team_players(
        team_two_roster,
        player_stats,
        team_two['id'],
        team_two_injuries,
    )
    team_one_box = simulate_box_score(team_one, team_one_players, team_one_score)
    team_two_box = simulate_box_score(team_two, team_two_players, team_two_score)
    full_box_score = pd.concat([team_one_box, team_two_box], ignore_index=True)

    winner = team_one if team_one_score > team_two_score else team_two
    winning_box_score = team_one_box if winner['id'] == team_one['id'] else team_two_box
    mvp = choose_mvp(winning_box_score)

    return {
        'team_one': team_one,
        'team_two': team_two,
        'team_one_score': team_one_score,
        'team_two_score': team_two_score,
        'winner': winner,
        'mvp': mvp,
        'box_score': full_box_score[['Player', 'Team', 'MIN', 'PTS', 'REB', 'AST']],
    }


def print_simulation_result(result):
    team_one = result['team_one']
    team_two = result['team_two']
    mvp = result['mvp']

    print()
    print('Simulated Game')
    print(f'{team_one["full_name"]}: {result["team_one_score"]}')
    print(f'{team_two["full_name"]}: {result["team_two_score"]}')
    print()
    print(f'Winner: {result["winner"]["full_name"]}')
    print(f'MVP: {mvp["Player"]} ({mvp["PTS"]} PTS, {mvp["REB"]} REB, {mvp["AST"]} AST)')
    print()
    print('Player Box Score')
    print(format_table(result['box_score']))


class NBATeamSimulator:
    """Serializable entry point for the NBA team simulation algorithm."""

    def __init__(self, season=SEASON, random_seed=RANDOM_SEED):
        self.season = season
        self.random_seed = random_seed
        self.version = SIMULATOR_VERSION

    def find_team(self, team_name):
        return find_team(team_name)

    def get_roster(self, team_id):
        return get_roster(team_id)

    def simulate(self, team_one_name, team_two_name, team_one_injuries=None, team_two_injuries=None):
        global SEASON, RANDOM_SEED
        previous_season = SEASON
        previous_seed = RANDOM_SEED
        SEASON = self.season
        RANDOM_SEED = self.random_seed

        try:
            return build_simulation_result(
                team_one_name,
                team_two_name,
                team_one_injuries,
                team_two_injuries,
            )
        finally:
            SEASON = previous_season
            RANDOM_SEED = previous_seed


def save_simulator(path=SIMULATOR_PICKLE_FILE):
    simulator = NBATeamSimulator()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('wb') as simulator_file:
        pickle.dump(simulator, simulator_file, protocol=pickle.HIGHEST_PROTOCOL)

    return simulator


def load_simulator(path=SIMULATOR_PICKLE_FILE):
    path = Path(path)

    if path.exists():
        with path.open('rb') as simulator_file:
            simulator = pickle.load(simulator_file)

        if isinstance(simulator, NBATeamSimulator) and simulator.version == SIMULATOR_VERSION:
            return simulator

    return save_simulator(path)


def simulate_game(team_one_name, team_two_name, team_one_injuries=None, team_two_injuries=None):
    print(f'Loading NBA data for {SEASON}...')
    simulator = load_simulator()
    result = simulator.simulate(
        team_one_name,
        team_two_name,
        team_one_injuries,
        team_two_injuries,
    )
    print_simulation_result(result)


def main():
    if len(sys.argv) >= 3:
        team_one_name = sys.argv[1]
        team_two_name = sys.argv[2]
    else:
        team_one_name = input('First team: ')
        team_two_name = input('Second team: ')

    simulate_game(team_one_name, team_two_name)


if __name__ == '__main__':
    main()
