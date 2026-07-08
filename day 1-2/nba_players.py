import sys
from pathlib import Path

import numpy as np
import pandas as pd


FILE = 'data/NBA Players by State.xlsx'
PICKLE_FILE = 'data/nba_players_by_state.pkl'
STAT_COLUMNS = [
    'Yrs',
    'G',
    'MP',
    'FG',
    'FGA',
    '3P',
    '3PA',
    'FT',
    'FTA',
    'ORB',
    'TRB',
    'AST',
    'STL',
    'BLK',
    'TOV',
    'PF',
    'PTS',
    'FG%',
    '3P%',
    'FT%',
    'MP.1',
    'PTS.1',
    'TRB.1',
    'AST.1',
]


def clean_name(name):
    return str(name).replace('*', '').strip().lower()


def read_dataset():
    if Path(PICKLE_FILE).exists():
        return pd.read_pickle(PICKLE_FILE)

    return pd.read_excel(FILE)


def load_players():
    df = read_dataset()
    df['clean_player'] = df['Player'].apply(clean_name)

    for column in STAT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df


def find_player(df, player_name):
    cleaned_name = clean_name(player_name)
    exact_matches = df[df['clean_player'] == cleaned_name]

    if len(exact_matches) == 1:
        return exact_matches.iloc[0]

    partial_matches = df[df['clean_player'].str.contains(cleaned_name, na=False)]

    if len(partial_matches) == 1:
        return partial_matches.iloc[0]

    if len(partial_matches) > 1:
        matches = partial_matches['Player'].head(10).tolist()
        raise ValueError(f'Multiple matches for "{player_name}": {matches}')

    raise ValueError(f'No player found for "{player_name}"')


def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return 0

    return numerator / denominator


def player_score(player):
    games = player['G']
    steals_per_game = safe_divide(player['STL'], games)
    blocks_per_game = safe_divide(player['BLK'], games)
    turnovers_per_game = safe_divide(player['TOV'], games)

    points = 0 if pd.isna(player['PTS.1']) else player['PTS.1']
    rebounds = 0 if pd.isna(player['TRB.1']) else player['TRB.1']
    assists = 0 if pd.isna(player['AST.1']) else player['AST.1']
    minutes = 0 if pd.isna(player['MP.1']) else player['MP.1']
    field_goal_pct = 0 if pd.isna(player['FG%']) else player['FG%']
    three_point_pct = 0 if pd.isna(player['3P%']) else player['3P%']
    free_throw_pct = 0 if pd.isna(player['FT%']) else player['FT%']

    return (
        points * 3
        + rebounds * 1.5
        + assists * 2
        + steals_per_game * 2.5
        + blocks_per_game * 2.5
        + field_goal_pct * 20
        + three_point_pct * 8
        + free_throw_pct * 5
        + minutes * 0.5
        - turnovers_per_game * 2
    )


def comparison_table(player_one, player_two, score_one, score_two):
    rows = []
    comparison_columns = ['PTS.1', 'TRB.1', 'AST.1', 'FG%', '3P%', 'FT%', 'G', 'Yrs']

    for column in comparison_columns:
        value_one = player_one[column]
        value_two = player_two[column]
        rows.append(
            {
                'Header': column,
                player_one['Player']: value_one,
                player_two['Player']: value_two,
                'Advantage': player_one['Player'] if value_one > value_two else player_two['Player'],
            }
        )

    rows.append(
        {
            'Header': '1v1 score',
            player_one['Player']: round(score_one, 2),
            player_two['Player']: round(score_two, 2),
            'Advantage': player_one['Player'] if score_one > score_two else player_two['Player'],
        }
    )

    return pd.DataFrame(rows)


def predict_winner(df, player_one_name, player_two_name):
    player_one = find_player(df, player_one_name)
    player_two = find_player(df, player_two_name)
    score_one = player_score(player_one)
    score_two = player_score(player_two)

    if np.isclose(score_one, score_two):
        winner = 'Tie'
    else:
        winner = player_one['Player'] if score_one > score_two else player_two['Player']

    table = comparison_table(player_one, player_two, score_one, score_two)
    return winner, table


def main():
    df = load_players()

    if len(sys.argv) >= 3:
        player_one_name = sys.argv[1]
        player_two_name = sys.argv[2]
    else:
        player_one_name = input('First player: ')
        player_two_name = input('Second player: ')

    winner, table = predict_winner(df, player_one_name, player_two_name)

    print()
    print(table.to_string(index=False))
    print()
    print(f'Predicted 1v1 winner: {winner}')


if __name__ == '__main__':
    main()
