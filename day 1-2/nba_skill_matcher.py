import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PICKLE_FILE = PROJECT_DIR / 'data' / 'nba_players_by_state.pkl'
EXCEL_FILE = PROJECT_DIR / 'data' / 'NBA Players by State.xlsx'
MODEL_PICKLE_FILE = PROJECT_DIR / 'data' / 'nba_skill_matcher_model.pkl'
MODEL_VERSION = 2
FEATURE_COLUMNS = ['PTS.1', 'TRB.1', 'AST.1', 'FG%', '3P%', 'FT%', 'MP.1']
MATCH_WEIGHTS = {
    'PTS.1': 0.12,
    'TRB.1': 0.15,
    'AST.1': 0.16,
    'FG%': 0.15,
    '3P%': 0.15,
    'FT%': 0.15,
    'MP.1': 0.12,
}
DISPLAY_NAMES = {
    'PTS.1': 'Points/Game', 'TRB.1': 'Rebounds/Game', 'AST.1': 'Assists/Game',
    'FG%': 'FG%', '3P%': '3PT%', 'FT%': 'FT%', 'MP.1': 'Minutes/Game',
    'rf_skill_score': 'Skill Score', 'distance_from_you': 'Similarity',
}


def load_players():
    df = pd.read_pickle(DATA_PICKLE_FILE) if DATA_PICKLE_FILE.exists() else pd.read_excel(EXCEL_FILE)
    df['Player'] = df['Player'].astype(str).str.replace('*', '', regex=False).str.strip()
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=['PTS.1', 'TRB.1', 'AST.1']).copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())
    return df


def skill_score(df):
    return (
        df['PTS.1'] * 1.5 + df['TRB.1'] * 1.5 + df['AST.1'] * 2
        + df['FG%'] * 15 + df['3P%'] * 15 + df['FT%'] * 15 + df['MP.1'] * 0.5
    )


def train_random_forest(df):
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(df[FEATURE_COLUMNS], skill_score(df))
    model.nba_matcher_version = MODEL_VERSION
    return model


def load_or_train_model(df):
    if MODEL_PICKLE_FILE.exists():
        with MODEL_PICKLE_FILE.open('rb') as model_file:
            model = pickle.load(model_file)
        if (
            getattr(model, 'n_features_in_', None) == len(FEATURE_COLUMNS)
            and getattr(model, 'nba_matcher_version', None) == MODEL_VERSION
        ):
            return model

    model = train_random_forest(df)
    MODEL_PICKLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PICKLE_FILE.open('wb') as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    return model


def find_closest_players(df, model, user_stats, count=10):
    scaler = StandardScaler()
    player_features = scaler.fit_transform(df[FEATURE_COLUMNS])
    user_features = scaler.transform(user_stats[FEATURE_COLUMNS])
    feature_weights = np.array([MATCH_WEIGHTS[column] for column in FEATURE_COLUMNS])
    weighted_distances = np.sqrt(((player_features - user_features) ** 2 * feature_weights).sum(axis=1))
    results = df[['Player', *FEATURE_COLUMNS]].copy()
    results['rf_skill_score'] = model.predict(df[FEATURE_COLUMNS])
    results['distance_from_you'] = weighted_distances
    return results.sort_values('distance_from_you').head(count)


def format_match_table(closest_players):
    table = closest_players.copy()
    max_distance = table['distance_from_you'].max()
    table['distance_from_you'] = 100 if max_distance == 0 else 100 - (table['distance_from_you'] / max_distance * 35)
    table = table.rename(columns=DISPLAY_NAMES)
    table['Skill Score'] = table['Skill Score'].round(1)
    table['Similarity'] = table['Similarity'].round(1).astype(str) + '/100'
    for column in ['Points/Game', 'Rebounds/Game', 'Assists/Game', 'Minutes/Game']:
        table[column] = table[column].round(1)
    for column in ['FG%', '3PT%', 'FT%']:
        table[column] = (table[column] * 100).round(1).astype(str) + '%'
    return table[['Player', 'Similarity', 'Skill Score', 'Points/Game', 'Rebounds/Game', 'Assists/Game', 'FG%', '3PT%', 'FT%', 'Minutes/Game']]


def format_importance_table(model):
    importances = pd.DataFrame({
        'Stat': [DISPLAY_NAMES[column] for column in FEATURE_COLUMNS],
        'Importance': [MATCH_WEIGHTS[column] for column in FEATURE_COLUMNS],
    }).sort_values('Importance', ascending=False)
    importances['Importance'] = (importances['Importance'] * 100).round(1).astype(str) + '%'
    return importances
