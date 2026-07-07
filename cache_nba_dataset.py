import pandas as pd


EXCEL_FILE = 'data/NBA Players by State.xlsx'
PICKLE_FILE = 'data/nba_players_by_state.pkl'


df = pd.read_excel(EXCEL_FILE)
df.to_pickle(PICKLE_FILE)

print(f'Cached {len(df)} rows to {PICKLE_FILE}')
