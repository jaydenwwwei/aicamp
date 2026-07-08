import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


file = 'kaggle_chess-suite_leaderboard.json'


def get_model_family(model_name):
    families = ['Gemini', 'GPT', 'Claude', 'Grok', 'DeepSeek', 'o3']

    for family in families:
        if model_name.startswith(family):
            return family

    return 'Other'

with open(file, encoding='utf-8') as f:
    data = json.load(f)

rows = []
for row in data['rows']:
    flattened_row = {
        'modelVersionName': row['modelVersionName'],
        'modelVersionSlug': row['modelVersionSlug'],
    }

    for task in row['taskResults']:
        name = task['benchmarkTaskName']
        value = task['result'].get('numericResult', {}).get('value')
        flattened_row[name] = value

    rows.append(flattened_row)

df = pd.DataFrame(rows)
score_columns = [
    column
    for column in df.columns
    if column not in ['modelVersionName', 'modelVersionSlug']
]

scores = df[score_columns].copy()
filled_scores = scores.fillna(scores.mean())

scaled_scores = StandardScaler().fit_transform(filled_scores)
dbscan = DBSCAN(eps=0.8, min_samples=2)
df['category'] = dbscan.fit_predict(scaled_scores)
df['power_score'] = scores.mean(axis=1)
df['tier'] = pd.cut(
    df['power_score'],
    bins=[-float('inf'), 500, 1000, float('inf')],
    labels=['weak', 'medium', 'strong'],
)
df['winner'] = df['power_score'] == df['power_score'].max()
df['model_family'] = df['modelVersionName'].apply(get_model_family)

primary_score_column = 'Text Input'

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df[primary_score_column],
    df['power_score'],
    c=df['category'],
    cmap='viridis',
    s=90,
)

for _, row in df.iterrows():
    plt.annotate(
        row['modelVersionName'],
        (row[primary_score_column], row['power_score']),
        fontsize=8,
        xytext=(5, 5),
        textcoords='offset points',
    )

plt.xlabel(f'{primary_score_column} Score')
plt.ylabel('Power Score')
plt.title('AI Model DBSCAN Clusters by Chess Benchmark Score')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('chess_dbscan_clusters.png', dpi=200)

output_columns = [
    'modelVersionName',
    *score_columns,
    'power_score',
    'category',
    'tier',
    'winner',
    'model_family',
]
output_df = df[output_columns].sort_values(
    by=['category', 'power_score'],
    ascending=[True, False],
)
output_df['category'] = output_df['category'].replace({-1: 'Outlier'})

print(output_df.to_string(index=False))

print('Saved graph to chess_dbscan_clusters.png')
