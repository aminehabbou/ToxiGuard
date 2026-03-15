import pandas as pd
from pathlib import Path

df = pd.concat([pd.read_csv(f) for f in Path("data/raw").glob("*.csv")], ignore_index=True)
df = df.dropna(subset=['text', 'toxic'])
df = df[df['text'].str.len() >= 10].drop_duplicates(subset=['text'])
df.to_csv("data/processed/toxiguard_master_cleaned.csv", index=False)
print(f"Done: {len(df)} samples")