import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/toxiguard_master_cleaned.csv")
min_samples = min(len(df[df['toxic']==1]), len(df[df['toxic']==0]))

toxic = df[df['toxic']==1].sample(n=min_samples, random_state=42)
nontoxic = df[df['toxic']==0].sample(n=min_samples, random_state=42)
balanced = pd.concat([toxic, nontoxic]).sample(frac=1, random_state=42)

train, test = train_test_split(balanced, test_size=0.2, random_state=42, stratify=balanced['toxic'])
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"Train: {len(train)} Test: {len(test)}")
