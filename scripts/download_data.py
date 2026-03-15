# scripts/download_all_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

all_datasets = []

# SOURCE 1: Jigsaw Civil Comments (WORKING)
print("Downloading Jigsaw Civil Comments...")
dataset = load_dataset("google/civil_comments", split="train[:20000]")
df1 = pd.DataFrame(dataset)
df1['toxic'] = (df1['toxicity'] > 0.5).astype(int)
df1 = df1[['text', 'toxic']]
df1['source'] = 'jigsaw'
all_datasets.append(df1)
df1.to_csv("data/raw/jigsaw.csv", index=False)
print(f"Jigsaw: {len(df1)} samples")

# SOURCE 2: Hate Speech Dataset (WORKING)
print("Downloading Hate Speech Dataset...")
url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
df2 = pd.read_csv(url)
df2 = df2.rename(columns={'tweet': 'text', 'class': 'label'})
df2['toxic'] = (df2['label'] != 2).astype(int)
df2 = df2[['text', 'toxic']]
df2['source'] = 'hate_speech'
all_datasets.append(df2)
df2.to_csv("data/raw/hate_speech.csv", index=False)
print(f"Hate Speech: {len(df2)} samples")

# SOURCE 3: ParaDetox (FIXED)
print("Downloading ParaDetox...")
dataset = load_dataset("s-nlp/paradetox", split="train[:5000]")
df3 = pd.DataFrame(dataset)
df3_toxic = df3[['en_toxic_comment']].rename(columns={'en_toxic_comment': 'text'})
df3_toxic['toxic'] = 1
df3_neutral = df3[['en_neutral_comment']].rename(columns={'en_neutral_comment': 'text'})
df3_neutral['toxic'] = 0
df3_combined = pd.concat([df3_toxic, df3_neutral])
df3_combined['source'] = 'paradetox'
all_datasets.append(df3_combined)
df3_combined.to_csv("data/raw/paradetox.csv", index=False)
print(f"ParaDetox: {len(df3_combined)} samples")

# Combine all datasets
print("Combining datasets...")
master_df = pd.concat(all_datasets, ignore_index=True)
master_df = master_df.drop_duplicates(subset=['text'])
master_df = master_df.dropna(subset=['text'])
master_df = master_df[master_df['text'].str.len() > 10]

print(f"Total samples: {len(master_df)}")
print(f"Toxic: {len(master_df[master_df['toxic']==1])}")
print(f"Non-toxic: {len(master_df[master_df['toxic']==0])}")

# Save master dataset
master_df.to_csv("data/processed/toxiguard_master.csv", index=False)
