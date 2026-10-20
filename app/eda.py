"""Generate EDA plots for the UCI Cleveland heart-disease dataset."""
import argparse
from pathlib import Path
from train import load_data, save_eda
p=argparse.ArgumentParser(); p.add_argument("--data", default=None); a=p.parse_args()
df=load_data(a.data); print(df.describe(include="all").to_string()); print("Missing values by column:\n",df.isna().sum().to_string()); print("Target distribution:\n",df.target.value_counts(normalize=True).to_string()); save_eda(df,Path(__file__).resolve().parents[1]/"reports"/"figures")
