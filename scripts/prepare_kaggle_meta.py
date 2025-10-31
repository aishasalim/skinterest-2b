import argparse, pandas as pd
from pathlib import Path
from src.data.meta_utils import build_meta  # implement using your notebook code

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, help="folder with images")
parser.add_argument("--out",  required=True, help="csv path to write")
args = parser.parse_args()

df = build_meta(Path(args.root))   # -> DataFrame with: image_path, undertone_label, lighting_label, ITA, tone_bucket
df.to_csv(args.out, index=False)
print("Saved:", args.out, "| rows:", len(df))
