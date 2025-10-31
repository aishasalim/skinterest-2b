import argparse, yaml, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models.multitask import build_multitask
from src.data.datasets import make_ds, balanced_by
from src.training.abc import run_phases
from src.utils.experiment import create_run_dir, save_history, write_weights_pointer
from src.eval.evaluate import evaluate_and_report

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--meta",   required=True)
parser.add_argument("--run",    required=True)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
df  = pd.read_csv(args.meta)

df = df.dropna(subset=["image_path","undertone_label","lighting_label"]).copy()
df["undertone_label"] = df["undertone_label"].astype(str).str.strip().str.lower()
df["lighting_label"]  = df["lighting_label"].astype(int)

le = LabelEncoder()
df["undertone_id"] = le.fit_transform(df["undertone_label"])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["undertone_id"])
val_df,   test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df["undertone_id"])

train_bal = balanced_by(train_df, mode=cfg["balance_mode"], seed=42, cap=cfg.get("upsample_cap"))

train_ds = make_ds(train_bal, img_size=tuple(cfg["img_size"]), frac=cfg["center_crop_frac"], batch=cfg["batch"], training=True)
val_ds   = make_ds(val_df,    img_size=tuple(cfg["img_size"]), frac=cfg["center_crop_frac"], batch=cfg["batch"], training=False)
test_ds  = make_ds(test_df,   img_size=tuple(cfg["img_size"]), frac=cfg["center_crop_frac"], batch=cfg["batch"], training=False)

model = build_multitask(img_size=tuple(cfg["img_size"]), drop=cfg["dropout"],
                        num_classes=len(le.classes_), ccm_reg=cfg["ccm_reg"])

ckpt = tf.keras.callbacks.ModelCheckpoint("best.weights.h5", save_best_only=True, save_weights_only=True,
                                          monitor="val_undertone_out_acc", mode="max")
es   = tf.keras.callbacks.EarlyStopping(monitor="val_undertone_out_acc", mode="max", patience=4, restore_best_weights=True)
rlr  = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_undertone_out_acc", mode="max", factor=0.5, patience=2, verbose=1)

hist_A, hist_B, hist_C = run_phases(model, train_ds, val_ds, epochs=tuple(cfg["epochs_abc"]), callbacks=[ckpt, es, rlr])

run_dir = create_run_dir(args.run)
save_history(hist_C, run_dir)

# (optional) save full .keras locally but DO NOT commit it
model.save(f"{run_dir}/{args.run}.keras")
write_weights_pointer(run_dir, f"(local) experiments/{args.run}/{args.run}.keras  # upload to HF/GitHub Release if needed")

report = evaluate_and_report(model, test_ds, test_df, list(le.classes_), run_dir)
print("Done. See:", run_dir)
