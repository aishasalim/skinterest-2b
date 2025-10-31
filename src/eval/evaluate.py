import json, numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate_and_report(model, test_ds, test_df, classes, run_dir, lighting_auc=None):
    light_probs, under_probs = model.predict(test_ds, verbose=0)

    y_light = test_df["lighting_label"].astype(int).values
    light_bin = (light_probs.ravel() >= 0.5).astype(int)
    lighting_acc = float(accuracy_score(y_light, light_bin))

    y_under = test_df["undertone_id"].astype(int).values
    under_ids = np.argmax(under_probs, axis=1)
    under_acc = float((under_ids == y_under).mean())

    # per-class + macro
    per_class = []
    for k in range(len(classes)):
        idx = (y_under == k)
        if idx.sum() == 0: continue
        per_class.append(float((under_ids[idx] == y_under[idx]).mean()))
    macro = float(np.mean(per_class)) if per_class else 0.0

    report = {
      "results": {
        "lighting": {"test_acc": lighting_acc, "auc": float(lighting_auc) if lighting_auc else None},
        "diagnosis": {"test_acc": under_acc, "macro_acc": macro}
      },
      "labels": {i: c for i, c in enumerate(classes)}
    }
    with open(f"{run_dir}/report.json","w") as f: json.dump(report, f, indent=2)
    return report
