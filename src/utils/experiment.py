import os, json, pandas as pd

def create_run_dir(run_name):
    run_dir = f"experiments/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/figures", exist_ok=True)
    return run_dir

def save_history(history, run_dir):
    if history:
        pd.DataFrame(history.history).to_csv(f"{run_dir}/metrics.csv", index=False)

def write_weights_pointer(run_dir, text_line):
    with open(f"{run_dir}/weights.txt","w") as f:
        f.write(text_line.strip()+"\n")
