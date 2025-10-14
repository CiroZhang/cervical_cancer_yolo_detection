from pathlib import Path
import numpy as np
from ultralytics import YOLO
import pandas as pd
import csv
import matplotlib.pyplot as plt
import math

import json
with open("/project/aip-xli135/jeff418/YOLO/val_visualize_config.jsonl", "r") as f:
    cfg = json.load(f)

nc = cfg["num_classes"]
gen = cfg["generations"]
iou = cfg["iou"]
max_det = cfg["max_det"]
data_yaml = cfg["data_yaml"]
imgsz = cfg["imgsz"]


def run_val(batch_folder: Path, conf: float):
    out_dir = val_result_dir / f"conf_{conf:.2f}" / batch_folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, gen + 1):
        weight_path = batch_folder / f'gen{i}_{nc}cls' / 'weights' / 'best.pt'
        if not weight_path.exists():
            print(f"[skip] missing weights: {weight_path}")
            continue

        model = YOLO(str(weight_path))
        results = model.val(
            data=data_yaml,
            imgsz=imgsz,
            conf=float(conf),
            iou=iou,
            max_det=max_det,
        )

        # Confusion matrix array
        cm = results.confusion_matrix.matrix
        np.savetxt(out_dir / f"gen{i}_{nc}cls.csv", cm.astype(int), fmt="%d", delimiter=",")

        # Copy the PNG Ultralytics saved
        cm_png = Path(results.save_dir) / "confusion_matrix.png"
        if cm_png.exists():
            (out_dir / f"gen{i}_{nc}cls.png").write_bytes(cm_png.read_bytes())

def run_all_vals():
    confs = [round(x, 2) for x in np.arange(0.1, 0.601, 0.05)]  # 0.00 → 0.60 inclusive
    for folder in runs_folder_dir.iterdir():
        if folder.is_dir():
            for conf in confs:
                print(f"[run] {folder.name} @ conf={conf:.2f}")
                run_val(folder, conf)


def cal_confusion_matrix(folder):
    target_folder = val_result_dir / folder
    output_folder = val_result_dir / "numerical_result" / Path(folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_out = output_folder / "trend.csv"

    with open(csv_out, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Gen", "Classes",
                         "Total_Acc",
                         "Total_Background_rate",
                         "Lesion_ACC",
                         "Lesion_Background_rate",
                         "Lesion_incorrect_rate",
                         "NILM_ACC",
                         "NILM_Background_rate",
                         "NILM_incorrect_rate"])

    for i in range(1, gen + 1):
        target_file = target_folder / f'gen{i}_{nc}cls.csv'
        if not target_file.exists():
            print(f"[skip] missing file: {target_file}")
            continue

        df = pd.read_csv(target_file, header=None)
        cm = df.values

        #total acc
        TL_FN_FL = (cm[0,0] + cm[1,1]) / (cm[1,0] + cm[0,1] + cm[0,0] + cm[1,1] + cm[2,1] + cm[2,0]) 
        #total false background rate
        TL_FB_FB = (cm[2,0] + cm[2,1]) / (cm[1,0] + cm[0,1] + cm[0,0] + cm[1,1] + cm[2,1] + cm[2,0]) 
        #lesion acc and backfround rate
        L_acc = (cm[0,0]) / (cm[1,0] + cm[0,0] + cm[2,0])
        L_bg = (cm[2,0]) / (cm[1,0] + cm[0,0] + cm[2,0])
        L_fr = (cm[1,0]) / (cm[1,0] + cm[0,0] + cm[2,0]) #false_rate
        #niml acc and backfround rate
        N_acc = (cm[1,1]) / (cm[1,1] + cm[0,1] + cm[2,1])
        N_bg = (cm[2,1]) / (cm[1,1] + cm[0,1] + cm[2,1])
        N_fr = (cm[0,1]) / (cm[1,1] + cm[0,1] + cm[2,1])


        values = [
            round(TL_FN_FL, 2) if TL_FN_FL != math.inf else "inf",  
            round(TL_FB_FB, 2) if TL_FB_FB != math.inf else "inf",
            round(L_acc, 2) if L_acc != math.inf else "inf",
            round(L_bg, 2) if L_bg != math.inf else "inf",
            round(L_fr, 2) if L_fr != math.inf else "inf",
            round(N_acc, 2) if N_acc != math.inf else "inf",
            round(N_bg, 2) if N_bg != math.inf else "inf",
            round(N_fr, 2) if N_fr != math.inf else "inf"
        ]

        with open(csv_out, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, nc] + values)
    
    try:
        df_best = pd.read_csv(csv_out)

        col_tl_fn_fl = df_best.columns[2]  # TL_FN_FL
        col_tl_fb_fb = df_best.columns[3]  # TL_FB_FB
        col_L_acc    = df_best.columns[4]  # L_acc
        col_L_bg     = df_best.columns[5]  # L_bg
        col_L_fr     = df_best.columns[6]
        col_N_acc    = df_best.columns[7]  # N_acc
        col_N_bg     = df_best.columns[8]  # N_bg
        col_N_fr     = df_best.columns[9]

        # Normalize numeric values
        df_best["_TL_FN_FL"] = pd.to_numeric(df_best[col_tl_fn_fl].replace("inf", math.inf), errors="coerce")
        df_best["_TL_FB_FB"] = pd.to_numeric(df_best[col_tl_fb_fb].replace("inf", math.inf), errors="coerce")
        df_best["_L_acc"]    = pd.to_numeric(df_best[col_L_acc].replace("inf", math.inf), errors="coerce")
        df_best["_L_bg"]     = pd.to_numeric(df_best[col_L_bg].replace("inf", math.inf), errors="coerce")
        df_best["_L_fr"]     = pd.to_numeric(df_best[col_L_fr].replace("inf", math.inf), errors="coerce")
        df_best["_N_acc"]    = pd.to_numeric(df_best[col_N_acc].replace("inf", math.inf), errors="coerce")
        df_best["_N_bg"]     = pd.to_numeric(df_best[col_N_bg].replace("inf", math.inf), errors="coerce")
        df_best["_N_fr"]     = pd.to_numeric(df_best[col_N_fr].replace("inf", math.inf), errors="coerce")


        df_best["Gen"] = pd.to_numeric(df_best["Gen"], errors="coerce").astype("Int64")

        candidates = df_best[
            (df_best["_TL_FN_FL"] >= 0.5) &
            (df_best["_TL_FB_FB"] <= 0.5) &
            (df_best["_L_acc"]    >= 0.37) &
            (df_best["_L_bg"]     <= 0.65) &
            (df_best["_L_fr"]     <= 0.02) &
            (df_best["_N_acc"]    >= 0.61) &
            (df_best["_N_bg"]     <= 0.4) &
            (df_best["_N_fr"]     <= 0.02) 
        ].copy()
        

             #update this part start
        sort_cols = ["_TL_FN_FL", "_L_acc", "_N_acc", "_L_bg", "_N_bg", "_L_fr", "_N_fr", "_TL_FB_FB"]
        ascending = [False,       False,    False,    True,    True,    True,   True,   True]  # max acc, min bg/fr, then lower TL_FB_FB

        csv_best = output_folder / "best_model.csv"
        if csv_best.exists():
            csv_best.unlink()   # always delete old file first

        if candidates.empty:
            print("[best] No model met the criteria — best_model.csv not created")
        else:
            picked_row = candidates.sort_values(
                by=sort_cols,
                ascending=ascending,
                kind="mergesort"
            ).iloc[0]

            best_gen = int(picked_row["Gen"])

            # Write fresh CSV with one row
            with open(csv_best, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["best_generation", col_tl_fn_fl, col_tl_fb_fb])
                writer.writerow([best_gen, picked_row[col_tl_fn_fl], picked_row[col_tl_fb_fb]])

            print(f"[best] gen{best_gen} saved to CSV "
                  f"({col_tl_fn_fl}={picked_row[col_tl_fn_fl]}, {col_tl_fb_fb}={picked_row[col_tl_fb_fb]})")
        #end
    except Exception as e:
        print(f"[best] Failed to compute best model: {e}")

def read_best_model(csv_file):
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"{csv_file} not found")

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:  # expects one row
            return [
                float(row["Total_Acc"]),
                float(row["Total_Background_rate"]),
                float(row["best_generation"]),
                str(csv_file)  # keep the file address as the last element
            ]
    return [] 

def find_best_model_among_folder(folder):
    target_folder = val_result_dir / folder
    target_csv = target_folder / "best_model.csv"
    return read_best_model(target_csv)

def rank_paras(paras):
    # sort: highest tl_fl first, then highest tl_fb
    ranked = sorted(paras, key=lambda p: (p[0], p[1]), reverse=True)
    return ranked


def cal_confusion_matrix_full_process():
    paras = []
    confs = [round(x, 2) for x in np.arange(0.1, 0.601, 0.05)]  # 0.00 → 0.60 inclusive
    for conf in confs:
        target_conf_folder_dir = Path(f"numerical_result/conf_{conf:.2f}")
        only_conf = Path(f"conf_{conf:.2f}")
        for f in runs_folder_dir.iterdir():
            only_conf_folder = only_conf / f.name            
            cal_confusion_matrix(only_conf_folder)
            target_folder = target_conf_folder_dir / f.name
            try:
                para_v = find_best_model_among_folder(target_folder)
                if para_v:      
                    paras.append(para_v)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
            except KeyError as e:
                print(f"[skip] Missing column {e} in {target_folder/'best_model.csv'}")

    ranked = rank_paras(paras)
    out_csv = val_result_dir / "ranked_models.csv" 
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True Label vs False Label", "True Label vs False Background", "best_generation", "CSV Path"])
        writer.writerows(ranked)

    print(f"[done] Saved ranked results to {out_csv}")

    #rank them from top to lowest, protirize tl_fl


def graph_model_performance_trend(seed: int, val_conf: float):
    numerical_result_folder = val_result_dir / "numerical_result"
    target_file = numerical_result_folder / f"conf_{val_conf}" / f"seed{seed}_vc0.30" / "trend.csv"

    df = pd.read_csv(target_file)
    metrics = [col for col in df.columns if col not in ["Gen", "Classes"]]

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(df["Gen"], df[metric], marker='o', label=metric)

    plt.title("Model Performance Trends Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(np.arange(df["Gen"].min(), df["Gen"].max() + 1, 1))
    
    plt.tight_layout()
    plt.savefig(target_file.parent / "trend_plot.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {target_file.parent / 'trend_plot.png'}")




folder_to_check = [
    'pseudo_cycle_sept_14',
    'pseudo_cycle_sept_17_1',
    'pseudo_cycle_sept_17_2',
    'pseudo_cycle_sept_17_3',
    'pseudo_cycle_sept_18_1',
    'pseudo_cycle_sept_18_2',
]


runs_folder_dir = Path('/project/aip-xli135/jeff418/YOLO/runs/random_gen_runs')
val_result_dir = Path('/project/aip-xli135/jeff418/YOLO/val_folder_random_gen')

def main():
    run_all_vals(folder_to_check)

if __name__ == "__main__":
    #run_all_vals()
    #cal_confusion_matrix_full_process()
    graph_model_performance_trend(1390368751, 0.15)
    