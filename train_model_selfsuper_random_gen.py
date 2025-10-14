#!/usr/bin/env python3
from ultralytics import YOLO
import os, json, csv, gc, shutil
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import random

# ---------- config ----------
with open("config_random_gen.json", "r") as f:
    CFG = json.load(f)

BASE = CFG["base_dir"].rstrip("/")
IMAGES_TRAIN = f"{BASE}/yolo_data_downscale_{CFG['num_classes']}_class_sept_11/images/train"
IMAGES_VAL   = f"{BASE}/yolo_data_downscale_{CFG['num_classes']}_class_sept_11/images/val"
GT_LABELS_TRAIN = f"{BASE}/yolo_data_downscale_{CFG['num_classes']}_class_sept_11/labels/train"
GT_LABELS_VAL   = f"{BASE}/yolo_data_downscale_{CFG['num_classes']}_class_sept_11/labels/val"
WEIGHTS_INIT = f"{BASE}/{CFG['init_weights']}"
CLASS_NAMES  = CFG["class_names"]

# shorthand
NC          = CFG["num_classes"]
GENS        = CFG["generations"]
CONF_START  = CFG["conf_start"]
CONF_MIN    = CFG["conf_min"]
CONF_DECAY  = CFG["conf_decay"]
MODE        = CFG["mode"]
IOU_KEEP    = CFG["iou_keep"]
NMS_IOU     = CFG["nms_pseudo_iou"]
KEEP_CLASSES= CFG["pseudo_keep"]
PER_CLASS_CONF = CFG["per_class_conf"]
PER_CLASS_TOPK = CFG["per_class_topk"]


# training params (just forwarded to ultralytics)
TRAIN_ARGS  = CFG.get("train_args", {})
VAL_ARGS    = CFG.get("val_args", {})

# ---------- utils ----------
def d(p): 
    os.makedirs(p, exist_ok=True)

def link(src,dst):
    if os.path.islink(dst) or os.path.exists(dst):
        try:
            if os.path.islink(dst) and os.readlink(dst)==src: return
        except OSError: pass
        shutil.rmtree(dst) if (os.path.isdir(dst) and not os.path.islink(dst)) else os.unlink(dst)
    os.symlink(src,dst)

def read_txt(p):
    if not os.path.exists(p): return []
    with open(p) as f: return [ln.strip() for ln in f if ln.strip()]

def write_txt(p,lines):
    with open(p,"w") as f:
        if lines: f.write("\n".join(sorted(set(lines)))+"\n")

def to_scalar(x):
    if x is None: return None
    if hasattr(x,"item"):
        try: return float(x.item())
        except: pass
    if isinstance(x,(list,tuple,np.ndarray)):
        if len(x)==0: return None
        return float(np.nanmean(x))
    try: return float(x)
    except: return None

def parse_line(ln):
    s=ln.split()
    if len(s)<5: return None
    c=int(float(s[0])); x,y,w,h=map(float,s[1:5]); conf=float(s[5]) if len(s)>=6 else None
    return c,x,y,w,h,conf

def xywhn2xyxy(x,y,w,h): 
    return x-w/2,y-h/2,x+w/2,y+h/2

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0
    return inter/max(1e-12,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

def nms(lines,thr=0.5):
    parsed=[]
    for ln in lines:
        p=parse_line(ln); 
        if not p: continue
        c,x,y,w,h,conf=p; xyxy=xywhn2xyxy(x,y,w,h)
        parsed.append((c,x,y,w,h,1.0 if conf is None else conf,xyxy))
    out=[]
    byc={}
    for it in parsed: byc.setdefault(it[0],[]).append(it)
    for c,items in byc.items():
        items.sort(key=lambda t:t[5],reverse=True); keep=[]
        for cand in items:
            if all(iou(cand[6],k[6])<thr for k in keep): keep.append(cand)
        for _,x,y,w,h,_,_ in keep: out.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return out

def keep_ids():
    if KEEP_CLASSES=="*" or KEEP_CLASSES==["*"]: return set(range(len(CLASS_NAMES)))
    return {CLASS_NAMES.index(nm) for nm in KEEP_CLASSES if nm in CLASS_NAMES}

KEEP_IDS=keep_ids()

# ---------- merge ----------
def merge(gt_dir,pl_dir,out_dir,prev_dir=None):
    d(out_dir)
    gt_files={f for f in os.listdir(gt_dir) if f.endswith(".txt")}
    cur_files=set(os.listdir(pl_dir)) if pl_dir and os.path.exists(pl_dir) else set()
    prev_files=set(os.listdir(prev_dir)) if prev_dir and os.path.exists(prev_dir) else set()
    all_files=gt_files|cur_files|(prev_files if MODE=="cumulative" else set())

    for fn in all_files:
        out=[]; gt=read_txt(os.path.join(gt_dir,fn)) if fn in gt_files else []
        out+=gt
        cur_for_match=[]; cur_out=[]
        if fn in cur_files:
            raw=read_txt(os.path.join(pl_dir,fn)); buckets={}
            for ln in raw:
                p=parse_line(ln); 
                if not p: continue
                cid,x,y,w,h,conf=p
                if cid not in KEEP_IDS: continue
                cname=CLASS_NAMES[cid]
                if cname in PER_CLASS_CONF and conf is not None and conf<float(PER_CLASS_CONF[cname]): continue
                xyxy=xywhn2xyxy(x,y,w,h)
                line=f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                buckets.setdefault(cid,[]).append((1.0 if conf is None else conf,line,xyxy))
            for cid,items in buckets.items():
                cname=CLASS_NAMES[cid]; topk=int(PER_CLASS_TOPK.get(cname,0))
                items.sort(key=lambda t:t[0],reverse=True)
                keep=items if topk<=0 else items[:topk]
                for conf,line,xyxy in keep: cur_out.append(line); cur_for_match.append((cid,xyxy))
            if NMS_IOU>0: cur_out=nms(cur_out,NMS_IOU)
        if MODE=="cumulative" and fn in prev_files:
            for ln in read_txt(os.path.join(prev_dir,fn)):
                if ln in gt: out.append(ln); continue
                p=parse_line(ln); 
                if not p: continue
                pc,x,y,w,h,_=p; pxy=xywhn2xyxy(x,y,w,h)
                if any(c==pc and iou(pxy,xy)>=IOU_KEEP for c,xy in cur_for_match): out.append(ln)
        out+=cur_out
        write_txt(os.path.join(out_dir,fn),out)

# ---------- summary ----------
def summarize(label_dir,run_dir,gen,history_csv):
    cnt=Counter()
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"): continue
        for ln in read_txt(os.path.join(label_dir,f)):
            try: cid=int(ln.split()[0]); cnt[cid]+=1
            except: pass
    vals=[cnt[i] for i in range(len(CLASS_NAMES))]
    hdr=not os.path.exists(history_csv)
    with open(history_csv,"a",newline="") as cf:
        w=csv.DictWriter(cf,fieldnames=["gen",*CLASS_NAMES,"total"])
        if hdr: w.writeheader()
        w.writerow({"gen":gen,**{CLASS_NAMES[i]:vals[i] for i in range(len(CLASS_NAMES))},"total":sum(vals)})
    gens,totals,series=[],[],{c:[] for c in CLASS_NAMES}
    with open(history_csv) as cf:
        r=csv.DictReader(cf)
        for row in r:
            gens.append(int(row["gen"])); totals.append(int(row["total"]))
            for c in CLASS_NAMES: series[c].append(int(row[c]))
    plt.figure(figsize=(7,4)); plt.plot(gens,totals,marker="o"); plt.title("Total count"); plt.xlabel("Gen"); plt.ylabel("Total"); plt.grid(True,alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RUNS_ROOT,"total_counts.png")); plt.close()
    plt.figure(figsize=(8,5))
    for c in CLASS_NAMES: plt.plot(gens,series[c],marker="o",label=c)
    plt.title("Per-class counts"); plt.xlabel("Gen"); plt.ylabel("Count"); plt.legend(); plt.grid(True,alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RUNS_ROOT,"class_counts.png")); plt.close()

# ---------- dataset view ----------
def make_view(gen,merged_dir):
    root=f"{PSEUDO_ROOT}/gen{gen}"
    train=f"{root}/train"
    imgs=f"{train}/images"
    lbls=f"{train}/labels"

    d(train); d(lbls); link(IMAGES_TRAIN,imgs)

    for fn in os.listdir(lbls):
        if fn.endswith(".txt"): os.remove(os.path.join(lbls,fn))
    for fn in os.listdir(merged_dir):
        if fn.endswith(".txt"): shutil.copy2(os.path.join(merged_dir,fn),os.path.join(lbls,fn))
    yaml=f"{root}/dataset.yaml"
    with open(yaml,"w") as f:
        f.write(f"train: {imgs}\nval: {IMAGES_VAL}\nnc: {NC}\nnames: {json.dumps(CLASS_NAMES)}\n")
    return yaml

# ---------- run one gen ----------
def run_gen(gen, data_yaml, hist_csv):
    run = f"{RUNS_ROOT}/gen{gen}_{NC}cls"; d(run)
    conf = max(CONF_MIN, CONF_START - (gen - 1) * CONF_DECAY)

    # train
    #model = YOLO(WEIGHTS_INIT)
    init = WEIGHTS_INIT if gen == 1 else os.path.join(
        RUNS_ROOT, f"gen{gen-1}_{NC}cls", "weights", "best.pt"
    )
    model = YOLO(init)
    
    model.train(**{**TRAIN_ARGS, "data": data_yaml, "seed": seed}, project=RUNS_ROOT, name=os.path.basename(run), exist_ok=True)
    best = os.path.join(run, "weights", "best.pt")

    # validate (drives confusion matrix)
    metrics = model.val(
        data=data_yaml,
        imgsz=TRAIN_ARGS.get("imgsz", 256),
        conf=val_conf,
        iou=VAL_ARGS.get("iou", 0.50),
        max_det=VAL_ARGS.get("max_det", 2000),
        project=RUNS_ROOT, name=os.path.basename(run), exist_ok=True
    )
    with open(os.path.join(run, "val_metrics.json"), "w") as f:
        json.dump({
            "map50_95": to_scalar(getattr(metrics.box, "map", None)),
            "map50":    to_scalar(getattr(metrics.box, "map50", None)),
            "precision":to_scalar(getattr(metrics.box, "p", None)),
            "recall":   to_scalar(getattr(metrics.box, "r", None)),
            "val_conf": val_conf,
            "val_iou":  VAL_ARGS.get("iou", 0.50)
        }, f, indent=2)

    # optional: copy/rename CM so threshold is visible
    cm_src = os.path.join(run, "confusion_matrix.png")
    if os.path.exists(cm_src):
        cm_dst = os.path.join(run, f"confusion_matrix_conf{VAL_ARGS.get('conf',0.25):.2f}.png")
        try: shutil.copy2(cm_src, cm_dst)
        except Exception: pass

    # predict pseudo
    pl_root = f"{run}/pseudo_labels"; d(pl_root)
    YOLO(best).predict(source=IMAGES_TRAIN, conf=conf, save_txt=True, save_conf=True,
                       project=pl_root, name="preds", exist_ok=True, verbose=False)
    pl_labels = os.path.join(pl_root, "preds", "labels")

    # merge (class-aware, cumulative + pruning)
    prev = f"{PSEUDO_ROOT}/gen{gen-1}_merged" if (MODE == "cumulative" and gen > 1) else None
    merged = f"{PSEUDO_ROOT}/gen{gen}_merged"; d(merged)
    merge(GT_LABELS_TRAIN, pl_labels, merged, prev_dir=prev)

    summarize(merged, run, gen, hist_csv)
    next_yaml = make_view(gen, merged)

    del model; torch.cuda.empty_cache(); gc.collect()
    return next_yaml

# ---------- main ----------
def main():
    global RUNS_ROOT, PSEUDO_ROOT, seed, val_conf
    for i in range(40):  # 20 experiments

        # randomize only these two
        seed = random.randint(0, 2**31 - 1)        # integer seed
        val_conf = 0.3   # float in [0.1, 0.7]

        # folder names now tied only to seed (and val_conf if you want)
        RUNS_ROOT   = f"{BASE}/runs/random_gen_runs/seed{seed}_vc{val_conf:.2f}"
        PSEUDO_ROOT = f"{BASE}/random_gen_datasets_1/seed{seed}_vc{val_conf:.2f}"

        d(RUNS_ROOT); d(PSEUDO_ROOT)

        data_yaml = f"{BASE}/yolo_data_downscale_{NC}_class_sept_11/dataset.yaml"
        hist = os.path.join(RUNS_ROOT, "counts_history.csv")

        for g in range(1, GENS+1):
            data_yaml = run_gen(g, data_yaml, hist)

        

    

if __name__=="__main__": main()
