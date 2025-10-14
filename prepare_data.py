import argparse
from slide_data import *
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
args = parser.parse_args()

num_of_class = args.num_classes

sd = SlideDataList(
    ["075", "099", "156", "450", "470", "485", "501", "511", "514", "517",
     "518", "519", "527", "598", "600", "607", "617", "624", "630", "639", "648"],
    num_of_class
)

sd.cut_patches(PATCH_W=1024, PATCH_H=1204) 

sd.set_boxes(iou_thresh=0.0, min_overlap_px=0, min_overlap_frac=0.40)

summary = sd.export_all_patches_yolo(
    out_dir=f"patch_exports_downscale_{num_of_class}_class",
    class_map=None,
    preview_scale=1/8,  
    image_scale=1/8,   
    yolo_dirs=True,
    include_empty=False,
    auto_build_class_map=True,
)


Path(f"yolo_data_{num_of_class}_class").mkdir(parents=True, exist_ok=True)

stats = yolo_train_val_split_to(f"patch_exports_downscale_{num_of_class}_class", f"yolo_data_downscale_{num_of_class}_class", val_frac=0.2)