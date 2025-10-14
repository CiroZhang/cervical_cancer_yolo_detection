# type: ignore - this is for pylance

import os, shutil, json, random, openslide, matplotlib
from ultralytics import YOLO
import pandas as pd
import numpy as np

from pathlib import Path

import torch
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -------------------------- Multi-slide thin wrapper -------------------------- #
class SlideDataList:
    """
    Minimal wrapper around multiple SlideData objects with the same API:
      - cut_patches(PATCH_W, PATCH_H)
      - set_boxes(iou_thresh, min_overlap_px, min_overlap_frac)
      - export_all_patches_yolo(...): flat export to a single folder; filenames are prefixed by slide id
    """

    def __init__(self, slide_ids, num_of_class):
        self.slides = [SlideData(sid, num_of_class) for sid in slide_ids]
        self.num_of_class = num_of_class

    def cut_patches(self, PATCH_W, PATCH_H):
        for s in self.slides:
            s.cut_patches(PATCH_W, PATCH_H)

    def set_boxes(self, iou_thresh=0.0, min_overlap_px=0, min_overlap_frac=0.0):
        for s in self.slides:
            s.set_boxes(iou_thresh=iou_thresh,
                        min_overlap_px=min_overlap_px,
                        min_overlap_frac=min_overlap_frac)

    def _build_global_class_map(self):
        # collect all labels from already-matched boxes across slides
        labels = []
        for s in self.slides:
            for b in (s.bound_boxes or []):
                labels.extend(b.get("labels", []))
        return {lbl: i for i, lbl in enumerate(sorted(set(labels)))}

    def export_all_patches_yolo(self, out_dir="patch_exports",
                                class_map=None, preview_scale=0.5,
                                yolo_dirs=True, include_empty=False,
                                auto_build_class_map=True,
                                manifest_name="manifest.csv",
                                *,
                                # NEW: downsize saved patch images by a factor (e.g., 0.5 == 50%)
                                image_scale: float = 1.0):
        """
        FLAT EXPORT: all slides into the SAME out_dir (with subdirs images/labels/previews if yolo_dirs=True).
        Filenames are prefixed with the slide id, e.g. s075_patch_00012.jpg.
        Writes one combined manifest and a global class_map.json in out_dir.

        image_scale affects the SAVED patch images (labels are normalized, so still correct).
        """
        os.makedirs(out_dir, exist_ok=True)

        # global class map (stable across all slides)
        if class_map is None and auto_build_class_map:
            class_map = self._build_global_class_map()

        with open(os.path.join(out_dir, "class_map.json"), "w") as f:
            json.dump(class_map or {}, f, indent=2)

        total_imgs = 0
        total_labels = 0
        manifests = []

        for s in self.slides:
            # robust slide-id prefix: zero-pad if numeric
            sid_str = str(s.slide_id)
            try:
                sid_int = int(sid_str)
                sid_prefix = f"s{sid_int:03d}_"
            except ValueError:
                sid_prefix = f"s{sid_str}_"

            # export this slide into the SAME out_dir with a filename prefix
            res = s.export_all_patches_yolo(
                out_dir=out_dir,
                class_map=class_map,
                preview_scale=preview_scale,
                yolo_dirs=yolo_dirs,
                include_empty=include_empty,
                auto_build_class_map=False,  # already built globally
                manifest_name=f"manifest_slide_{s.slide_id}.csv",
                name_prefix=sid_prefix,      # <-- key: prefix filenames
                image_scale=image_scale,     # <-- NEW
            )
            total_imgs += res.get("images", 0)
            total_labels += res.get("labels", 0)

            # load per-slide manifest, ensure slide column exists
            try:
                m = pd.read_csv(res["manifest"])
                if "slide" not in m.columns:
                    m.insert(0, "slide", s.slide_id)
                manifests.append(m)
            except Exception:
                pass

        # write combined manifest at root
        combined = (pd.concat(manifests, ignore_index=True)
                    if manifests else pd.DataFrame(
            columns=["slide", "index", "image", "label", "preview", "num_labels"]
        ))
        manifest_path = os.path.join(out_dir, manifest_name)
        combined.to_csv(manifest_path, index=False)

        return {
            "images": int(total_imgs),
            "labels": int(total_labels),
            "manifest": manifest_path,
            "class_map": class_map or {}
        }

    def close(self):
        for s in self.slides:
            try:
                s.close()
            except Exception:
                pass

    # optional: context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ------------------------------- Single-slide core ---------------------------- #
class SlideData:
    def __init__(self, slide_number, num_of_class):
        svs_path = f"raw_data/images/{slide_number}.svs"
        boundaries_csv = f"raw_data/boundaries(csv)_{num_of_class}_class/{slide_number}.csv"
        patches_csv = f"raw_data/patches(csv)/{slide_number}.csv"

        self.slide_id = slide_number

        self.boundaries = pd.read_csv(boundaries_csv)
        self.patches = pd.read_csv(patches_csv)
        self.slide = openslide.OpenSlide(svs_path)

        self.W0, self.H0 = self.slide.dimensions

        self.patch_w = None
        self.patch_h = None
        self.patch_list = []  # [{x,y,w,h,corners, boxes:[...]}]

        # Each bound box stores: {"x","y","w","h","labels":[...]}
        self.bound_boxes = []
        self.patch_list_with_box = []
        self.patch_list_without_box = []

    def cut_patches(self, PATCH_W, PATCH_H, use_csv=False):
        self.patch_list = []
        self.patch_w, self.patch_h = int(PATCH_W), int(PATCH_H)

        if use_csv:
            if not {"x_tl", "y_tl"}.issubset(self.patches.columns):
                raise ValueError("patches CSV must contain columns: x_tl, y_tl")

            for _, r in self.patches.iterrows():
                x0, y0 = int(r["x_tl"]), int(r["y_tl"])
                w, h = self.patch_w, self.patch_h
                corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
                self.patch_list.append({
                    "x": x0, "y": y0, "w": w, "h": h,
                    "corners": corners, "boxes": []
                })
        else:
            W, H = self.W0, self.H0
            pw, ph = self.patch_w, self.patch_h

            def make_positions(L, p):
                """
                Return a list of start positions so that:
                  - first = 0
                  - last = L - p (clamped to >= 0)
                  - equal spacing => evenly distributed overlap when L % p != 0
                """
                if L <= p:
                    # Single position; patch will start at 0 (patch size stays constant)
                    return [0]
                n = int(np.ceil(L / p))  # number of patches along this axis
                xs = np.linspace(0, L - p, n)  # float positions, inclusive of end
                xs = [int(round(v)) for v in xs]
                xs[0] = 0
                xs[-1] = L - p
                # de-duplicate while preserving order and keep within [0, L-p]
                pos = []
                for v in xs:
                    v = 0 if v < 0 else (L - p if v > (L - p) else v)
                    if not pos or v != pos[-1]:
                        pos.append(v)
                return pos

            xs = make_positions(W, pw)
            ys = make_positions(H, ph)

            for y0 in ys:
                for x0 in xs:
                    w, h = pw, ph
                    corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
                    self.patch_list.append({
                        "x": int(x0), "y": int(y0), "w": w, "h": h,
                        "corners": corners, "boxes": []
                    })

    @staticmethod
    def _split_labels(raw):
        """Accept 'a;b', 'a,b', 'a|b', list-like, or single string; return list[str]."""
        if isinstance(raw, (list, tuple, set)):
            return [str(x).strip() for x in raw if str(x).strip()]
        s = str(raw)
        for sep in [';', ',', '|']:
            if sep in s:
                return [t.strip() for t in s.split(sep) if t.strip()]
        s = s.strip()
        return [s] if s else []

    def set_boxes(self, iou_thresh=0.0, min_overlap_px=0, min_overlap_frac=0.0):
        """
        Match annotation boxes to patches, then filter *after* matching.
        - iou_thresh: minimum IoU to count as a match (0.0 = any intersection)
        - min_overlap_px: minimum pixels of the box that must fall inside the patch
        - min_overlap_frac: minimum fraction of the box area that must fall inside the patch
        Supports multiple labels per box (boundaries['type'] can be 'a;b', 'a,b', 'a|b', list, or single).
        """
        need = {"type", "boundary_center_x", "boundary_center_y", "height", "width"}
        if not need.issubset(self.boundaries.columns):
            raise ValueError(f"boundaries CSV must contain: {need}")

        # Build absolute boxes (no pre-size filtering)
        bx = self.boundaries["boundary_center_x"].astype(float).to_numpy()
        by = self.boundaries["boundary_center_y"].astype(float).to_numpy()
        bw = self.boundaries["width"].astype(float).to_numpy()
        bh = self.boundaries["height"].astype(float).to_numpy()
        raw_types = self.boundaries["type"].to_list()

        bx0 = bx - bw / 2.0
        by0 = by - bh / 2.0

        self.bound_boxes = [{
            "x": float(x0), "y": float(y0),
            "w": float(w), "h": float(h),
            "labels": self._split_labels(lbl)
        } for x0, y0, w, h, lbl in zip(bx0, by0, bw, bh, raw_types)]

        # Reset per-patch boxes
        for p in self.patch_list:
            p["boxes"] = []

        B = len(self.bound_boxes)
        if B == 0:
            self.patch_list_with_box = []
            self.patch_list_without_box = list(range(len(self.patch_list)))
            return

        # Vectorize boxes
        BB_x0 = np.array([b["x"] for b in self.bound_boxes], dtype=float)
        BB_y0 = np.array([b["y"] for b in self.bound_boxes], dtype=float)
        BB_x1 = BB_x0 + np.array([b["w"] for b in self.bound_boxes], dtype=float)
        BB_y1 = BB_y0 + np.array([b["h"] for b in self.bound_boxes], dtype=float)
        BB_area = (BB_x1 - BB_x0) * (BB_y1 - BB_y0)

        self.patch_list_with_box = []
        self.patch_list_without_box = []

        for idx, p in enumerate(self.patch_list):
            px0, py0, pw, ph = p["x"], p["y"], p["w"], p["h"]
            px1, py1 = px0 + pw, py0 + ph

            # coarse overlap
            cand = (BB_x1 > px0) & (BB_x0 < px1) & (BB_y1 > py0) & (BB_y0 < py1)
            if not np.any(cand):
                self.patch_list_without_box.append(idx)
                continue

            # exact intersection
            cx0 = np.maximum(BB_x0[cand], px0)
            cy0 = np.maximum(BB_y0[cand], py0)
            cx1 = np.minimum(BB_x1[cand], px1)
            cy1 = np.minimum(BB_y1[cand], py1)

            inter_w = np.maximum(0.0, cx1 - cx0)
            inter_h = np.maximum(0.0, cy1 - cy0)
            inter_area = inter_w * inter_h

            # IoU (optional) for compatibility
            patch_area = pw * ph
            boxes_area = BB_area[cand]
            union = patch_area + boxes_area - inter_area
            iou = np.where(union > 0, inter_area / union, 0.0)

            # post-match filters
            frac_box_in_patch = np.where(boxes_area > 0, inter_area / boxes_area, 0.0)

            keep_mask = (iou > float(iou_thresh)) & \
                        (inter_area >= float(min_overlap_px)) & \
                        (frac_box_in_patch >= float(min_overlap_frac))

            if not np.any(keep_mask):
                self.patch_list_without_box.append(idx)
                continue

            # add kept boxes
            kept_indices_global = np.nonzero(cand)[0][keep_mask]
            kept_cx0 = cx0[keep_mask]; kept_cy0 = cy0[keep_mask]
            kept_cx1 = cx1[keep_mask]; kept_cy1 = cy1[keep_mask]
            kept_iou = iou[keep_mask]
            kept_inter_area = inter_area[keep_mask]
            kept_frac = frac_box_in_patch[keep_mask]

            for j_g, kx0, ky0, kx1, ky1, kiou, karea, kfrac in zip(
                    kept_indices_global, kept_cx0, kept_cy0, kept_cx1, kept_cy1,
                    kept_iou, kept_inter_area, kept_frac
            ):
                # patch-local clipped xywh
                rx = float(kx0 - px0)
                ry = float(ky0 - py0)
                rw = float(max(0.0, kx1 - kx0))
                rh = float(max(0.0, ky1 - ky0))

                p["boxes"].append({
                    "global": self.bound_boxes[int(j_g)],  # contains "labels":[...]
                    "local_xywh": (rx, ry, rw, rh),
                    "iou": float(kiou),
                    "inter_area": float(karea),
                    "frac_box_in_patch": float(kfrac)
                })

            self.patch_list_with_box.append(idx)

    def export_patch_yolo(self, i, out_dir="patch_exports", class_map=None,
                          preview_scale=0.5, yolo_dirs=True, name_prefix="",
                          *,
                          # NEW: downsize saved patch image (not the preview) by factor, e.g., 0.5
                          image_scale: float = 1.0):
        """
        Save:
          - raw patch image as JPEG (optionally downsized by image_scale)
          - YOLO txt file with boxes (one line per label for multi-label boxes)
          - reduced-size JPEG with boxes drawn + label text (controlled by preview_scale)

        yolo_dirs=True -> organize into out_dir/images, out_dir/labels, out_dir/previews
        name_prefix -> prepended to filenames (useful when exporting many slides to one folder)

        NOTE: YOLO labels are normalized; resizing the saved image does NOT require label changes.
        """
        # dirs
        if yolo_dirs:
            images_dir = os.path.join(out_dir, "images")
            labels_dir = os.path.join(out_dir, "labels")
            previews_dir = os.path.join(out_dir, "previews")
            for d in (images_dir, labels_dir, previews_dir):
                os.makedirs(d, exist_ok=True)
        else:
            images_dir = labels_dir = previews_dir = out_dir
            os.makedirs(out_dir, exist_ok=True)

        p = self.patch_list[i]
        w, h = p["w"], p["h"]

        base = f"{name_prefix}patch_{i:05d}"

        # raw image (optionally scaled down for saving)
        img = self.slide.read_region((p["x"], p["y"]), 0, (w, h)).convert("RGB")
        if image_scale is not None and image_scale > 0 and image_scale != 1.0:
            sf = max(0.01, float(image_scale))
            new_w = max(1, int(round(img.width * sf)))
            new_h = max(1, int(round(img.height * sf)))
            img_to_save = img.resize((new_w, new_h), Image.BILINEAR)
        else:
            img_to_save = img

        img_path = os.path.join(images_dir, f"{base}.jpg")
        img_to_save.save(img_path, quality=95)

        # YOLO labels (multi-label -> multiple lines per box)
        yolo_lines = []
        for b in p["boxes"]:
            rx, ry, rw, rh = b["local_xywh"]
            if rw <= 0 or rh <= 0:
                continue
            cx = (rx + rw / 2) / w
            cy = (ry + rh / 2) / h
            nw = rw / w
            nh = rh / h

            labels_here = b["global"].get("labels", [])
            if class_map is None:
                # if no map, default to a single class 0 to stay valid
                labels_here = labels_here or ["0"]
            for lbl in labels_here:
                cls = class_map.get(lbl, 0) if class_map else 0
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        txt_path = os.path.join(labels_dir, f"{base}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # preview with boxes + label text (independent of image_scale)
        preview_w = max(1, int(w * preview_scale))
        preview_h = max(1, int(h * preview_scale))
        preview_img = img.resize((preview_w, preview_h), Image.BILINEAR)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(preview_img)
        ax.axis("off")
        for b in p["boxes"]:
            rx, ry, rw, rh = b["local_xywh"]
            if rw <= 0 or rh <= 0:
                continue
            x_draw = rx * preview_scale
            y_draw = ry * preview_scale
            ax.add_patch(Rectangle((x_draw, y_draw),
                                   rw * preview_scale, rh * preview_scale,
                                   fill=False, linewidth=1))
            # label text: join multi-labels, prefer names even if class_map exists
            lbl_txt = ",".join(b["global"].get("labels", [])) or "cls0"
            ax.text(x_draw + 2, y_draw + 10, lbl_txt,
                    fontsize=8, va="top",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

        preview_path = os.path.join(previews_dir, f"{base}_preview.jpg")
        plt.savefig(preview_path, bbox_inches="tight", dpi=150)
        plt.close()

        return img_path, txt_path, preview_path, len(yolo_lines)

    def export_all_patches_yolo(self, out_dir="patch_exports", class_map=None, preview_scale=0.5,
                                yolo_dirs=True, include_empty=False, manifest_name="manifest.csv",
                                auto_build_class_map=True, name_prefix="",
                                *,
                                # NEW: pass through to export_patch_yolo
                                image_scale: float = 1.0):
        """
        Export many patches at once.
        - include_empty: if False, only export patches that have at least one box
        - auto_build_class_map: if True and class_map is None, build from *all* labels across boxes
        - name_prefix: prepended to filenames (useful for multi-slide combined export)
        - image_scale: downsize saved patch images by factor (labels unaffected)
        Returns dict with summary counts and writes a CSV manifest.
        """
        # auto-build class map if requested
        if class_map is None and auto_build_class_map:
            all_labels = []
            for b in (self.bound_boxes or []):
                all_labels.extend(b.get("labels", []))
            labels = sorted(set(all_labels))
            class_map = {lbl: i for i, lbl in enumerate(labels)}

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "class_map.json"), "w") as f:
            json.dump(class_map or {}, f, indent=2)

        indices = range(len(self.patch_list)) if include_empty else self.patch_list_with_box

        rows = []
        total_imgs = total_labels = 0
        for i in indices:
            img_p, txt_p, prev_p, nlab = self.export_patch_yolo(
                i, out_dir=out_dir, class_map=class_map,
                preview_scale=preview_scale, yolo_dirs=yolo_dirs,
                name_prefix=name_prefix,
                image_scale=image_scale,  # <-- NEW
            )
            rows.append({
                "index": i,
                "image": img_p,
                "label": txt_p,
                "preview": prev_p,
                "num_labels": nlab
            })
            total_imgs += 1
            total_labels += nlab

        manifest_path = os.path.join(out_dir, manifest_name)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        return {
            "images": total_imgs,
            "labels": total_labels,
            "manifest": manifest_path,
            "class_map": class_map
        }

    def save_patches_overlay(self, out_path="patches_overlay.jpg",
                             max_dim=2000, line_width=0.8):
        """
        Save a whole-slide thumbnail with all *patch* rectangles overlaid.
        Requires cut_patches(...) to have been called.
        """
        if not self.patch_list:
            raise ValueError("No patches found. Call cut_patches(...) first.")

        # choose a scale that fits the longer side to max_dim
        scale = min(max_dim / float(self.W0), max_dim / float(self.H0))
        Wt, Ht = max(1, int(self.W0 * scale)), max(1, int(self.H0 * scale))

        thumb = self.slide.get_thumbnail((Wt, Ht)).convert("RGB")

        fig_w, fig_h, dpi = Wt / 100.0, Ht / 100.0, 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(thumb)
        ax.axis("off")

        for p in self.patch_list:
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            ax.add_patch(Rectangle((x * scale, y * scale),
                                   w * scale, h * scale,
                                   fill=False, linewidth=line_width))

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return out_path

    def save_annotations_overlay(self, out_path="annotations_overlay.jpg",
                                 max_dim=2000, line_width=0.8):
        """
        Save a whole-slide thumbnail with all *annotation* boxes overlaid.
        Uses raw boundary boxes (not per-patch filtered), no labels/text.
        """
        # ensure self.bound_boxes exists even if set_boxes(...) hasn't been called
        if not self.bound_boxes:
            need = {"boundary_center_x", "boundary_center_y", "width", "height", "type"}
            if not need.issubset(self.boundaries.columns):
                raise ValueError(f"boundaries CSV must contain: {need}")

            bx = self.boundaries["boundary_center_x"].astype(float).to_numpy()
            by = self.boundaries["boundary_center_y"].astype(float).to_numpy()
            bw = self.boundaries["width"].astype(float).to_numpy()
            bh = self.boundaries["height"].astype(float).to_numpy()
            raw_types = self.boundaries["type"].to_list()

            bx0 = bx - bw / 2.0
            by0 = by - bh / 2.0

            # build minimal boxes (multi-label kept but not drawn)
            self.bound_boxes = [{
                "x": float(x0), "y": float(y0),
                "w": float(w), "h": float(h),
                "labels": self._split_labels(lbl)
            } for x0, y0, w, h, lbl in zip(bx0, by0, bw, bh, raw_types)]

        # choose a scale that fits the longer side to max_dim
        scale = min(max_dim / float(self.W0), max_dim / float(self.H0))
        Wt, Ht = max(1, int(self.W0 * scale)), max(1, int(self.H0 * scale))

        thumb = self.slide.get_thumbnail((Wt, Ht)).convert("RGB")

        fig_w, fig_h, dpi = Wt / 100.0, Ht / 100.0, 100
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.imshow(thumb)
        ax.axis("off")

        for b in self.bound_boxes:
            x0, y0, w, h = b["x"], b["y"], b["w"], b["h"]
            ax.add_patch(Rectangle((x0 * scale, y0 * scale),
                                   w * scale, h * scale,
                                   fill=False, linewidth=line_width))

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return out_path

    def close(self):
        try:
            self.slide.close()
        except Exception:
            pass


# -------------------------- Train-Test split -------------------------- #
def yolo_train_val_split_to(src_root: str, out_root: str, val_frac: float = 0.2, seed: int = 42):
    """
    Copy a YOLO dataset from src_root (with images/ and labels/) into out_root
    as images/{train,val} and labels/{train,val}. Only keeps images that have labels.
    Overwrites any existing split folders in out_root. Also writes dataset.yaml and
    copies class_map.json if present.
    """
    src = Path(src_root)
    out = Path(out_root)
    src_img, src_lbl = src / "images", src / "labels"
    assert src_img.is_dir(), f"Missing images dir: {src_img}"
    assert src_lbl.is_dir(), f"Missing labels dir: {src_lbl}"

    out_img, out_lbl = out / "images", out / "labels"
    for split in ("train", "val"):
        for base in (out_img, out_lbl):
            d = base / split
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # pair only labeled images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in src_img.iterdir() if p.is_file() and p.suffix.lower() in exts)
    pairs = []
    for im in images:
        lb = src_lbl / f"{im.stem}.txt"
        if lb.exists():
            pairs.append((im, lb))

    random.Random(seed).shuffle(pairs)
    n_total = len(pairs)
    n_val = int(round(n_total * val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    def _copy(im_path: Path, lbl_path: Path, split: str):
        shutil.copy2(im_path, out_img / split / im_path.name)
        shutil.copy2(lbl_path, out_lbl / split / (im_path.stem + ".txt"))

    for im, lb in train_pairs: _copy(im, lb, "train")
    for im, lb in val_pairs:   _copy(im, lb, "val")

    # class names from class_map.json (if available)
    names_list = None
    cmap_src = src / "class_map.json"
    if cmap_src.exists():
        shutil.copy2(cmap_src, out / "class_map.json")
        with open(cmap_src, "r") as f:
            cmap = json.load(f)  # {"label": id, ...}
        inv = {int(v): k for k, v in cmap.items()}
        names_list = [inv[i] for i in range(len(inv))] if inv else None

    # dataset.yaml (Ultralytics)
    yaml_lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if names_list is not None:
        yaml_lines.append("names:")
        for i, name in enumerate(names_list):
            yaml_lines.append(f"  {i}: {name}")
    else:
        yaml_lines.append("names: {}")
    (out / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {"total": n_total, "train": len(train_pairs), "val": len(val_pairs),
            "yaml": str(out / "dataset.yaml"), "out_root": str(out.resolve())}


# --------------------- Compare val images: ref vs pred --------------------- #
def batch_compare_val_with_ref(
    model_path: str,
    val_images_dir: str,
    ref_previews_dir: str,
    out_dir: str = "compare_val",
    conf: float = 0.5,
    iou: float = 0.5,
    max_det: int = 300,
    *,
    # NEW: Resize stitched output to a percentage of its original size (e.g., 50 => 50%)
    final_pct: float | None = None,
    # (optional legacy): Instead of percent, cap the longer edge to this many pixels
    final_max_dim: int | None = None,
    # Optional: swap order (pred | ref) if True; default False keeps (ref | pred)
    reverse_order: bool = False,
    # Optional: JPEG quality for stitched output
    jpeg_quality: int = 95,
):
    """
    Create side-by-side images comparing reference previews vs model predictions.

    - final_pct: If provided (0-100], resizes the stitched canvas by that percent.
                 Example: 50 -> width and height become 50% of original.
    - final_max_dim: Alternative to final_pct; if provided, scales so max(w,h)==final_max_dim.
    - reverse_order=False -> [ref | pred]; True -> [pred | ref]

    If both final_pct and final_max_dim are provided, final_pct takes precedence.
    """
    val_dir = Path(val_images_dir)
    ref_dir = Path(ref_previews_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect validation images (non-recursive)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = sorted([p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not imgs:
        raise FileNotFoundError(f"No images found in {val_dir}")

    # Load model once
    model = YOLO(model_path)

    saved = []
    skipped = []

    def _resize_to_h(img: Image.Image, target_h: int) -> Image.Image:
        w = int(round(img.width * (target_h / img.height)))
        return img.resize((w, target_h), Image.BILINEAR)

    def _downsize_long_edge(img: Image.Image, max_dim: int) -> Image.Image:
        if max_dim is None:
            return img
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= max_dim or max_dim <= 0:
            return img
        scale = max_dim / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.BILINEAR)

    def _resize_by_pct(img: Image.Image, pct: float) -> Image.Image:
        # clamp and interpret as percentage
        s = max(1.0, min(100.0, float(pct)))
        scale = s / 100.0
        if scale == 1.0:
            return img
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        return img.resize((new_w, new_h), Image.BILINEAR)

    with torch.no_grad():
        for img_path in imgs:
            stem = img_path.stem
            # find reference preview
            cand = [ref_dir / f"{stem}_preview.jpg", ref_dir / f"{stem}_preview.png"]
            ref_path = next((c for c in cand if c.exists()), None)

            if ref_path is None:
                skipped.append((img_path, "missing_ref"))
                continue

            # Inference
            res = model(str(img_path), conf=conf, iou=iou, max_det=max_det)[0]
            pred_bgr = res.plot()                         # numpy BGR
            pred = Image.fromarray(pred_bgr[:, :, ::-1])  # to RGB PIL
            ref = Image.open(ref_path).convert("RGB")

            # Make same height, then stitch
            target_h = max(ref.height, pred.height)
            ref_r = _resize_to_h(ref, target_h)
            pred_r = _resize_to_h(pred, target_h)

            if reverse_order:
                left, right = pred_r, ref_r
            else:
                left, right = ref_r, pred_r

            canvas = Image.new("RGB", (left.width + right.width, target_h), (255, 255, 255))
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width, 0))

            # NEW: percent-based resize takes precedence
            if final_pct is not None:
                canvas = _resize_by_pct(canvas, final_pct)
            elif final_max_dim is not None:
                canvas = _downsize_long_edge(canvas, final_max_dim)

            out_path = out / f"{stem}_ref_vs_pred.jpg"
            canvas.save(out_path, quality=jpeg_quality)
            saved.append(str(out_path))

    return {
        "total_val_images": len(imgs),
        "saved": len(saved),
        "skipped_missing_ref": len([s for s in skipped if s[1] == "missing_ref"]),
        "outputs": saved,
        "out_dir": str(out.resolve()),
    }
