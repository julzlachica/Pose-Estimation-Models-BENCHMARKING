#!/usr/bin/env python3
"""
Pose-estimator benchmark (image dataset, CPU only)
Designs: A = MediaPipe Pose (full), B = RTMPose-s (rtmlib, lightweight), C = YOLOv8n-Pose
Metrics: PCK accuracy, inference time, peak RAM, storage, frame rate
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only for every design
import sys, json, time, argparse, subprocess, threading, platform, re, csv
from pathlib import Path
from importlib import metadata
import numpy as np
import cv2
import psutil

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RESULTS = WORK / "results"
MODELS = WORK / "models"

NAMES = {"A": "MediaPipe Pose", "B": "RTMPose-s", "C": "YOLOv8n-Pose"}
PACKAGES = {"A": ["mediapipe"], "B": ["rtmlib", "onnxruntime"], "C": ["ultralytics"]}
SHARED = {"numpy", "opencv-python", "opencv-python-headless", "opencv-contrib-python",
          "psutil", "pandas", "matplotlib"}      # common to all designs -> excluded
MP_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
          "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

IMG_LONG_SIDE = 640      # every model receives the same image, long side = 640 px
WARMUP = 10
SAMPLE_INTERVAL = 0.005  # 5 ms RAM sampling
FPS_FRAMES = 300         # images preloaded in RAM for the end-to-end FPS pass
ALPHAS = [0.05, 0.1, 0.2]
PRIMARY_ALPHA = 0.2

COCO_KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
           "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
           "l_knee", "r_knee", "l_ankle", "r_ankle"]
# MediaPipe 33 landmarks -> COCO 17
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Roboflow "fighter" skeleton (14 pts), in dataset order:
# 0 head, 1 neck, 2/3 shoulder, 4/6 elbow, 5/7 hand, 8/9 hip, 10/13 knee, 11/12 foot
JOINT_NAMES = ["head", "neck", "shoulder", "elbow", "hand", "hip", "knee", "foot"]
GT_IDX = [(0,), (1,), (2, 3), (4, 6), (5, 7), (8, 9), (10, 13), (11, 12)]
PRED_PAIRS = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]  # COCO-17 left/right


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def load_dataset(images_dir, ann_path, n_max=None, allow_multi=False):
    if not ann_path:                      # no ground truth: speed/RAM/storage only
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts)
        items = [{"id": i, "path": str(p), "kpts": None, "bbox": None} for i, p in enumerate(files)]
        return items[:n_max] if n_max else items
    coco = json.load(open(ann_path, "r", encoding="utf-8"))
    anns = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        kp = a.get("keypoints", [])
        if len(kp) != 42 or "bbox" not in a:      # skips 'clinch' (3 values)
            continue
        if sum(1 for v in kp[2::3] if v > 0) == 0:
            continue
        anns.setdefault(a["image_id"], []).append({"kpts": kp, "bbox": a["bbox"]})
    items = []
    for im in sorted(coco["images"], key=lambda x: x["id"]):
        a = anns.get(im["id"], [])
        if not a or (len(a) > 1 and not allow_multi):
            continue
        p = Path(images_dir) / im["file_name"]
        if not p.exists():
            continue
        items.append({"id": im["id"], "path": str(p), "persons": a})
    return items[:n_max] if n_max else items


def resize_img(img):
    s = IMG_LONG_SIDE / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return img, s


def read_resize(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError("Cannot read " + path)
    return resize_img(img)


# ----------------------------------------------------------------------------
# Predictors: each returns native keypoints (K, 3) = x, y, conf in image pixels, or None
# ----------------------------------------------------------------------------
def ensure_file(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    return dest


def make_predictor(design):
    """Returns (predict_fn, model_files, native_to_coco_index)."""
    if design == "A":
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        model = ensure_file(MP_URL, MODELS / "pose_landmarker_full.task")
        opts = vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        lm = vision.PoseLandmarker.create_from_options(opts)

        def predict(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.pose_landmarks:
                return None
            h, w = bgr.shape[:2]
            return np.array([[l.x * w, l.y * h, getattr(l, "visibility", 0.0) or 0.0]
                             for l in res.pose_landmarks[0]], dtype=np.float32)
        return predict, [str(model)], MP_TO_COCO

    if design == "B":
        from rtmlib import Body
        body = Body(mode="lightweight", backend="onnxruntime", device="cpu")  # YOLOX-tiny + RTMPose-s

        def predict(bgr):
            kp, sc = body(bgr)
            if kp is None or len(kp) == 0:
                return None
            i = int(np.argmax(sc.mean(axis=1)))          # most confident person
            return np.concatenate([kp[i], sc[i][:, None]], axis=1).astype(np.float32)
        files = [str(p) for p in (Path.home() / ".cache" / "rtmlib").rglob("*.onnx")]
        return predict, files, None

    if design == "C":
        os.chdir(MODELS)
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")

        def predict(bgr):
            r = model(bgr, imgsz=640, device="cpu", verbose=False)[0]
            if r.keypoints is None or len(r.boxes) == 0:
                return None
            i = int(r.boxes.conf.argmax())               # most confident person
            xy = r.keypoints.xy[i].cpu().numpy()
            cf = (r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None
                  else np.ones(len(xy)))
            return np.concatenate([xy, cf[:, None]], axis=1).astype(np.float32)
        ck = getattr(model, "ckpt_path", None) or str(MODELS / "yolov8n-pose.pt")
        return predict, [str(Path(ck).resolve())], None
    raise ValueError(design)


# ----------------------------------------------------------------------------
# Worker (one design, one repeat, own process so memory is not shared)
# ----------------------------------------------------------------------------
class PeakRSS(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.p, self.peak, self.stop_evt = psutil.Process(), 0, threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(SAMPLE_INTERVAL)


def worker(a):
    items = load_dataset(a.images, a.ann, a.n, a.allow_multi)
    if a.prepare:
        items = items[:WARMUP]
    proc = psutil.Process()
    baseline = proc.memory_info().rss            # before any AI framework is imported
    sampler = PeakRSS()
    sampler.start()

    t0 = time.perf_counter()
    predict, model_files, idx = make_predictor(a.worker)
    load_s = time.perf_counter() - t0

    for it in items[:WARMUP]:                    # not timed
        predict(read_resize(it["path"])[0])
    if a.prepare:
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "%s_run%d" % (a.worker, a.rep)
    csv_path = RESULTS / ("keypoints_%s.csv" % tag)
    lat, preds, detected, native_k = [], [], 0, 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for it in items:
            img, s = read_resize(it["path"])     # I/O + resize are NOT timed
            t = time.perf_counter()
            kp = predict(img)
            lat.append((time.perf_counter() - t) * 1000.0)
            if kp is None:
                preds.append(np.full((17, 3), np.nan, dtype=np.float32))
                w.writerow([it["id"]])
                continue
            detected += 1
            kp = kp.copy()
            kp[:, :2] /= s                       # back to original image coordinates
            native_k = len(kp)
            w.writerow([it["id"]] + np.round(kp.reshape(-1), 3).tolist())
            preds.append(kp[idx] if idx is not None else kp)
    sampler.stop_evt.set()
    sampler.join()
    np.save(RESULTS / ("preds_%s.npy" % tag), np.stack(preds))

    # End-to-end FPS pass (after RAM sampling has stopped): frames are decoded and held in
    # memory first, then resize + color conversion + inference + keypoint extraction are
    # timed together as wall-clock time, like a webcam stream.
    raw = [cv2.imread(it["path"]) for it in items[:FPS_FRAMES]]
    t = time.perf_counter()
    for im in raw:
        small, _ = resize_img(im)
        predict(small)
    fps_e2e = len(raw) / (time.perf_counter() - t)
    del raw

    lat = np.array(lat)
    out = {
        "design": a.worker, "run": a.rep, "n_images": len(items),
        "mean_ms": float(lat.mean()), "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)), "std_ms": float(lat.std()),
        "fps": float(fps_e2e), "fps_frames": min(FPS_FRAMES, len(items)),
        "fps_infer_only": float(len(lat) / (lat.sum() / 1000.0)),
        "peak_ram_mb": (sampler.peak - baseline) / 1e6,
        "abs_peak_mb": sampler.peak / 1e6, "baseline_mb": baseline / 1e6,
        "load_s": load_s, "detected": detected, "native_keypoints": native_k,
        "csv_mb": csv_path.stat().st_size / 1e6, "model_files": model_files,
    }
    json.dump(out, open(RESULTS / ("run_%s.json" % tag), "w"), indent=2)


# ----------------------------------------------------------------------------
# Accuracy: PCK
# ----------------------------------------------------------------------------
def score_person(pr, gt, norm, alphas):
    """Score one predicted person against one GT fighter, per joint type (8 types)."""
    xy = pr[:, :2].astype(np.float64)
    pred_pts = [[xy[0]], [(xy[5] + xy[6]) / 2]] + [[xy[l], xy[r]] for l, r in PRED_PAIRS]
    hits = {a: np.zeros(8) for a in alphas}
    total = np.zeros(8)
    for s_, gi in enumerate(GT_IDX):
        P = np.array(pred_pts[s_])
        G = gt[list(gi), :2]
        V = gt[list(gi), 2] > 0
        with np.errstate(invalid="ignore"):
            D = np.nan_to_num(np.linalg.norm(P[:, None] - G[None], axis=2), nan=np.inf)
        if len(gi) == 2:                      # left/right unknown: take the better pairing
            straight = np.array([D[0, 0], D[1, 1]])
            swapped = np.array([D[1, 0], D[0, 1]])
            d = swapped if np.where(V, swapped, 0).sum() < np.where(V, straight, 0).sum() else straight
        else:
            d = np.array([D[0, 0]])
        total[s_] += V.sum()
        for al in alphas:
            hits[al][s_] += ((d <= al * norm) & V).sum()
    return hits, total


def compute_pck(preds, items, alphas):
    """PCK@alpha on the shared joints. norm = max(GT bbox w, h). In multi-fighter images the
    prediction is matched to the best-fitting fighter. A missed detection scores 0."""
    hits = {a: np.zeros(8) for a in alphas}
    total = np.zeros(8)
    for pr, it in zip(preds, items):
        best = None
        for p in it["persons"]:
            gt = np.array(p["kpts"], dtype=np.float64).reshape(14, 3)
            norm = max(p["bbox"][2], p["bbox"][3])
            if norm <= 0:
                vis = gt[:, 2] > 0
                norm = max(np.ptp(gt[vis, 0]), np.ptp(gt[vis, 1]), 1.0)
            h, t = score_person(pr, gt, norm, alphas)
            if best is None or h[PRIMARY_ALPHA].sum() > best[0][PRIMARY_ALPHA].sum():
                best = (h, t)
        for al in alphas:
            hits[al] += best[0][al]
        total += best[1]
    overall = {a: float(hits[a].sum() / total.sum() * 100) for a in alphas}
    per_kp = {a: hits[a] / np.maximum(total, 1) * 100 for a in alphas}
    return overall, per_kp


# ----------------------------------------------------------------------------
# Storage helpers
# ----------------------------------------------------------------------------
def _norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def package_closure(roots):
    seen, stack = {}, list(roots)
    while stack:
        n = _norm(stack.pop())
        if n in seen or n in SHARED:
            continue
        try:
            d = metadata.distribution(n)
        except metadata.PackageNotFoundError:
            continue
        seen[n] = d
        for r in d.requires or []:
            if "extra" in r.split(";", 1)[-1] and ";" in r:
                continue
            stack.append(re.split(r"[\s;<>=!~\[(]", r, maxsplit=1)[0])
    return seen


def dist_size_mb(d):
    tot = 0
    for f in d.files or []:
        try:
            tot += d.locate_file(f).stat().st_size
        except OSError:
            pass
    return tot / 1e6


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def run_worker(a, design, rep, prepare=False):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", design, "--rep", str(rep),
           "--images", a.images]
    if a.ann:
        cmd += ["--ann", a.ann]
    if a.n:
        cmd += ["--n", str(a.n)]
    if a.allow_multi:
        cmd.append("--allow-multi")
    if prepare:
        cmd.append("--prepare")
    subprocess.run(cmd, check=True)


def main(a):
    import pandas as pd
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    items = load_dataset(a.images, a.ann, a.n, a.allow_multi)
    if not items:
        sys.exit("No usable images found. Check --images and --ann.")
    if not a.ann:
        print("No --ann given: PCK accuracy will be NaN (speed/RAM/storage only).")
    print("Usable images:", len(items))

    json.dump({
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cores": psutil.cpu_count(True),
        "physical_cores": psutil.cpu_count(False),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "compute": "CPU only (CUDA_VISIBLE_DEVICES empty)", "n_images": len(items),
        "image_long_side": IMG_LONG_SIDE, "warmup_images": WARMUP, "reps": a.reps,
        "pck_normalization": "max(GT bbox width, height)", "alphas": ALPHAS,
    }, open(RESULTS / "device.json", "w"), indent=2)

    for d in a.designs:                                 # discarded pass: model downloads
        print("Preparing Design", d)
        run_worker(a, d, 0, prepare=True)
    for rep in range(1, a.reps + 1):                    # interleaved repeats
        for d in a.designs:
            print("Run %d  Design %s (%s)" % (rep, d, NAMES[d]))
            run_worker(a, d, rep)

    rows, pck_rows, kp_rows = [], [], []
    for d in a.designs:
        runs = [json.load(open(RESULTS / ("run_%s_run%d.json" % (d, r)))) for r in range(1, a.reps + 1)]
        df = pd.DataFrame(runs)
        pkgs = package_closure(PACKAGES[d])
        pkg_mb = sum(dist_size_mb(x) for x in pkgs.values())
        model_mb = sum(Path(f).stat().st_size for f in set(runs[0]["model_files"]) if Path(f).exists()) / 1e6
        csv_mb = df["csv_mb"].mean()
        preds = np.load(RESULTS / ("preds_%s_run1.npy" % d))
        has_gt = items[0].get("persons") is not None
        if has_gt:
            overall, per_kp = compute_pck(preds, items, ALPHAS)
        else:
            overall = {a_: float("nan") for a_ in ALPHAS}
            per_kp = {a_: np.full(8, np.nan) for a_ in ALPHAS}
        rows.append({
            "Design": "Design %s: %s" % (d, NAMES[d]),
            "PCK@%.2f (%%)" % PRIMARY_ALPHA: overall[PRIMARY_ALPHA],
            "Inference Time (ms)": df["mean_ms"].mean(),
            "Peak RAM (MB)": df["peak_ram_mb"].mean(),
            "Storage (MB)": pkg_mb + model_mb + csv_mb,
            "Frame Rate (FPS)": df["fps"].mean(),
        })
        pck_rows.append({
            "Design": NAMES[d], **{"PCK@%.2f" % a_: overall[a_] for a_ in ALPHAS},
            "Detection rate (%)": runs[0]["detected"] / runs[0]["n_images"] * 100,
            "Keypoints native": runs[0]["native_keypoints"],
            "FPS inference-only": df["fps_infer_only"].mean(), "FPS frames": runs[0]["fps_frames"],
            "Median ms": df["median_ms"].mean(), "P95 ms": df["p95_ms"].mean(),
            "Std ms": df["std_ms"].mean(), "Model load (s)": df["load_s"].mean(),
            "Packages MB": pkg_mb, "Model MB": model_mb, "Keypoint CSV MB": csv_mb,
            "Abs peak RAM MB": df["abs_peak_mb"].mean(), "Baseline MB": df["baseline_mb"].mean(),
            "Images": runs[0]["n_images"],
        })
        for k, name in enumerate(JOINT_NAMES):
            kp_rows.append({"Design": NAMES[d], "Keypoint": name,
                            **{"PCK@%.2f" % a_: per_kp[a_][k] for a_ in ALPHAS}})
        print("  %s packages counted: %s" % (NAMES[d], ", ".join(sorted(pkgs))))
        # Per-run table
        df[["design", "run", "mean_ms", "fps", "peak_ram_mb"]].to_csv(
            RESULTS / ("per_run_%s.csv" % d), index=False)

    pd.DataFrame(rows).to_csv(RESULTS / "summary.csv", index=False)
    pd.DataFrame(pck_rows).to_csv(RESULTS / "supporting.csv", index=False)
    pd.DataFrame(kp_rows).to_csv(RESULTS / "pck_per_keypoint.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Summary (means of %d runs, %d images) ===" % (a.reps, len(items)))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\n=== Supporting ===")
    print(pd.DataFrame(pck_rows).round(2).to_string(index=False))
    print("\nFiles saved in", RESULTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--designs", default="ABC")
    ap.add_argument("--allow-multi", action="store_true")
    ap.add_argument("--worker", default=None)       # internal
    ap.add_argument("--rep", type=int, default=0)   # internal
    ap.add_argument("--prepare", action="store_true")  # internal
    args = ap.parse_args()
    args.images = str(Path(args.images).resolve())   # Design C changes the working directory
    if args.ann:
        args.ann = str(Path(args.ann).resolve())
    if args.worker:
        worker(args)
    else:
        main(args)