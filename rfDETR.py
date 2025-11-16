# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:27:40 2025

@author: MNC
"""

import os, io, re, glob, time, cv2, numpy as np
from statistics import median
from PIL import Image

import sys
import torch
import torch.nn as nn
import supervision as sv
from rfdetr import RFDETRBase
#from rfdetr import RFDETRLarge
try:
    from tqdm import tqdm
    _USE_TQDM = True
except Exception:
    _USE_TQDM = False

# ================== CONFIG ==================
INPUT_DIR  = r"C:\Users\MNC\frames"
OUTPUT_DIR = r"C:\Users\MNC\footballCV\annotated_frames"
THRESHOLD  = 0.5
SAVE_PNG   = False
PRINT_INTERVAL = 50  # extra console print every N frames

# ===== Spyder/IPython: disable autoreload =====
try:
    get_ipython().run_line_magic("autoreload", "0")  # noqa
except Exception:
    pass

# ================== CUDA & AMP ==================
assert torch.cuda.is_available(), "CUDA not available (need CUDA-enabled PyTorch)."
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision  = "ieee"
except Exception:
    pass

print(f"[INFO] Device: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
print(f"[INFO] cuDNN benchmark: {torch.backends.cudnn.benchmark}")

# ================== Helpers ==================
def pad_to_multiple(img_bgr: np.ndarray, multiple: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    nh = (h + multiple - 1) // multiple * multiple
    nw = (w + multiple - 1) // multiple * multiple
    if nh == h and nw == w:
        return img_bgr
    return cv2.copyMakeBorder(img_bgr, 0, nh - h, 0, nw - w, cv2.BORDER_REPLICATE)

def detect_block_size(model_like) -> int | None:
    try:
        node = getattr(model_like, "model", model_like)
        for name in ("backbone", "encoder"):
            if hasattr(node, name):
                node = getattr(node, name)
        patch = getattr(node, "patch_size", None)
        nwin  = getattr(node, "num_windows", None)
        if isinstance(patch, int) and isinstance(nwin, int):
            return patch * nwin
    except Exception:
        pass
    return None

def try_move_all_nn_modules(obj):
    seen = set()
    def walk(x, depth=0):
        if id(x) in seen or depth > 3:
            return
        seen.add(id(x))
        if isinstance(x, nn.Module):
            x.to(device).eval()
        for name in ("model", "module", "backbone", "encoder", "inference_model"):
            if hasattr(x, name):
                walk(getattr(x, name), depth + 1)
    walk(obj)

def predict_with_autopad(pil_img: Image.Image, rfd: RFDETRBase, threshold: float):
    def _call(img):
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            if "device" in rfd.predict.__code__.co_varnames:
                return rfd.predict(img, threshold=threshold, device=device)
            return rfd.predict(img, threshold=threshold)
    try:
        return _call(pil_img)
    except AssertionError as e:
        m = re.search(r"divisible by (\d+)", str(e))
        if not m:
            raise
        blk = int(m.group(1))
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        bgr = pad_to_multiple(bgr, blk)
        pil_padded = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return _call(pil_padded)

# ================== Build model ==================
t_build0 = time.perf_counter()
model = RFDETRBase()
#model = RFDETRLarge()
optimized = True
if hasattr(model, "optimize_for_inference"):
    try:
        model.optimize_for_inference(device="cuda", dtype=torch.float16)
        optimized = True
        print("[INFO] RFDETR optimized for inference on CUDA (fp16).")
    except TypeError:
        try:
            model.optimize_for_inference("cuda", torch.float16)
            optimized = True
            print("[INFO] RFDETR optimized for inference on CUDA (fp16).")
        except Exception as e:
            print(f"[WARN] optimize_for_inference call failed ({e}); falling back to manual .to(cuda).")
    except Exception as e:
        print(f"[WARN] optimize_for_inference not usable ({e}); falling back to manual .to(cuda).")

if not optimized:
    try_move_all_nn_modules(model)

def any_param_on_cuda(x) -> bool:
    try:
        stack = [getattr(x, "model", x)]
        seen = set()
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if isinstance(cur, nn.Module):
                for p in cur.parameters(recurse=True):
                    return p.is_cuda
            for name in ("model", "module", "backbone", "encoder", "inference_model"):
                if hasattr(cur, name):
                    stack.append(getattr(cur, name))
    except Exception:
        pass
    return False

assert any_param_on_cuda(model), "No RFDETR parameters on CUDA. Check env or library version."
t_build1 = time.perf_counter()
print(f"[INFO] Model build/placement time: {(t_build1 - t_build0):.2f}s")

# ================== I/O & annotator ==================
os.makedirs(OUTPUT_DIR, exist_ok=True)
paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
assert paths, f"No images found in {INPUT_DIR}"
n_total = len(paths)
print(f"[INFO] Found {n_total} frames in {INPUT_DIR}")
ellipse_annotator = sv.EllipseAnnotator(thickness=1)

blk = detect_block_size(model)
if blk:
    print(f"[INFO] Detected Dinov2 block size: {blk}")

# Warmup (helps cudnn autotune)
try:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = predict_with_autopad(Image.new("RGB", (1024, 640)), model, THRESHOLD)
    torch.cuda.synchronize()
    print(f"[INFO] Warmup inference: {(time.perf_counter()-t0)*1000:.1f} ms")
except Exception as e:
    print(f"[WARN] Warmup failed (continuing): {e}")

# ================== Loop with progress & timing ==================
per_frame_sec = []
start = time.perf_counter()
max_mem_gb_start = torch.cuda.max_memory_allocated() / 1e9

if _USE_TQDM:
    pbar = tqdm(
        total=n_total,
        desc="Annotating",
        unit="frame",
        dynamic_ncols=True,
        mininterval=0.2,    # throttle screen updates
        ascii=True,         # safer in some consoles
        leave=True,
        position=0,
        disable=False,
        file=sys.stdout,    # <- critical for Spyder runfile
    )
else:
    pbar = None
    print("[INFO] tqdm not available; using periodic prints.")

for i, pth in enumerate(paths, 1):
    # Load
    with open(pth, "rb") as f:
        pil = Image.open(io.BytesIO(f.read())).convert("RGB")

    # Optional pre-pad
    if blk:
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        bgr = pad_to_multiple(bgr, blk)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Timed predict -> synchronize for accurate CUDA timings
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dets = predict_with_autopad(pil, model, THRESHOLD)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    per_frame_sec.append(dt)

    # Annotate + save (CPU)
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    annotated = ellipse_annotator.annotate(scene=frame, detections=dets)
    out_name = os.path.splitext(os.path.basename(pth))[0] + (".png" if SAVE_PNG else ".jpg")
    out_pil  = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    out_pil.save(os.path.join(OUTPUT_DIR, out_name), optimize=True)

    # Progress UI
    elapsed = time.perf_counter() - start
    avg_fps = i / elapsed if elapsed > 0 else float("nan")
    inst_fps = (1.0 / dt) if dt > 0 else float("inf")
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    if pbar:
        pbar.set_postfix({
    "inst_fps": f"{inst_fps:6.1f}",
    "avg_fps":  f"{avg_fps:6.1f}",
    "mem_GB":   f"{mem_gb:4.2f}"
    }, refresh=False)
    pbar.update(1)
    
    # force a visible refresh periodically (helps Spyder)
    if (i % 20) == 0:
        pbar.refresh()
    else:
        if (i % PRINT_INTERVAL) == 0 or i == n_total:
            print(f"[{i:6d}/{n_total}] inst {inst_fps:6.1f} fps | avg {avg_fps:6.1f} fps | mem {mem_gb:4.2f} GB")

if pbar:
    pbar.close()

total_time = time.perf_counter() - start
max_mem_gb = torch.cuda.max_memory_allocated() / 1e9
p50 = median(per_frame_sec) if per_frame_sec else float("nan")
p95 = np.percentile(per_frame_sec, 95) if per_frame_sec else float("nan")

print("\n========== Summary ==========")
print(f"Frames:          {n_total}")
print(f"Total time:      {total_time:.2f} s")
print(f"Throughput:      {n_total/total_time:.2f} fps")
print(f"Median/frame:    {p50*1000:.1f} ms")
print(f"P95/frame:       {p95*1000:.1f} ms")
print(f"CUDA max mem:    {max_mem_gb:4.2f} GB (start {max_mem_gb_start:4.2f} GB)")
print(f"Output dir:      {OUTPUT_DIR}")
print("Done – all annotated frames saved.")
