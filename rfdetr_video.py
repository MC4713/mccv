# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:30:00 2025
Based on rfDETR.py

@author: MNC
"""

import os, io, re, time, cv2, numpy as np, argparse
from statistics import median
from PIL import Image

import sys
import torch
import torch.nn as nn
import supervision as sv
from rfdetr import RFDETRBase
#from rfdetr import RFDETRLarge

# ================== CONFIG ==================
THRESHOLD = 0.5
PRINT_INTERVAL = 30  # extra console print every N frames

# ===== Spyder/IPython: disable autoreload =====
try:
    get_ipython().run_line_magic("autoreload", "0")  # noqa
except Exception:
    pass

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

def try_move_all_nn_modules(obj, device):
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

def predict_with_autopad(pil_img: Image.Image, rfd: RFDETRBase, threshold: float, device: torch.device):
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
                    if p.is_cuda: return True
            for name in ("model", "module", "backbone", "encoder", "inference_model"):
                if hasattr(cur, name):
                    stack.append(getattr(cur, name))
    except Exception:
        pass
    return False

def run(video_source):
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

    # ================== Build model ==================
    t_build0 = time.perf_counter()
    model = RFDETRBase()
    #model = RFDETRLarge()
    optimized = False
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
        try_move_all_nn_modules(model, device)

    assert any_param_on_cuda(model), "No RFDETR parameters on CUDA. Check env or library version."
    t_build1 = time.perf_counter()
    print(f"[INFO] Model build/placement time: {(t_build1 - t_build0):.2f}s")

    # ================== Video I/O & annotator ==================
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")

    source_name = "webcam" if isinstance(video_source, int) else os.path.basename(video_source)
    print(f"[INFO] Processing video source: {source_name}")

    ellipse_annotator = sv.EllipseAnnotator(thickness=1)
    blk = detect_block_size(model)
    if blk:
        print(f"[INFO] Detected Dinov2 block size: {blk}")

    # Warmup (helps cudnn autotune)
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predict_with_autopad(Image.new("RGB", (1024, 640)), model, THRESHOLD, device)
        torch.cuda.synchronize()
        print(f"[INFO] Warmup inference: {(time.perf_counter()-t0)*1000:.1f} ms")
    except Exception as e:
        print(f"[WARN] Warmup failed (continuing): {e}")

    # ================== Loop with progress & timing ==================
    frame_count = 0
    start = time.perf_counter()
    print("[INFO] Starting video processing loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        # Convert to PIL for model
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Optional pre-pad
        if blk:
            bgr_padded = pad_to_multiple(frame, blk)
            pil_img = Image.fromarray(cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB))

        # Timed predict -> synchronize for accurate CUDA timings
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dets = predict_with_autopad(pil_img, model, THRESHOLD, device)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        frame_count += 1

        # Annotate + display
        annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=dets)
        
        # FPS display
        inst_fps = 1.0 / dt if dt > 0 else float('inf')
        cv2.putText(annotated_frame, f"FPS: {inst_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RFDETR Video Annotation", annotated_frame)

        # Console printout
        if (frame_count % PRINT_INTERVAL) == 0:
            elapsed = time.perf_counter() - start
            avg_fps = frame_count / elapsed if elapsed > 0 else float("nan")
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"[Frame {frame_count:5d}] inst {inst_fps:6.1f} fps | avg {avg_fps:6.1f} fps | mem {mem_gb:4.2f} GB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User requested quit.")
            break

    # ================== Cleanup ==================
    cap.release()
    cv2.destroyAllWindows()
    total_time = time.perf_counter() - start
    avg_throughput = frame_count / total_time if total_time > 0 else 0

    print("\n========== Summary ==========")
    print(f"Frames processed: {frame_count}")
    print(f"Total time:       {total_time:.2f} s")
    print(f"Avg throughput:   {avg_throughput:.2f} fps")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RFDETR object detection on a video source.")
    parser.add_argument(
        "--source",
        default="0",
        help="Path to video file or integer for webcam index (default: 0)."
    )
    args = parser.parse_args()

    # Try to convert source to integer for webcam, otherwise use as path
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    run(video_source)
