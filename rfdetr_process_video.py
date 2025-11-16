# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:00:00 2025
Processes a video file to create a new video with RFDETR annotations.

@author: MNC
"""

import os, io, re, time, cv2, numpy as np, argparse
from PIL import Image

import sys
import torch
import torch.nn as nn
import supervision as sv
from rfdetr import RFDETRBase
from tqdm import tqdm

# ================== CONFIG ==================
THRESHOLD = 0.5

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

def run(source_path, output_path, show_preview=True):
    # ================== CUDA & AMP ==================
    assert torch.cuda.is_available(), "CUDA not available (need CUDA-enabled PyTorch)."
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] Device: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")

    # ================== Build model ==================
    t_build0 = time.perf_counter()
    model = RFDETRBase()
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

    # ================== Video I/O ==================
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer for path: {output_path}")

    print(f"[INFO] Processing {source_path} ({total_frames} frames at {fps:.2f} FPS)")
    print(f"[INFO] Saving annotated video to: {output_path}")

    ellipse_annotator = sv.EllipseAnnotator(thickness=1)
    blk = detect_block_size(model)
    if blk:
        print(f"[INFO] Detected Dinov2 block size: {blk}")

    # ================== Loop with Progress Bar ==================
    pbar = tqdm(total=total_frames, desc="Annotating Video", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if blk:
            bgr_padded = pad_to_multiple(frame, blk)
            pil_img = Image.fromarray(cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB))

        torch.cuda.synchronize()
        dets = predict_with_autopad(pil_img, model, THRESHOLD, device)
        torch.cuda.synchronize()

        annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=dets)
        
        writer.write(annotated_frame)

        if show_preview:
            # Resize for display if the video is very large
            preview_h, preview_w = annotated_frame.shape[:2]
            if preview_w > 1280:
                scale = 1280 / preview_w
                preview_frame = cv2.resize(annotated_frame, (0,0), fx=scale, fy=scale)
            else:
                preview_frame = annotated_frame
            
            cv2.imshow("Live Annotation Preview", preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User requested quit during preview.")
                break
        
        pbar.update(1)

    # ================== Cleanup ==================
    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Successfully saved annotated video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file with RFDETR and save the annotated output.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output annotated video file."
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window to run faster."
    )
    args = parser.parse_args()
    
    run(args.source, args.output, show_preview=not args.no_preview)
