# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 01:38:57 2025

@author: MNC
"""

# pip install -U transformers torch torchvision pillow opencv-python av



import os, gc, cv2, torch, numpy as np
from transformers import Sam2VideoModel, Sam2VideoProcessor, infer_device
from tqdm.auto import tqdm

# ========= CONFIG =========
VIDEO_IN   = r"C:\Users\MNC\Small soccer clip.mp4"  # full video    # input video
VIDEO_OUT   = r"C:\Users\MNC\sam2_track\football_SAM_test.mp4"
MODEL_ID    = "facebook/sam2.1-hiera-tiny"                 # try -small/-base if you have more VRAM
TARGET_WIDTH = 640     # inference width (downscale). None = original (heavier)
ALPHA        = 0.50    # overlay opacity (0..1)
COLOR_BGR    = (0, 0, 255)  # red overlay in BGR (OpenCV)

os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)

# ========= HELPERS =========
def resize_keep_w(rgb_np, target_w):
    if not target_w or rgb_np.shape[1] <= target_w:
        return rgb_np
    h, w = rgb_np.shape[:2]
    new_h = int(round(h * (target_w / w)))
    return cv2.resize(rgb_np, (target_w, new_h), interpolation=cv2.INTER_LINEAR)

def overlay_mask_bgr(bgr, mask_bool, color=COLOR_BGR, alpha=ALPHA):
    out = bgr.copy()
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool.astype(bool)
    out[mask_bool] = (out[mask_bool].astype(np.float32) * (1.0 - alpha) +
                      np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out

# ========= OPEN VIDEO / WRITER =========
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open: {VIDEO_IN}")

orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tot      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, orig_fps, (orig_w, orig_h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to create writer: {VIDEO_OUT}")

# ========= FIRST FRAME + CLICK =========
ok, bgr0 = cap.read()
if not ok:
    writer.release(); cap.release()
    raise RuntimeError("Could not read first frame.")
rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
rgb0_small = resize_keep_w(rgb0, TARGET_WIDTH)

win = "Click target (Enter=confirm, Esc=cancel)"
click = {}
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (x, y)
        disp = param.copy()
        cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
        cv2.imshow(win, disp)

cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.imshow(win, cv2.cvtColor(rgb0_small, cv2.COLOR_RGB2BGR))
cv2.setMouseCallback(win, on_mouse, cv2.cvtColor(rgb0_small, cv2.COLOR_RGB2BGR))
while True:
    k = cv2.waitKey(1) & 0xFF
    if k in (13, 10):  # Enter
        if "pt" in click: break
    if k == 27:        # Esc
        cap.release(); writer.release(); cv2.destroyAllWindows()
        raise SystemExit("Cancelled.")
cv2.destroyAllWindows()
CLICK_X, CLICK_Y = click["pt"]

# ========= MODEL (STREAMING SESSION) =========
device = infer_device()  # "cuda:0" if available else "cpu"
dtype  = torch.float16 if "cuda" in str(device) else torch.float32

model = Sam2VideoModel.from_pretrained(MODEL_ID).to(device, dtype=dtype)
processor = Sam2VideoProcessor.from_pretrained(MODEL_ID)

# initialize empty session (no preloaded video)
session = processor.init_video_session(
    inference_device=device,
    dtype=dtype,
    video_storage_device="cpu",
)

# prepare inputs for frame 0 at inference size
inputs0 = processor(images=rgb0_small, return_tensors="pt").to(device)

# register click on frame 0
processor.add_inputs_to_inference_session(
    inference_session=session,
    frame_idx=0,
    obj_ids=1,
    input_points=[[[[CLICK_X, CLICK_Y]]]],
    input_labels=[[[1]]],
    original_size=inputs0.original_sizes[0],
)

# infer frame 0
with torch.inference_mode():
    out0 = model(inference_session=session, frame=inputs0.pixel_values[0])
mask0_small = processor.post_process_masks(
    [out0.pred_masks], original_sizes=inputs0.original_sizes, binarize=True
)[0][0][0].detach().cpu().numpy().astype(np.uint8)
mask0 = cv2.resize(mask0_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
writer.write(overlay_mask_bgr(bgr0, mask0))

# ========= STREAM THE REST WITH PROGRESS =========
pbar = tqdm(total=tot if tot > 0 else None, desc="Masking video", unit="frame")
pbar.update(1)  # we already wrote frame 0
written = 1

while True:
    ok, bgr = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_small = resize_keep_w(rgb, TARGET_WIDTH)

    inputs = processor(images=rgb_small, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(inference_session=session, frame=inputs.pixel_values[0])

    mask_small = processor.post_process_masks(
        [out.pred_masks], original_sizes=inputs.original_sizes, binarize=True
    )[0][0][0].detach().cpu().numpy().astype(np.uint8)

    mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    writer.write(overlay_mask_bgr(bgr, mask))
    written += 1
    pbar.update(1)

    # free transient memory
    del inputs, out, mask_small, mask, rgb, rgb_small
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

pbar.close()
writer.release()
cap.release()
print(f"Done. Wrote {written} frames to: {VIDEO_OUT}")
