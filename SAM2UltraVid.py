# -*- coding: utf-8 -*-
# SAM2 video tracking — chunked, GPU-optimized, NVENC output, quiet console.

import os, cv2, time, warnings, logging, gc, subprocess, shutil
import numpy as np
from contextlib import nullcontext
warnings.filterwarnings("ignore")

import torch
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.CRITICAL)
from ultralytics.models.sam import SAM2VideoPredictor

# print(torch.__version__, torch.version.cuda)          # expect cu126
# print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
# print("cc:", torch.cuda.get_device_capability(0))     # expect (12, 0)
# try:
#     print("archs:", torch.cuda.get_arch_list())       # look for 'sm_120' (or PTX)
# except Exception as e:
#     print("archs: n/a ->", e)
# ---------------------- config ----------------------
VIDEO = r"C:\Users\MNC\Downloads\metric_MechanicalWorkRate_1BN0X6_GameDaySportable_1603.0-2000.0_00-50_2-20.mp4"
MODEL = "sam2.1_l.pt"
IMG_SIZE = 1024
PREVIEW_SECONDS = 30
SAVE_ROOT = r"C:\Users\MNC\sam2_track_ultralytics"
RUN_NAME  = "sam2_ultralytics"
VID_STRIDE = 1

OUTPUT_MODE = "overlay"        # "overlay" or "mask"
ALPHA = 0.55
COLOR = (255, 0, 0)            # BGR

CHUNK = 1800                    # frames per chunk
CLEANUP_EVERY = 60

# Speed toggles
USE_AUTOCAST = True            # fp16 autocast
USE_TF32     = True            # TF32 on (5090 supports it)
CUDNN_BENCH  = True            # fastest convs
USE_NVENC    = True            # use ffmpeg NVENC if available

os.makedirs(SAVE_ROOT, exist_ok=True)

# ---------------- speed knobs -----------------------
if USE_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if CUDNN_BENCH:
    torch.backends.cudnn.benchmark = True

def fmt_hms(s):
    if not s or s == float("inf") or s < 0: return "--:--"
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# ---------------- open video & scrubber --------------
cap0 = cv2.VideoCapture(VIDEO)
if not cap0.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO}")

total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
H0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
W0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))   # <-- fixed name
fps = cap0.get(cv2.CAP_PROP_FPS) or 25.0
fps = 25.0 if fps <= 1e-3 else fps

end_frame = min(total - 1, int(PREVIEW_SECONDS * fps))
win = "Scrub 30s: drag bar, click point, Enter=confirm (Esc=cancel)"
choice = {"pt": None, "start": 0}
_disp_w = min(1280, W0); _scale = _disp_w / float(W0); _disp_h = int(round(H0 * _scale))

def _show_frame(fi, mark=None):
    cap0.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, bgr = cap0.read()
    if not ok: return
    vis = cv2.resize(bgr, (_disp_w, _disp_h), cv2.INTER_LINEAR)
    if mark is not None: cv2.circle(vis, mark, 6, (0,255,255), -1)
    cv2.putText(vis, f"f{fi}/{end_frame}  t={fi/fps:0.2f}s",
                (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    cv2.imshow(win, vis)

def _on_trackbar(val):
    mark=None
    if choice["pt"] is not None and choice["start"]==val:
        mx=int(round(choice["pt"][0]*_scale)); my=int(round(choice["pt"][1]*_scale))
        mark=(mx,my)
    _show_frame(val, mark)

def _on_mouse(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        fi=cv2.getTrackbarPos("frame", win)
        px=int(round(x/_scale)); py=int(round(y/_scale))
        px=max(0,min(W0-1,px)); py=max(0,min(H0-1,py))
        choice["pt"]=(px,py); choice["start"]=fi
        _show_frame(fi,(x,y))

cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, _disp_w, _disp_h)
cv2.createTrackbar("frame", win, 0, max(0,end_frame), _on_trackbar)
cv2.setMouseCallback(win, _on_mouse)
_show_frame(0)

while True:
    k=cv2.waitKey(1)&0xFF
    if k in (13,10) and choice["pt"] is not None: break
    if k==27:
        cv2.destroyAllWindows(); cap0.release(); raise SystemExit("Cancelled.")
cv2.destroyAllWindows(); cap0.release()

PX,PY = choice["pt"]; START_F = int(choice["start"])

# ---------------- paths ----------------
base = os.path.splitext(os.path.basename(VIDEO))[0]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
trim_path = os.path.join(SAVE_ROOT, f"{base}_from_{START_F}.mp4")
out_path  = os.path.join(SAVE_ROOT, f"{base}_{RUN_NAME}_{'OVERLAY' if OUTPUT_MODE=='overlay' else 'MASKONLY'}_from_{START_F}.mp4")

# ---------------- trim tail --------------
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, START_F)
tail_writer = cv2.VideoWriter(trim_path, fourcc, fps, (W0, H0))
if not tail_writer.isOpened(): raise RuntimeError(f"Writer failed: {trim_path}")
while True:
    ok,bgr = cap.read()
    if not ok: break
    tail_writer.write(bgr)
tail_writer.release(); cap.release()

# ---------------- output writer ----------
def open_nvenc_writer(path, width, height, fps):
    # ffmpeg stdin raw BGR -> NVENC H.264
    cmd = [
        "ffmpeg","-loglevel","error","-y",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{width}x{height}",
        "-r",str(fps),"-i","pipe:0",
        "-c:v","h264_nvenc","-preset","p5","-tune","hq","-b:v","0","-cq","18",
        "-pix_fmt","yuv420p",
        path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

have_ffmpeg = shutil.which("ffmpeg") is not None
ffmpeg_proc = None
if USE_NVENC and have_ffmpeg:
    ffmpeg_proc = open_nvenc_writer(out_path, W0, H0, fps)
    out_writer = None
else:
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (W0, H0), True)
    if not out_writer.isOpened():
        raise RuntimeError(f"Writer failed: {out_path}")

# -------------- helpers ------------------
def centroid_from_mask(m, fallback_xy):
    ys, xs = np.nonzero(m)
    if xs.size == 0: return fallback_xy
    return (int(xs.mean()), int(ys.mean()))

# -------------- chunked video-mode loop --------------
cap_tail = cv2.VideoCapture(trim_path)
trim_total = int(cap_tail.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

start = time.time()
ema_fps, done_prev = None, -1
bar_width = 50
written = 0
current_prompt = (PX, PY)
chunk_idx = 0

with torch.inference_mode():
    while True:
        # Write a temporary chunk video file
        frames_in_chunk = 0
        chunk_path = os.path.join(SAVE_ROOT, f"{base}_chunk_{chunk_idx:04d}.mp4")
        chunk_writer = cv2.VideoWriter(chunk_path, fourcc, fps, (W0, H0))
        if not chunk_writer.isOpened():
            cap_tail.release()
            if out_writer: out_writer.release()
            if ffmpeg_proc: ffmpeg_proc.stdin.close(); ffmpeg_proc.wait()
            raise RuntimeError(f"Writer failed: {chunk_path}")

        while frames_in_chunk < CHUNK:
            ok, frame = cap_tail.read()
            if not ok: break
            chunk_writer.write(frame)
            frames_in_chunk += 1
        chunk_writer.release()

        if frames_in_chunk == 0:
            if os.path.exists(chunk_path): os.remove(chunk_path)
            break

        # Fresh predictor per chunk
        predictor = SAM2VideoPredictor(overrides=dict(
            model=MODEL,
            imgsz=IMG_SIZE,
            device='cuda:0',
            vid_stride=VID_STRIDE,
            save=False, show=False, verbose=False,
            project=SAVE_ROOT, name=RUN_NAME, exist_ok=True,
        ))

        # Open the chunk for reading frames in sync
        chunk_cap = cv2.VideoCapture(chunk_path)

        # Prompt (centroid propagated if available)
        current_prompt = (int(current_prompt[0]), int(current_prompt[1]))
        gen = predictor(source=chunk_path, stream=True,
                        points=[current_prompt[0], current_prompt[1]], labels=[1])

        last_mask = None

        amp_ctx = torch.autocast("cuda", dtype=torch.float16) if USE_AUTOCAST else nullcontext()
        with amp_ctx:
            for res in gen:
                ok, frame = chunk_cap.read()
                if not ok: break

                # ---- GPU-side union -> move only final mask to CPU ----
                masks = getattr(res, "masks", None)
                if masks is not None and getattr(masks, "data", None) is not None:
                    md = masks.data  # CUDA tensor [N,H,W] or [H,W]
                    try:
                        m_gpu = (md.any(dim=0) if md.ndim == 3 else (md > 0))
                        m = (m_gpu.to(torch.uint8).mul_(255)).cpu().numpy()
                    finally:
                        del md, m_gpu
                else:
                    m = np.zeros((H0, W0), dtype=np.uint8)

                # drop heavy refs
                try:
                    if hasattr(res, "orig_img"): res.orig_img = None
                    if hasattr(res, "path"):     res.path = None
                    if hasattr(res, "boxes"):    res.boxes = None
                    if hasattr(res, "probs"):    res.probs = None
                    if hasattr(res, "names"):    res.names = None
                except Exception:
                    pass
                del res, masks

                # ---- compose & write ----
                if OUTPUT_MODE.lower() == "overlay":
                    overlay = frame.copy()
                    overlay[m > 0] = COLOR
                    out = cv2.addWeighted(overlay, ALPHA, frame, 1.0 - ALPHA, 0.0)
                else:
                    out = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

                if ffmpeg_proc:
                    ffmpeg_proc.stdin.write(out.tobytes())
                else:
                    out_writer.write(out)

                last_mask = m
                written += 1

                # ---- progress bar ----
                elapsed = max(time.time() - start, 1e-6)
                inst_fps = written / elapsed
                ema_fps = inst_fps if ema_fps is None else 0.1*inst_fps + 0.9*ema_fps
                if trim_total > 0:
                    pct = int(100 * written / trim_total)
                    if pct != done_prev:
                        filled = "#" * (pct // 2)
                        bar = f"[{filled:<{bar_width}}]"
                        remaining = max(trim_total - written, 0)
                        eta = remaining / max(ema_fps, 1e-6)
                        print(f"\r{bar} {written}/{trim_total} ({pct:3d}%) | {ema_fps:6.2f} fps | ETA {fmt_hms(eta)} | Elapsed {fmt_hms(elapsed)}",
                              end="", flush=True)
                        done_prev = pct
                else:
                    print(f"\rFrames: {written} | {ema_fps:6.2f} fps | Elapsed {fmt_hms(elapsed)}",
                          end="", flush=True)

                if (written % CLEANUP_EVERY) == 0:
                    torch.cuda.empty_cache(); gc.collect()

        chunk_cap.release()
        del predictor
        torch.cuda.empty_cache(); gc.collect()

        # next-chunk prompt from last-mask centroid
        if last_mask is not None:
            ys, xs = np.nonzero(last_mask)
            if xs.size > 0:
                current_prompt = (int(xs.mean()), int(ys.mean()))

        # cleanup temp file
        if os.path.exists(chunk_path): os.remove(chunk_path)

        if frames_in_chunk < CHUNK:
            break
        chunk_idx += 1

print()
cap_tail.release()
if ffmpeg_proc:
    ffmpeg_proc.stdin.close(); ffmpeg_proc.wait()
else:
    out_writer.release()
torch.cuda.empty_cache(); gc.collect()

print(f"Saved output to: {out_path}")

