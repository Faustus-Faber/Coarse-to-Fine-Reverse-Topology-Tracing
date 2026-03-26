import cv2
import numpy as np
import csv
import os
import time

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


def compute_psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(a, b):
    """SSIM computation using OpenCV (no skimage dependency)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


# ---- Load all scenario data ----
rows = list(csv.DictReader(open(os.path.join(DATASET_DIR, 'results.csv'))))
scenario_names = [r['name'] for r in rows]

# ---- Methods ----


def baseline_linear_blend(I0, I1, t=0.5):
    """Baseline 1: Simple linear blend (no optical flow)."""
    return ((1 - t) * I0.astype(float) + t * I1.astype(float)).astype(np.uint8)


def baseline_backward_flow(I0, I1, t=0.5):
    """Baseline 2: Direct backward flow from I1 to I0, standard remap."""
    g0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = I0.shape[:2]
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32) + flow[..., 0] * t
    map_y = np.tile(np.arange(h).reshape(-1, 1), (1, w)
                    ).astype(np.float32) + flow[..., 1] * t
    return cv2.remap(I1, map_x, map_y, cv2.INTER_LINEAR)


def ablation_proxy_only(I0, I1, t=0.5):
    """Ablation A: Proxy frame only (Eq.1, no flow refinement)."""
    return baseline_linear_blend(I0, I1, t)


def ablation_proxy_flow_no_fading(I0, I1, t=0.5):
    """Ablation B: Proxy + flow, but uniform blending (no exp fading)."""
    proxy = ((1 - t) * I0.astype(float) + t *
             I1.astype(float)).astype(np.uint8)
    gp = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
    g0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    f0 = cv2.calcOpticalFlowFarneback(gp, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    f1 = cv2.calcOpticalFlowFarneback(gp, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = I0.shape[:2]
    mx = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    my = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float32)
    r0 = cv2.remap(I0, mx + f0[..., 0], my + f0[..., 1], cv2.INTER_LINEAR)
    r1 = cv2.remap(I1, mx + f1[..., 0], my + f1[..., 1], cv2.INTER_LINEAR)
    # Uniform blend (no exponential fading)
    return (((1 - t) * r0.astype(float) + t * r1.astype(float))).astype(np.uint8)


def full_pipeline(I0, I1, t=0.5, alpha=1.0):
    """Full C2F pipeline with proxy + flow + exp fading + HUD mask."""
    proxy = ((1 - t) * I0.astype(float) + t *
             I1.astype(float)).astype(np.uint8)
    gp = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
    g0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    f0 = cv2.calcOpticalFlowFarneback(gp, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    f1 = cv2.calcOpticalFlowFarneback(gp, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = I0.shape[:2]
    mx = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    my = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float32)
    r0 = cv2.remap(I0, mx + f0[..., 0], my + f0[..., 1], cv2.INTER_LINEAR)
    r1 = cv2.remap(I1, mx + f1[..., 0], my + f1[..., 1], cv2.INTER_LINEAR)
    m0 = np.sqrt(f0[..., 0]**2 + f0[..., 1]**2)
    m1 = np.sqrt(f1[..., 0]**2 + f1[..., 1]**2)
    w0 = np.exp(-alpha * m0) * (1 - t)
    w1 = np.exp(-alpha * m1) * t
    w0_3 = w0[..., np.newaxis]
    w1_3 = w1[..., np.newaxis]
    denom = w0_3 + w1_3 + 1e-8
    result = ((w0_3 * r0.astype(float) + w1_3 *
              r1.astype(float)) / denom).astype(np.uint8)
    # HUD mask
    diff = np.abs(I0.astype(float) - I1.astype(float))
    static_mask = (diff.max(axis=2) == 0)
    result[static_mask] = I0[static_mask]
    return result


# ---- Evaluate all methods on a sample of scenarios ----
methods = {
    'Linear Blend': baseline_linear_blend,
    'Backward Flow': baseline_backward_flow,
    'Proxy Only (Ablation A)': ablation_proxy_only,
    'Proxy+Flow (Ablation B)': ablation_proxy_flow_no_fading,
    'Full C2F Pipeline': full_pipeline,
}

# Sample 20 representative scenarios (2 per category)
sample_ids = [1, 3, 11, 15, 19, 24, 26, 31, 36,
              40, 45, 48, 51, 55, 58, 63, 66, 76, 80, 85]

print("=" * 80)
print("BASELINE COMPARISON + ABLATION STUDY")
print("=" * 80)

results = {name: {'psnr': [], 'ssim': [], 'time': []} for name in methods}

for sid in sample_ids:
    sname = scenario_names[sid - 1]

    # Files are named like: s001_fps_horizontal_pan_60fps.mp4 and s001_fps_horizontal_pan_30fps.mp4
    # Let's locate them
    prefix = f"s{sid:03d}_{sname}"
    gt_path = os.path.join(DATASET_DIR, 'gt', f"{prefix}_60fps.mp4")
    inp_path = os.path.join(DATASET_DIR, 'input', f"{prefix}_30fps.mp4")

    if not os.path.isfile(gt_path) or not os.path.isfile(inp_path):
        # Try without the prefix just in case
        gt_path = os.path.join(DATASET_DIR, 'gt', f"{sname}_60fps.mp4")
        inp_path = os.path.join(DATASET_DIR, 'input', f"{sname}_30fps.mp4")
        if not os.path.isfile(gt_path) or not os.path.isfile(inp_path):
            continue

    cap_inp = cv2.VideoCapture(inp_path)
    cap_gt = cv2.VideoCapture(gt_path)

    if not cap_inp.isOpened() or not cap_gt.isOpened():
        continue

    # Read first 6 frames from input (30fps) - this gives us 5 pairs
    inp_frames = []
    for _ in range(6):
        ret, frame = cap_inp.read()
        if not ret:
            break
        inp_frames.append(frame)

    # Read the first 12 frames from GT (60fps)
    gt_frames = []
    for _ in range(12):
        ret, frame = cap_gt.read()
        if not ret:
            break
        gt_frames.append(frame)

    cap_inp.release()
    cap_gt.release()

    if len(inp_frames) < 3 or len(gt_frames) < 5:
        continue

    test_pairs = min(5, len(inp_frames) - 1)

    for method_name, method_fn in methods.items():
        psnrs, ssims_vals, times = [], [], []
        for i in range(test_pairs):
            I0 = inp_frames[i]
            I1 = inp_frames[i + 1]

            # The interpolated frame is at t=0.5, which corresponds to index 2*i + 1 in the 60fps GT sequence
            # Frame mapping:
            # Input 30fps:   I0(tgt=0), I1(tgt=2)
            # GT 60fps:      G0(tgt=0), G1(tgt=1), G2(tgt=2)
            gt_idx = 2 * i + 1
            if gt_idx >= len(gt_frames):
                continue

            GT = gt_frames[gt_idx]

            if I0.shape != GT.shape:
                continue

            start = time.perf_counter()
            pred = method_fn(I0, I1)
            elapsed = (time.perf_counter() - start) * 1000  # ms

            if pred.shape != GT.shape:
                pred = cv2.resize(pred, (GT.shape[1], GT.shape[0]))

            p = compute_psnr(pred, GT)
            s = compute_ssim(pred, GT)
            psnrs.append(p)
            ssims_vals.append(s)
            times.append(elapsed)

        if psnrs:
            results[method_name]['psnr'].extend(psnrs)
            results[method_name]['ssim'].extend(ssims_vals)
            results[method_name]['time'].extend(times)

# Print results
print(f"\n{'Method':<30} {'PSNR (dB)':>12} {'SSIM':>10} {'Time (ms)':>12}")
print("-" * 66)
for name, data in results.items():
    if data['psnr']:
        avg_p = np.mean(data['psnr'])
        avg_s = np.mean(data['ssim'])
        avg_t = np.mean(data['time'])
        print(f"{name:<30} {avg_p:>10.2f}   {avg_s:>8.4f}   {avg_t:>10.2f}")

# Save to CSV
with open(os.path.join(DATASET_DIR, 'baselines.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['method', 'mean_psnr', 'std_psnr', 'mean_ssim',
               'std_ssim', 'mean_time_ms', 'std_time_ms', 'n_frames'])
    for name, data in results.items():
        if data['psnr']:
            w.writerow([
                name,
                f"{np.mean(data['psnr']):.2f}",
                f"{np.std(data['psnr']):.2f}",
                f"{np.mean(data['ssim']):.4f}",
                f"{np.std(data['ssim']):.4f}",
                f"{np.mean(data['time']):.2f}",
                f"{np.std(data['time']):.2f}",
                len(data['psnr']),
            ])

print(f"\nResults saved to dataset/baselines.csv")
print("=" * 80)
