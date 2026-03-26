import cv2
import numpy as np
import os
import csv
import time

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


def compute_psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255.0 ** 2 / mse)


def test_sensitivity(I0, I1, GT, alpha=1.0, pyr_scale=0.5, levels=3):
    t = 0.5
    proxy = ((1 - t) * I0.astype(float) + t *
             I1.astype(float)).astype(np.uint8)
    gp = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
    g0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    # Target parameter exploration
    f0 = cv2.calcOpticalFlowFarneback(
        gp, g0, None, pyr_scale, levels, 15, 3, 5, 1.2, 0)
    f1 = cv2.calcOpticalFlowFarneback(
        gp, g1, None, pyr_scale, levels, 15, 3, 5, 1.2, 0)

    h, w = I0.shape[:2]
    mx = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    my = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float32)
    r0 = cv2.remap(I0, mx + f0[..., 0], my + f0[..., 1], cv2.INTER_LINEAR)
    r1 = cv2.remap(I1, mx + f1[..., 0], my + f1[..., 1], cv2.INTER_LINEAR)

    m0 = np.sqrt(f0[..., 0]**2 + f0[..., 1]**2)
    m1 = np.sqrt(f1[..., 0]**2 + f1[..., 1]**2)
    w0 = np.exp(-alpha * m0) * (float(1 - t))
    w1 = np.exp(-alpha * m1) * float(t)
    w0_3 = w0[..., np.newaxis]
    w1_3 = w1[..., np.newaxis]

    denom = w0_3 + w1_3 + 1e-8
    result = ((w0_3 * r0.astype(float) + w1_3 *
              r1.astype(float)) / denom).astype(np.uint8)

    # HUD mask
    diff = np.abs(I0.astype(float) - I1.astype(float))
    static_mask = (diff.max(axis=2) == 0)
    result[static_mask] = I0[static_mask]

    if result.shape != GT.shape:
        result = cv2.resize(result, (GT.shape[1], GT.shape[0]))

    return compute_psnr(result, GT)


# Subset of scenarios (5 diverse scenarios)
rows = list(csv.DictReader(open(os.path.join(DATASET_DIR, 'results.csv'))))
scenario_names = [r['name'] for r in rows]
sample_ids = [3, 24, 45, 63, 80]  # FPS, Character, Platformer, Space, Stress

alphas = [0.1, 0.5, 1.0, 3.0, 5.0]
pyr_scales = [0.3, 0.5, 0.7]
levels_list = [2, 3, 5]

results = {
    'alpha': {a: [] for a in alphas},
    'pyr_scale': {p: [] for p in pyr_scales},
    'levels': {l: [] for l in levels_list}
}

for sid in sample_ids:
    sname = scenario_names[sid - 1]
    prefix = f"s{sid:03d}_{sname}"
    gt_path = os.path.join(DATASET_DIR, 'gt', f"{prefix}_60fps.mp4")
    inp_path = os.path.join(DATASET_DIR, 'input', f"{prefix}_30fps.mp4")

    if not os.path.isfile(gt_path):
        gt_path = os.path.join(DATASET_DIR, 'gt', f"{sname}_60fps.mp4")
        inp_path = os.path.join(DATASET_DIR, 'input', f"{sname}_30fps.mp4")
        if not os.path.isfile(gt_path):
            continue

    cap_inp = cv2.VideoCapture(inp_path)
    cap_gt = cv2.VideoCapture(gt_path)

    ret_i0, I0 = cap_inp.read()
    ret_i1, I1 = cap_inp.read()
    ret_g0, G0 = cap_gt.read()  # idx 0
    ret_g1, GT = cap_gt.read()  # idx 1 (t=0.5)

    cap_inp.release()
    cap_gt.release()

    if not (ret_i0 and ret_i1 and ret_g1):
        continue

    # Alpha sensitivity
    for a in alphas:
        p = test_sensitivity(I0, I1, GT, alpha=a, pyr_scale=0.5, levels=3)
        results['alpha'][a].append(p)

    # Pyr_scale sensitivity
    for p in pyr_scales:
        v = test_sensitivity(I0, I1, GT, alpha=1.0, pyr_scale=p, levels=3)
        results['pyr_scale'][p].append(v)

    # Levels sensitivity
    for l in levels_list:
        v = test_sensitivity(I0, I1, GT, alpha=1.0, pyr_scale=0.5, levels=l)
        results['levels'][l].append(v)

print("="*50)
print("ALPHA DECAY RATE SENSITIVITY")
for a in alphas:
    print(f"Alpha={a:<4}: {np.mean(results['alpha'][a]):.2f} dB")

print("\nFARNEBACK PYRAMID SCALE SENSITIVITY")
for p in pyr_scales:
    print(f"Scale={p:<4}: {np.mean(results['pyr_scale'][p]):.2f} dB")

print("\nFARNEBACK LEVELS SENSITIVITY")
for l in levels_list:
    print(f"Levels={l:<3}: {np.mean(results['levels'][l]):.2f} dB")
print("="*50)
