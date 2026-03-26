import cv2
import numpy as np
import os
import csv
import torch
import lpips

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


def to_tensor(img):
    # Convert BGR (OpenCV) to RGB, normalize to [-1, 1], transpose to NCHW
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0)


# Initialize LPIPS
loss_fn_vgg = lpips.LPIPS(net='vgg')


def full_pipeline(I0, I1, t=0.5, alpha=1.0):
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
    diff = np.abs(I0.astype(float) - I1.astype(float))
    static_mask = (diff.max(axis=2) == 0)
    result[static_mask] = I0[static_mask]
    return result


def baseline_linear_blend(I0, I1, t=0.5):
    return ((1 - t) * I0.astype(float) + t * I1.astype(float)).astype(np.uint8)


# Evaluate
rows = list(csv.DictReader(open(os.path.join(DATASET_DIR, 'results.csv'))))
scenario_names = [r['name'] for r in rows]
sample_ids = [3, 24, 45, 63, 80]  # Diverse test set

lpips_c2f = []
lpips_linear = []

print("Running LPIPS Evaluation (Lower is better)...")

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
    ret_g0, G0 = cap_gt.read()
    ret_g1, GT = cap_gt.read()  # GT at t=0.5

    cap_inp.release()
    cap_gt.release()

    if not (ret_i0 and ret_i1 and ret_g1):
        continue

    pred_c2f = full_pipeline(I0, I1)
    pred_lin = baseline_linear_blend(I0, I1)

    if pred_c2f.shape != GT.shape:
        pred_c2f = cv2.resize(pred_c2f, (GT.shape[1], GT.shape[0]))
        pred_lin = cv2.resize(pred_lin, (GT.shape[1], GT.shape[0]))

    t_gt = to_tensor(GT)
    t_c2f = to_tensor(pred_c2f)
    t_lin = to_tensor(pred_lin)

    with torch.no_grad():
        score_c2f = loss_fn_vgg(t_c2f, t_gt).item()
        score_lin = loss_fn_vgg(t_lin, t_gt).item()

    lpips_c2f.append(score_c2f)
    lpips_linear.append(score_lin)

print("="*40)
print(f"Linear Blend LPIPS: {np.mean(lpips_linear):.4f}")
print(f"Full C2F LPIPS:     {np.mean(lpips_c2f):.4f}")
print("="*40)
