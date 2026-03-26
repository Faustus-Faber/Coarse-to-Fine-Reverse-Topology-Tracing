"""
Master Pipeline: Generate 100 scenarios, interpolate, evaluate PSNR, export CSV.
"""
from scenarios_batch4 import BATCH_4
from scenarios_batch3 import BATCH_3
from scenarios_batch2 import BATCH_2
from scenarios_batch1 import BATCH_1
from scenario_engine import generate_scenario, W, H
import os
import sys
import csv
import time
import cv2
import numpy as np

BASE = r'C:\Users\farha\OneDrive\Desktop\Planning\dataset'
WORK = r'C:\Users\farha\OneDrive\Desktop\Planning'

sys.path.insert(0, WORK)

ALL_SCENARIOS = BATCH_1 + BATCH_2 + BATCH_3 + BATCH_4


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim_channel(a, b):
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sig_a2 = cv2.GaussianBlur(a*a, (11, 11), 1.5) - mu_a*mu_a
    sig_b2 = cv2.GaussianBlur(b*b, (11, 11), 1.5) - mu_b*mu_b
    sig_ab = cv2.GaussianBlur(a*b, (11, 11), 1.5) - mu_a*mu_b
    num = (2*mu_a*mu_b + C1)*(2*sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1)*(sig_a2 + sig_b2 + C2)
    ssim_map = num / den
    return np.mean(ssim_map)


def compute_ssim(img1, img2):
    vals = []
    for ch in range(3):
        vals.append(compute_ssim_channel(img1[:, :, ch], img2[:, :, ch]))
    return np.mean(vals)


def interpolate_pair(prev_frame, curr_frame):
    """Generate one intermediate frame at t=0.5 using our Coarse-to-Fine algorithm."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow_map_x, flow_map_y = np.meshgrid(np.arange(W), np.arange(H))

    coarse = cv2.addWeighted(prev_frame, 0.5, curr_frame, 0.5, 0)
    coarse_gray = cv2.cvtColor(coarse, cv2.COLOR_BGR2GRAY)

    flow_to_0 = cv2.calcOpticalFlowFarneback(
        coarse_gray, prev_gray, None, 0.5, 5, 25, 5, 7, 1.5, 0)
    flow_to_1 = cv2.calcOpticalFlowFarneback(
        coarse_gray, curr_gray, None, 0.5, 5, 25, 5, 7, 1.5, 0)

    for ch in range(2):
        flow_to_0[..., ch] = cv2.medianBlur(flow_to_0[..., ch], 5)
        flow_to_1[..., ch] = cv2.medianBlur(flow_to_1[..., ch], 5)

    map_x0 = (flow_map_x + flow_to_0[..., 0]).astype(np.float32)
    map_y0 = (flow_map_y + flow_to_0[..., 1]).astype(np.float32)
    warped_A = cv2.remap(prev_frame, map_x0, map_y0,
                         interpolation=cv2.INTER_LINEAR)

    map_x1 = (flow_map_x + flow_to_1[..., 0]).astype(np.float32)
    map_y1 = (flow_map_y + flow_to_1[..., 1]).astype(np.float32)
    warped_B = cv2.remap(curr_frame, map_x1, map_y1,
                         interpolation=cv2.INTER_LINEAR)

    mag_0 = np.sqrt(flow_to_0[..., 0]**2 + flow_to_0[..., 1]**2)
    mag_1 = np.sqrt(flow_to_1[..., 0]**2 + flow_to_1[..., 1]**2)
    w_A = np.expand_dims(np.exp(-mag_0)*0.5, -1)
    w_B = np.expand_dims(np.exp(-mag_1)*0.5, -1)
    s = w_A + w_B + 1e-5
    gen = (warped_A*(w_A/s) + warped_B*(w_B/s)).astype(np.uint8)

    static = (cv2.absdiff(prev_frame, curr_frame) == 0).all(axis=-1)
    gen[static] = prev_frame[static]
    return gen


def evaluate_scenario(sid, name):
    """Run interpolation + evaluate PSNR/SSIM for one scenario."""
    gt_path = os.path.join(BASE, 'gt', f's{sid:03d}_{name}_60fps.mp4')
    in_path = os.path.join(BASE, 'input', f's{sid:03d}_{name}_30fps.mp4')
    pred_path = os.path.join(
        BASE, 'predicted', f's{sid:03d}_{name}_60fps_pred.mp4')

    if not os.path.exists(in_path):
        return None

    # Read input frames (30fps)
    cap_in = cv2.VideoCapture(in_path)
    in_frames = []
    while True:
        ret, fr = cap_in.read()
        if not ret:
            break
        in_frames.append(fr)
    cap_in.release()

    if len(in_frames) < 2:
        return None

    # Generate predicted 60fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pred_path, fourcc, 60.0, (W, H))
    out.write(in_frames[0])
    for idx in range(1, len(in_frames)):
        gen = interpolate_pair(in_frames[idx-1], in_frames[idx])
        out.write(gen)
        out.write(in_frames[idx])
    out.release()

    # Read GT and predicted for comparison
    cap_gt = cv2.VideoCapture(gt_path)
    cap_pr = cv2.VideoCapture(pred_path)
    psnr_vals, ssim_vals = [], []
    fidx = 0
    while True:
        ret_g, fg = cap_gt.read()
        ret_p, fp = cap_pr.read()
        if not (ret_g and ret_p):
            break
        if fidx % 2 == 1:  # Only evaluate synthesized frames
            psnr_vals.append(compute_psnr(fg, fp))
            ssim_vals.append(compute_ssim(fg, fp))
        fidx += 1
    cap_gt.release()
    cap_pr.release()

    if len(psnr_vals) == 0:
        return None

    return {
        'id': sid,
        'name': name,
        'avg_psnr': np.mean(psnr_vals),
        'min_psnr': np.min(psnr_vals),
        'max_psnr': np.max(psnr_vals),
        'avg_ssim': np.mean(ssim_vals),
        'min_ssim': np.min(ssim_vals),
        'max_ssim': np.max(ssim_vals),
        'num_frames': len(psnr_vals),
    }


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 1: Generating 100 Ground Truth + Input Scenarios")
    print("=" * 60)
    t0 = time.time()
    for sid, name, fn in ALL_SCENARIOS:
        generate_scenario(sid, name, fn)
    gen_time = time.time() - t0
    print(f"\nGeneration complete in {gen_time:.1f}s\n")

    print("=" * 60)
    print("PHASE 2: Interpolation + PSNR/SSIM Evaluation")
    print("=" * 60)
    results = []
    t0 = time.time()
    for sid, name, fn in ALL_SCENARIOS:
        r = evaluate_scenario(sid, name)
        if r:
            results.append(r)
            print(
                f"  [{r['id']:03d}] {r['name']:30s} PSNR={r['avg_psnr']:6.2f} dB  SSIM={r['avg_ssim']:.4f}")
        else:
            print(f"  [{sid:03d}] {name:30s} SKIPPED")
    eval_time = time.time() - t0
    print(f"\nEvaluation complete in {eval_time:.1f}s\n")

    # Export CSV
    csv_path = os.path.join(BASE, 'results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
                                'id', 'name', 'avg_psnr', 'min_psnr', 'max_psnr', 'avg_ssim', 'min_ssim', 'max_ssim', 'num_frames'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Summary statistics
    if results:
        psnrs = [r['avg_psnr'] for r in results]
        ssims = [r['avg_ssim'] for r in results]
        print("=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"  Scenarios evaluated: {len(results)}")
        print(f"  Overall Mean PSNR:   {np.mean(psnrs):.2f} dB")
        print(f"  Overall Median PSNR: {np.median(psnrs):.2f} dB")
        print(f"  Overall Min PSNR:    {np.min(psnrs):.2f} dB")
        print(f"  Overall Max PSNR:    {np.max(psnrs):.2f} dB")
        print(f"  Overall Mean SSIM:   {np.mean(ssims):.4f}")
        print(f"  Overall Median SSIM: {np.median(ssims):.4f}")
        print(f"  CSV saved to: {csv_path}")
    print("\nDone!")
