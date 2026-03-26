import os
import cv2
import numpy as np
import csv
import time

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


def run_1080p_scalability_test():
    print("Loading existing frame pair from dataset...")
    rows = list(csv.DictReader(open(os.path.join(DATASET_DIR, 'results.csv'))))
    sname = rows[0]['name']
    prefix = f"s001_{sname}"
    inp_path = os.path.join(DATASET_DIR, 'input', f"{prefix}_30fps.mp4")
    if not os.path.isfile(inp_path):
        inp_path = os.path.join(DATASET_DIR, 'input', f"{sname}_30fps.mp4")

    cap = cv2.VideoCapture(inp_path)
    ret, I0_720 = cap.read()
    ret, I1_720 = cap.read()
    cap.release()

    if I0_720 is None or I1_720 is None:
        print(f"Error: Could not load frames from {inp_path}")
        return

    # Upscale to 1080p
    I0 = cv2.resize(I0_720, (1920, 1080))
    I1 = cv2.resize(I1_720, (1920, 1080))

    # Run pipeline on 1080p
    def full_pipeline(I0, I1, t=0.5, alpha=1.0):
        proxy = ((1 - t) * I0.astype(float) + t *
                 I1.astype(float)).astype(np.uint8)
        gp = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
        g0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        f0 = cv2.calcOpticalFlowFarneback(
            gp, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        f1 = cv2.calcOpticalFlowFarneback(
            gp, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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

    print("Running 10 1080p interpolations to average runtime...")
    times = []

    # Warmup
    _ = full_pipeline(I0, I1)

    for _ in range(10):
        start = time.perf_counter()
        pred = full_pipeline(I0, I1)
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)

    # 720p reference for scale factor
    print("\nRunning 720p reference...")
    t720 = []
    _ = full_pipeline(I0_720, I1_720)
    for _ in range(10):
        start = time.perf_counter()
        _ = full_pipeline(I0_720, I1_720)
        t720.append((time.perf_counter() - start) * 1000)
    avg_720 = np.mean(t720)

    print("="*40)
    print(f"1080p Runtime (Python): {avg_time:.2f} ms")
    print(f" 720p Runtime (Python): {avg_720:.2f} ms")
    print(f"Scaling Factor:         {avg_time/avg_720:.2f}x (Expected ~2.25x)")
    print("="*40)


if __name__ == '__main__':
    run_1080p_scalability_test()
