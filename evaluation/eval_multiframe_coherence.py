import os
import cv2
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


def generate_120fps_proxy():
    width, height = 1280, 720
    frames = []
    # Generate 10 consecutive frames at 120 FPS
    for i in range(10):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Fast moving object
        cx, cy = width//2 + \
            int(np.sin(i/5.0)*300), height//2 + int(np.cos(i/5.0)*200)
        cv2.circle(frame, (cx, cy), 100, (50, 200, 100), -1)
        # Static HUD
        cv2.putText(frame, "HUD ELEMENT", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(frame)
    return frames


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


def evaluate_4x():
    print("Generating 120 FPS high-motion proxy sequence...")
    frames = generate_120fps_proxy()

    # Target inputs: Frame 0 and Frame 4 (representing 30 FPS boundary)
    I0 = frames[0]
    I1 = frames[4]

    # Ground Truth intermediates: Frames 1, 2, 3 (120 FPS sequence)
    GT_025 = frames[1]  # t = 0.25
    GT_050 = frames[2]  # t = 0.50
    GT_075 = frames[3]  # t = 0.75

    print("Interpolating intermediate frame t=0.25...")
    pred_025 = full_pipeline(I0, I1, t=0.25)

    print("Interpolating intermediate frame t=0.50...")
    pred_050 = full_pipeline(I0, I1, t=0.50)

    print("Interpolating intermediate frame t=0.75...")
    pred_075 = full_pipeline(I0, I1, t=0.75)

    def calc_psnr(a, b):
        mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
        return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

    p25 = calc_psnr(pred_025, GT_025)
    p50 = calc_psnr(pred_050, GT_050)
    p75 = calc_psnr(pred_075, GT_075)

    print("="*50)
    print("MULTI-FRAME (4x) INTERPOLATION CAPABILITY TEST")
    print(f"t=0.25 PSNR: {p25:.2f} dB")
    print(f"t=0.50 PSNR: {p50:.2f} dB")
    print(f"t=0.75 PSNR: {p75:.2f} dB")
    print(f"Mean 3-Frame PSNR: {np.mean([p25, p50, p75]):.2f} dB")
    print("="*50)


if __name__ == '__main__':
    evaluate_4x()
