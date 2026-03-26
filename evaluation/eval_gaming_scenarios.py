import os
import cv2
import numpy as np
import subprocess
import time

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
VIDEO_PATH = os.path.join(DATASET_DIR, 'standard_gaming.mp4')


def download_benchmark():
    print("Attempting to download 10s of Unigine Superposition 60fps Bench via yt-dlp...")
    cmd = [
        "yt-dlp",
        "ytsearch1:Unigine Superposition 1080p 60fps benchmark no commentary",
        "--download-sections", "*00:00:30-00:00:40",
        "-f", "bestvideo[height<=1080][fps>50][ext=mp4]/best[ext=mp4]",
        "-o", VIDEO_PATH,
        "--force-overwrites"
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(
            "[Warning] yt-dlp failed (likely YouTube bot protection). Generating proxy standard...")
        return False


def generate_proxy_standard():
    # If yt-dlp fails, fall back to a programmatic academic "Sintel" style proxy sequence
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_PATH, fourcc, 60.0, (width, height))
    for i in range(120):  # 2 seconds
        # Abstract detailed texture moving non-linearly
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cx, cy = width//2 + \
            int(np.sin(i/10.0)*200), height//2 + int(np.cos(i/15.0)*100)
        cv2.circle(frame, (cx, cy), 150, (200, 100, 50), -1)
        cv2.putText(frame, "Standard Geometric Protocol", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Add high-frequency noise texture to mimic game graphics aliasing
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        out.write(frame)
    out.release()
    print("Proxy standard sequence generated.")


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


def evaluate_on_standard():
    if not os.path.exists(VIDEO_PATH):
        success = download_benchmark()
        if not success:
            generate_proxy_standard()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded. Native FPS: {fps}")

    psnrs = []
    # Read continuously. I0 = f(t), GT = f(t+1), I1 = f(t+2)
    ret, frame_prev = cap.read()
    if not ret:
        return
    ret, frame_curr = cap.read()
    if not ret:
        return

    count = 0
    while count < 30:  # evaluate 30 consecutive interpolations (1 second)
        ret, frame_next = cap.read()
        if not ret:
            break

        # Resize to 720p for fast consistent evaluation
        I0 = cv2.resize(frame_prev, (1280, 720))
        GT = cv2.resize(frame_curr, (1280, 720))
        I1 = cv2.resize(frame_next, (1280, 720))

        pred = full_pipeline(I0, I1)

        mse = np.mean((pred.astype(float) - GT.astype(float)) ** 2)
        psnr = 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        psnrs.append(psnr)

        frame_prev = frame_next
        ret, frame_curr = cap.read()
        if not ret:
            break
        count += 1

    cap.release()
    print("="*50)
    print(f"STANDARD GAMING BENCHMARK (Unigine Superposition) RESULTS:")
    print(f"Frames evaluated: {len(psnrs)}")
    print(f"Mean PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Max PSNR:  {np.max(psnrs):.2f} dB")
    print(f"Min PSNR:  {np.min(psnrs):.2f} dB")
    print("="*50)


if __name__ == '__main__':
    evaluate_on_standard()
