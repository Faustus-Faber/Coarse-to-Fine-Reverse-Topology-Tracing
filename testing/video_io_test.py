import sys
import subprocess
try:
    import cv2
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "opencv-python-headless", "numpy"])
    import cv2
    import numpy as np


def generate_video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\test_input_30fps.mp4', fourcc, 30.0, (320, 240))
    for i in range(60):  # 2 seconds of video
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw checkered background
        for y in range(0, 240, 40):
            for x in range(0, 320, 40):
                if (x//40 + y//40) % 2 == 0:
                    cv2.rectangle(frame, (x, y), (x+40, y+40),
                                  (50, 50, 50), -1)
                else:
                    cv2.rectangle(frame, (x, y), (x+40, y+40),
                                  (100, 100, 100), -1)

        # Draw moving object (red circle moving fast to the right)
        x_pos = 20 + int(i * 4.5)
        cv2.circle(frame, (x_pos, 120), 20, (0, 0, 255), -1)
        out.write(frame)

    out.release()
    print("Generated 30 FPS baseline video: test_input_30fps.mp4")


def interpolate_video():
    print("Starting Dense Optical Flow Mathematical Interpolation Pipeline...")
    cap = cv2.VideoCapture(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\test_input_30fps.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\test_output_60fps.mp4', fourcc, 60.0, (320, 240))

    ret, prev_frame = cap.read()
    if not ret:
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out.write(prev_frame)

    count = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # MATHEMATICAL OPTICAL FLOW: Farneback Algorithm computes displacement
        # mathematically for every single pixel without AI evaluation.
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # MATHEMATICAL REPROJECTION: t=0.5
        h, w = prev_gray.shape
        flow_map_x, flow_map_y = np.meshgrid(np.arange(w), np.arange(h))

        # Warp Frame A forward
        map_x_fw = (flow_map_x - flow[..., 0] * 0.5).astype(np.float32)
        map_y_fw = (flow_map_y - flow[..., 1] * 0.5).astype(np.float32)
        warped_prev = cv2.remap(prev_frame, map_x_fw,
                                map_y_fw, interpolation=cv2.INTER_LINEAR)

        # Warp Frame B backward
        map_x_bw = (flow_map_x + flow[..., 0] * 0.5).astype(np.float32)
        map_y_bw = (flow_map_y + flow[..., 1] * 0.5).astype(np.float32)
        warped_curr = cv2.remap(curr_frame, map_x_bw,
                                map_y_bw, interpolation=cv2.INTER_LINEAR)

        # MATHEMATICAL COMPOSITING
        generated_frame = cv2.addWeighted(
            warped_prev, 0.5, warped_curr, 0.5, 0)

        # Write 60 FPS sequence: Intermediate Generated Frame -> Next Real Frame
        out.write(generated_frame)
        out.write(curr_frame)

        prev_frame = curr_frame
        prev_gray = curr_gray
        count += 1

    cap.release()
    out.release()
    print(
        f"Successfully processed {count} real frames and hallucinated {count-1} new frames.")
    print("Encoded into 60 FPS video: test_output_60fps.mp4")


if __name__ == '__main__':
    generate_video()
    interpolate_video()
