import sys
import cv2
import numpy as np
from typing import Optional


def generate_intermediate_frame(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    t: float,
    flow_map_x: np.ndarray,
    flow_map_y: np.ndarray,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """
    Generates an intermediate video frame at a given normalized time 't' using
    a coarse-to-fine Farneback Optical Flow model with pristine pixel gathering.

    Args:
        prev_frame (np.ndarray): Previous BGR frame (H, W, 3).
        curr_frame (np.ndarray): Next BGR frame (H, W, 3).
        t (float): Normalized temporal distance (0.0 to 1.0).
        flow_map_x (np.ndarray): Precomputed meshgrid X coordinates (H, W).
        flow_map_y (np.ndarray): Precomputed meshgrid Y coordinates (H, W).
        prev_gray (np.ndarray): Grayscale version of prev_frame (H, W).
        curr_gray (np.ndarray): Grayscale version of curr_frame (H, W).

    Returns:
        np.ndarray: The generated intermediate BGR frame (H, W, 3).
    """

    # 1. Structural Ghost Geolocation (Weighted for Arbitrary Time 't')
    weight_b: float = t
    weight_a: float = 1.0 - t
    coarse_t = cv2.addWeighted(prev_frame, weight_a, curr_frame, weight_b, 0)
    coarse_t_gray = cv2.cvtColor(coarse_t, cv2.COLOR_BGR2GRAY)

    # 2. Reverse Target Tracing (Originating exactly at sub-timeline t)
    flow_t_to_0 = cv2.calcOpticalFlowFarneback(
        coarse_t_gray, prev_gray, None, 0.5, 5, 25, 5, 7, 1.5, 0
    )
    flow_t_to_1 = cv2.calcOpticalFlowFarneback(
        coarse_t_gray, curr_gray, None, 0.5, 5, 25, 5, 7, 1.5, 0
    )

    # Mathematical Boundary Snapping
    flow_t_to_0[..., 0] = cv2.medianBlur(flow_t_to_0[..., 0], 5)
    flow_t_to_0[..., 1] = cv2.medianBlur(flow_t_to_0[..., 1], 5)
    flow_t_to_1[..., 0] = cv2.medianBlur(flow_t_to_1[..., 0], 5)
    flow_t_to_1[..., 1] = cv2.medianBlur(flow_t_to_1[..., 1], 5)

    # 3. Pristine Pixel Gathering
    map_x_to_0 = (flow_map_x + flow_t_to_0[..., 0]).astype(np.float32)
    map_y_to_0 = (flow_map_y + flow_t_to_0[..., 1]).astype(np.float32)
    warped_a = cv2.remap(prev_frame, map_x_to_0, map_y_to_0,
                         interpolation=cv2.INTER_LINEAR)

    map_x_to_1 = (flow_map_x + flow_t_to_1[..., 0]).astype(np.float32)
    map_y_to_1 = (flow_map_y + flow_t_to_1[..., 1]).astype(np.float32)
    warped_b = cv2.remap(curr_frame, map_x_to_1, map_y_to_1,
                         interpolation=cv2.INTER_LINEAR)

    # 4. Magnitude Cross-Fading using Continuous Topology
    mag_0 = np.sqrt(flow_t_to_0[..., 0] ** 2 + flow_t_to_0[..., 1] ** 2)
    mag_1 = np.sqrt(flow_t_to_1[..., 0] ** 2 + flow_t_to_1[..., 1] ** 2)

    alpha: float = 1.0
    weight_a_temporal = np.exp(-alpha * mag_0) * weight_a
    weight_b_temporal = np.exp(-alpha * mag_1) * weight_b

    sum_weights = weight_a_temporal + weight_b_temporal + 1e-5
    norm_w_a = np.expand_dims(weight_a_temporal / sum_weights, axis=-1)
    norm_w_b = np.expand_dims(weight_b_temporal / sum_weights, axis=-1)

    generated_frame = (warped_a * norm_w_a + warped_b *
                       norm_w_b).astype(np.uint8)

    # 5. Absolute Static Pinning of HUD/Zero-motional fields
    static_hud_mask = (cv2.absdiff(prev_frame, curr_frame) == 0).all(axis=-1)
    generated_frame[static_hud_mask] = prev_frame[static_hud_mask]

    return generated_frame


def interpolate_video(input_path: str, output_path: str, multiplier: int = 2) -> None:
    """
    Reads an input video and generates a new video with upscaled framerate.

    Args:
        input_path (str): Filepath of the source video.
        output_path (str): Filepath for the generated output video.
        multiplier (int): Frame extrapolation multiplier (default 2x).
    """
    print(
        f"Applying Generative Multi-Frame (x{multiplier}) Coarse-to-Fine Pipeline: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Failed to open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = fps * float(multiplier)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        return

    out.write(prev_frame)
    flow_map_x, flow_map_y = np.meshgrid(np.arange(width), np.arange(height))

    frame_count: int = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Generate sequence of arbitrary intermediate frames
        for m in range(1, multiplier):
            t = float(m) / float(multiplier)
            gen_frame = generate_intermediate_frame(
                prev_frame,
                curr_frame,
                t,
                flow_map_x,
                flow_map_y,
                prev_gray,
                curr_gray,
            )
            out.write(gen_frame)

        out.write(curr_frame)

        prev_frame = curr_frame
        frame_count += 1

        if frame_count % 10 == 0:
            print(
                f"Processed {frame_count} frames -> Generated {frame_count * multiplier} frames...")

    cap.release()
    out.release()
    print(
        f"Pipeline Complete! Rendered at {out_fps} FPS. Extrapolation saved to {output_path}")


if __name__ == "__main__":
    in_vid = sys.argv[1] if len(
        sys.argv) > 1 else r"C:\Users\farha\OneDrive\Desktop\Planning\shmup_input_30fps.mp4"
    frame_mult = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out_vid = sys.argv[3] if len(
        sys.argv) > 3 else rf"C:\Users\farha\OneDrive\Desktop\Planning\shmup_output_x{frame_mult}_phd.mp4"

    interpolate_video(in_vid, out_vid, frame_mult)
