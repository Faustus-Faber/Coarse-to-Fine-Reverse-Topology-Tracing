import cv2
import numpy as np


def generate_complex_video(output_path, fps=30.0, duration=2.0, size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    frames = int(fps * duration)
    w, h = size

    for i in range(frames):
        # Create a dynamic, panning background
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        bg_shift = (i * 2) % 40
        for y in range(-40, h, 40):
            for x in range(-40, w, 40):
                if ((x - bg_shift) // 40 + (y - bg_shift) // 40) % 2 == 0:
                    cv2.rectangle(frame, (x - bg_shift, y - bg_shift),
                                  (x - bg_shift + 40, y - bg_shift + 40), (40, 40, 60), -1)
                else:
                    cv2.rectangle(frame, (x - bg_shift, y - bg_shift),
                                  (x - bg_shift + 40, y - bg_shift + 40), (80, 80, 100), -1)

        # Draw a moving foreground object 1 (Fast moving circle with non-linear sine path)
        c1_x = int(50 + i * 8.5)
        c1_y = int(h / 2 + np.sin(i * 0.3) * 60)
        cv2.circle(frame, (c1_x, c1_y), 35, (0, 0, 255), -1)

        # Draw a moving foreground object 2 (Rotating square moving in opposite direction)
        rect_center = (w - int(50 + i * 6), int(h / 3))
        angle = i * 12
        rect_pts = np.array([[-30, -30], [30, -30], [30, 30], [-30, 30]])
        transformed_pts = []
        for pt in rect_pts:
            x_new = pt[0] * np.cos(np.radians(angle)) - pt[1] * \
                np.sin(np.radians(angle)) + rect_center[0]
            y_new = pt[0] * np.sin(np.radians(angle)) + pt[1] * \
                np.cos(np.radians(angle)) + rect_center[1]
            transformed_pts.append([int(x_new), int(y_new)])
        cv2.fillPoly(frame, [np.array(transformed_pts)], (0, 255, 0))

        # Overlay a semi-transparent sweeping bar to simulate global occlusion sweep
        overlay = frame.copy()
        bar_x = int(w/2 + np.cos(i * 0.15) * 120)
        cv2.rectangle(overlay, (bar_x, 0), (bar_x + 50, h), (255, 100, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        out.write(frame)

    out.release()
    print(f"Generated complex video: {output_path}")


if __name__ == '__main__':
    generate_complex_video(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\complex_input_30fps.mp4')
