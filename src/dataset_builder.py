import cv2
import numpy as np
import os


def create_gaming_dataset():
    print("Synthesizing Gaming Benchmark Dataset (60fps GT + 30fps Input)")
    fps = 60.0
    duration = 2.0
    size = (640, 480)
    w, h = size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_gt = cv2.VideoWriter(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_gaming_60fps_gt.mp4', fourcc, fps, size)
    out_in = cv2.VideoWriter(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_gaming_30fps_in.mp4', fourcc, 30.0, size)

    frames = int(fps * duration)
    bg = np.zeros((h, w*3, 3), dtype=np.uint8)
    for y in range(0, h, 40):
        for x in range(0, w*3, 40):
            if (x//40 + y//40) % 2 == 0:
                cv2.rectangle(bg, (x, y), (x+40, y+40), (255, 255, 255), -1)

    for i in range(frames):
        # 7.5 pixels per frame at 60 FPS (equivalent to 15 at 30 FPS)
        cam_x = int(w + i * 7.5)
        frame = bg[:, cam_x:cam_x+w].copy()

        # Player scaling
        enemy_x = int(w/2 - 100 + i * 4)
        enemy_y = int(h/2 - 50 + i * 1)
        scale = 1.0 + i * 0.01

        box_w = int(40 * scale)
        box_h = int(80 * scale)
        cv2.rectangle(frame, (enemy_x, enemy_y),
                      (enemy_x+box_w, enemy_y+box_h), (0, 0, 200), -1)
        cv2.circle(frame, (enemy_x + box_w//2, enemy_y + box_h//2),
                   int(15*scale), (0, 0, 255), -1)

        # Static HUD
        cv2.rectangle(frame, (10, 10), (120, 50), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_gt.write(frame)
        if i % 2 == 0:
            out_in.write(frame)

    out_gt.release()
    out_in.release()


def create_shmup_dataset():
    print("Synthesizing Shmup (Bullet Hell) Benchmark Dataset (60fps GT + 30fps Input)")
    fps = 60.0
    duration = 2.0
    size = (640, 480)
    w, h = size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_gt = cv2.VideoWriter(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_shmup_60fps_gt.mp4', fourcc, fps, size)
    out_in = cv2.VideoWriter(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_shmup_30fps_in.mp4', fourcc, 30.0, size)

    frames = int(fps * duration)
    np.random.seed(42)
    stars_x = np.random.randint(0, w, 150)
    stars_y = np.random.randint(0, h, 150)
    stars_speed = np.random.randint(2, 6, 150)  # Star parallax speed

    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        for s in range(150):
            # 60 FPS speed scaling
            sy = int(stars_y[s] + (i * stars_speed[s]) / 2.0) % h
            sx = stars_x[s]
            c = int(50 + (stars_speed[s] - 2) * 50)
            cv2.circle(frame, (sx, sy), 1, (c, c, c), -1)

        # Player Ship
        player_x = int(w/2 + np.sin(i * 0.075) * 200)
        player_y = h - 60 + int(np.cos(i * 0.15) * 10)
        pt1, pt2, pt3 = (player_x, player_y - 20), (player_x -
                                                    15, player_y + 15), (player_x + 15, player_y + 15)
        cv2.drawContours(
            frame, [np.array([pt1, pt2, pt3])], 0, (0, 255, 0), -1)

        # Enemy Hexagon
        enemy_x = int(w/2 + np.cos(i * 0.05) * 250)
        enemy_y = 100 + int(np.sin(i * 0.1) * 20)
        e_size = 40
        hex_pts = []
        for angle in range(0, 360, 60):
            hx = int(enemy_x + e_size * np.cos(np.radians(angle + i*2.5)))
            hy = int(enemy_y + e_size * np.sin(np.radians(angle + i*2.5)))
            hex_pts.append([hx, hy])
        cv2.fillPoly(frame, [np.array(hex_pts)], (0, 0, 200))

        # Laser Bullets
        for b in range(frames):
            # Fired every 10 frames (was 5 at 30 FPS)
            if b % 10 == 0 and b < i:
                b_age = i - b
                bx = int(w/2 + np.sin(b * 0.075) * 200)
                by = (h - 60) - b_age * 9  # Speed 9 per 60FPS frame
                if by > 0:
                    cv2.rectangle(frame, (bx-2, by-10),
                                  (bx+2, by+10), (0, 255, 255), -1)

            if b % 16 == 0 and b < i:
                b_age = i - b
                bx = int(w/2 + np.cos(b * 0.05) * 250)
                by = 100 + b_age * 6
                if by < h:
                    cv2.circle(frame, (bx, by), 6, (255, 0, 255), -1)

        # Static HUD
        cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, 38), (w, 40), (255, 255, 255), -1)

        out_gt.write(frame)
        if i % 2 == 0:
            out_in.write(frame)

    out_gt.release()
    out_in.release()


if __name__ == '__main__':
    create_gaming_dataset()
    create_shmup_dataset()
    print("Created academic dataset bounds successfully!")
