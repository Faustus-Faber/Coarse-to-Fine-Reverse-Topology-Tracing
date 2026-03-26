import cv2
import numpy as np


def generate_shmup_video(output_path, fps=30.0, duration=2.0, size=(640, 480)):
    print("Generating simulated Bullet Hell 'Shmup' test video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    frames = int(fps * duration)
    w, h = size

    # 1. Pre-generate Starfield (Parallax Background)
    # 150 random stars. We will wrap them around as they scroll.
    np.random.seed(42)  # For consistent testing
    stars_x = np.random.randint(0, w, 150)
    stars_y = np.random.randint(0, h, 150)
    # Parallax: stars move at different speeds
    stars_speed = np.random.randint(2, 6, 150)

    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw & Update Starfield
        for s in range(150):
            sy = (stars_y[s] + i * stars_speed[s]) % h
            sx = stars_x[s]
            # Dimmer stars move slower, brighter move faster
            c = int(50 + (stars_speed[s] - 2) * 50)
            cv2.circle(frame, (sx, sy), 1, (c, c, c), -1)

        # 2. Draw Moving Player Ship
        # Moves horizontally with a slight sine wave in Y to make flow tracking non-linear.
        player_x = int(w/2 + np.sin(i * 0.15) * 200)
        player_y = h - 60 + int(np.cos(i * 0.3) * 10)

        # Player is a green triangle (Ship)
        pt1 = (player_x, player_y - 20)
        pt2 = (player_x - 15, player_y + 15)
        pt3 = (player_x + 15, player_y + 15)
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(frame, [triangle_cnt], 0, (0, 255, 0), -1)

        # 3. Draw Moving Enemy Boss
        # Large red hexagon moving in opposition.
        enemy_x = int(w/2 + np.cos(i * 0.1) * 250)
        enemy_y = 100 + int(np.sin(i * 0.2) * 20)
        e_size = 40
        hex_pts = []
        for angle in range(0, 360, 60):
            # Spinning slightly!
            hx = int(enemy_x + e_size * np.cos(np.radians(angle + i*5)))
            hy = int(enemy_y + e_size * np.sin(np.radians(angle + i*5)))
            hex_pts.append([hx, hy])
        cv2.fillPoly(frame, [np.array(hex_pts)], (0, 0, 200))

        # 4. BULLET HELL (The ultimate test of micro-geometry scaling failure in flow)
        # Fast yellow player lasers moving UP
        for b in range(frames):
            if b % 5 == 0 and b < i:  # Fire a bullet every 5 frames
                b_age = i - b
                # Fired from where player WAS
                bx = int(w/2 + np.sin(b * 0.15) * 200)
                by = (h - 60) - b_age * 18  # Very fast upward motion!
                if by > 0:
                    cv2.rectangle(frame, (bx-2, by-10),
                                  (bx+2, by+10), (0, 255, 255), -1)

        # Fast purple enemy orbs moving DOWN
        for b in range(frames):
            if b % 8 == 0 and b < i:
                b_age = i - b
                bx = int(w/2 + np.cos(b * 0.1) * 250)  # Fired from enemy
                by = 100 + b_age * 12  # Fast downward
                if by < h:
                    cv2.circle(frame, (bx, by), 6, (255, 0, 255), -1)

        # 5. STATIC HUD
        # Top banner covering the space behind it. Optical flow must absolutely freeze here.
        cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, 38), (w, 40), (255, 255, 255), -1)

        # Dummy "Score" and "Lives"
        for j in range(3):  # Lives
            cv2.circle(frame, (30 + j*25, 20), 8, (0, 255, 0), -1)

        out.write(frame)

    out.release()
    print(f"Generated Bullet Hell video: {output_path}")


if __name__ == '__main__':
    generate_shmup_video(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\shmup_input_30fps.mp4')
