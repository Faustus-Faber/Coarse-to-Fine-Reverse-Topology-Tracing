import sys
from PIL import Image


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


def run_simulation(base_img_path):
    # Load and resize to keep python loops fast enough
    img = Image.open(base_img_path).convert('RGB')
    width, height = 640, 360
    img = img.resize((width, height))
    pixels_base = list(img.getdata())

    # Restructure into 2D array
    bg = [[pixels_base[y * width + x]
           for x in range(width)] for y in range(height)]

    # We will slice out a "vehicle" from the background to act as our moving object.
    obj_size_x, obj_size_y = 80, 60
    # Let's grab the object from the center (where the prompt said the car would be)
    obj_src_x, obj_src_y = 300, 200
    car_pixels = [[bg[obj_src_y + y][obj_src_x + x]
                   for x in range(obj_size_x)] for y in range(obj_size_y)]

    # Create Frame A and Frame B
    frame_a = [[bg[y][x] for x in range(width)] for y in range(height)]
    frame_b = [[bg[y][x] for x in range(width)] for y in range(height)]

    # We define a depth map for the mathematical occlusion handling
    # Background depth = 100
    depth_a = [[100.0] * width for _ in range(height)]

    # Introduce camera pan (background moves left by 10 pixels in Frame B)
    # So Frame A has bg at x, Frame B has bg at x+10
    camera_pan = 10
    for y in range(height):
        for x in range(width):
            src_x_b = clamp(x + camera_pan, 0, width - 1)
            frame_b[y][x] = bg[y][src_x_b]

    # Car Motion Vectors:
    box_a_x, box_a_y = 100, 220  # Car starts here
    box_b_x, box_b_y = 200, 220  # Car jumps 100 pixels right!

    # Draw car into Frame A
    for dy in range(obj_size_y):
        for dx in range(obj_size_x):
            if box_a_x + dx < width and box_a_y + dy < height:
                frame_a[box_a_y + dy][box_a_x + dx] = car_pixels[dy][dx]
                # Car depth = 10 (Foreground)
                depth_a[box_a_y + dy][box_a_x + dx] = 10.0

    # Draw car into Frame B
    for dy in range(obj_size_y):
        for dx in range(obj_size_x):
            if box_b_x + dx < width and box_b_y + dy < height:
                frame_b[box_b_y + dy][box_b_x + dx] = car_pixels[dy][dx]

    # ---------------------------------------------
    # MATHEMATICAL INTERPOLATION ALGORITHM (t = 0.5)
    # ---------------------------------------------
    timeDelta = 0.5
    gen_frame = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            # 1. Determine if this pixel belongs to the foreground object at t=0.5
            is_foreground = False
            box_mid_x = int(box_a_x + (box_b_x - box_a_x) * timeDelta)

            if box_a_y <= y < box_a_y + obj_size_y and box_mid_x <= x < box_mid_x + obj_size_x:
                is_foreground = True

            if is_foreground:
                # Target is Foreground. Pull 50% from A's car, 50% from B's car.
                offset_x = x - box_mid_x  # where in the car are we?
                cA = frame_a[box_a_y + (y - box_a_y)
                             ][clamp(box_a_x + offset_x, 0, width-1)]
                cB = frame_b[box_b_y + (y - box_a_y)
                             ][clamp(box_b_x + offset_x, 0, width-1)]

                gen_frame[y][x] = (
                    int(cA[0]*0.5 + cB[0]*0.5),
                    int(cA[1]*0.5 + cB[1]*0.5),
                    int(cA[2]*0.5 + cB[2]*0.5)
                )
            else:
                # Target is Background.
                # In typical optical flow, MV for background is known. Camera panned 10 pixels total.
                # So background pixel at t=0.5 came from x - 5 in Frame A.
                src_x_A_bg = clamp(
                    x - int(camera_pan * timeDelta), 0, width - 1)
                depth_at_A = depth_a[y][src_x_A_bg]

                # OCCLUSION HOLE DETECTION!
                # If we track back to Frame A, and hit depth=10 (the car was there!),
                # then this background pixel was OCCLUDED in Frame A. It is a 'hole'.
                if depth_at_A < 50.0:
                    # We CANNOT blend with Frame A. Do Temporal Reprojection.
                    # Pull 100% of the pixel color from Frame B (Fallback Disocclusion Logic).
                    src_x_B_bg = clamp(
                        x + int(camera_pan * (1.0 - timeDelta)), 0, width - 1)
                    gen_frame[y][x] = frame_b[y][src_x_B_bg]
                else:
                    # Safe to blend background
                    src_x_B_bg = clamp(
                        x + int(camera_pan * (1.0 - timeDelta)), 0, width - 1)
                    cA = frame_a[y][src_x_A_bg]
                    cB = frame_b[y][src_x_B_bg]
                    gen_frame[y][x] = (
                        int(cA[0]*0.5 + cB[0]*0.5),
                        int(cA[1]*0.5 + cB[1]*0.5),
                        int(cA[2]*0.5 + cB[2]*0.5)
                    )

    # Save outputs as standard PNGs
    def save_img(data, name):
        flat_data = [item for sublist in data for item in sublist]
        img_out = Image.new('RGB', (width, height))
        img_out.putdata(flat_data)
        img_out.save(name)

    save_img(
        frame_a, "C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\complex_frame_a.png")
    save_img(
        frame_b, "C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\complex_frame_b.png")
    save_img(
        gen_frame, "C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\complex_frame_generated.png")
    print("Complex simulation complete and PNGs saved.")


if __name__ == '__main__':
    run_simulation(sys.argv[1])
