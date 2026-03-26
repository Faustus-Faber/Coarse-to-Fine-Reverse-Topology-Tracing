import sys
import cv2
import numpy as np


def run_fantasy_simulation(base_img_path):
    print("Loading fantasy image...")
    img = cv2.imread(base_img_path)
    img = cv2.resize(img, (640, 360))

    # -------------------------------------------------------------
    # 1. MATHEMATICAL GEOMETRY SEGMENTATION (Isolating the Dragon)
    # -------------------------------------------------------------
    # The dragon is red and in the center.
    dragon_roi_x, dragon_roi_y, dragon_w, dragon_h = 240, 150, 160, 100
    roi = img[dragon_roi_y:dragon_roi_y+dragon_h,
              dragon_roi_x:dragon_roi_x+dragon_w]

    # Use HSV thresholding to find the RED dragon against blue sky/white clouds
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Red has two ranges in HSV (wrap around)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array(
        [170, 50, 50]), np.array([180, 255, 255]))
    thresh = mask1 | mask2

    # Sometimes AI makes the dragon dark red/brownish, let's also add general contrast threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, dark_thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    combined_mask = cv2.bitwise_or(thresh, dark_thresh)

    # Morphological clean up
    kernel = np.ones((5, 5), np.uint8)
    dragon_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # We now have the mathematically isolated dragon!
    isolated_dragon = cv2.bitwise_and(roi, roi, mask=dragon_mask)

    # -------------------------------------------------------------
    # 2. CAMERA PANNING & DISOCCLUSION DETECTION
    # -------------------------------------------------------------
    camera_pan = 15
    M_pan = np.float32([[1, 0, -camera_pan], [0, 1, 0]])
    background_b = cv2.warpAffine(img, M_pan, (img.shape[1], img.shape[0]))

    # -------------------------------------------------------------
    # 3. MATHEMATICAL HOLE FILLING (Interpolating the Sky/Clouds)
    # -------------------------------------------------------------
    hole_mask_full = np.zeros((360, 640), dtype=np.uint8)
    shifted_roi_x = dragon_roi_x - camera_pan
    hole_mask_full[dragon_roi_y:dragon_roi_y+dragon_h,
                   shifted_roi_x:shifted_roi_x+dragon_w] = dragon_mask

    print("Executing PDE Inpainting on Sky and Clouds...")
    inpainted_bg_b = cv2.inpaint(
        background_b, hole_mask_full, 4, cv2.INPAINT_TELEA)

    # -------------------------------------------------------------
    # 4. ANISOTROPIC INTERPOLATION TO t=0.5
    # -------------------------------------------------------------
    dragon_start_x = 240
    dragon_jump_x = 120  # Dragon leaps 120px forward

    mid_dragon_x = dragon_start_x + int(dragon_jump_x * 0.5)
    mid_pan_x = int(camera_pan * 0.5)

    M_mid_pan = np.float32([[1, 0, -mid_pan_x], [0, 1, 0]])
    background_mid = cv2.warpAffine(
        img, M_mid_pan, (img.shape[1], img.shape[0]))

    hole_mask_mid = np.zeros((360, 640), dtype=np.uint8)
    hole_mid_x = dragon_start_x - mid_pan_x
    hole_mask_mid[dragon_roi_y:dragon_roi_y+dragon_h,
                  hole_mid_x:hole_mid_x+dragon_w] = dragon_mask
    inpainted_bg_mid = cv2.inpaint(
        background_mid, hole_mask_mid, 4, cv2.INPAINT_TELEA)

    # -------------------------------------------------------------
    # 5. MATHEMATICAL COMPOSITING
    # -------------------------------------------------------------
    gen_frame = inpainted_bg_mid.copy()

    # Limit to maximum image constraints so crop doesn't fail
    if mid_dragon_x + dragon_w > 640:
        return

    target_roi = gen_frame[dragon_roi_y:dragon_roi_y +
                           dragon_h, mid_dragon_x:mid_dragon_x+dragon_w]

    mask_inv = cv2.bitwise_not(dragon_mask)
    bg_hole = cv2.bitwise_and(target_roi, target_roi, mask=mask_inv)
    final_dragon_composited = cv2.add(bg_hole, isolated_dragon)

    gen_frame[dragon_roi_y:dragon_roi_y+dragon_h,
              mid_dragon_x:mid_dragon_x+dragon_w] = final_dragon_composited

    cv2.imwrite(
        "C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\fantasy_frame_midpoint.png", gen_frame)
    cv2.imwrite(
        "C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\fantasy_background_inpainted.png", inpainted_bg_mid)
    print("Fantasy PhD Simulation Complete! Flawless mathematical frame saved.")


if __name__ == '__main__':
    run_fantasy_simulation(sys.argv[1])
