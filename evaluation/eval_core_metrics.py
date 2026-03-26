import cv2
import numpy as np


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def evaluate(gt_path, pred_path, name):
    print(f"\nEvaluating dataset: {name}")
    calc_cap = cv2.VideoCapture(pred_path)
    gt_cap = cv2.VideoCapture(gt_path)

    psnr_list = []

    frame_idx = 0
    while True:
        ret_c, f_c = calc_cap.read()
        ret_g, f_g = gt_cap.read()
        if not (ret_c and ret_g):
            break

        # Evaluate only the statistically generated midway frames (odd index)
        if frame_idx % 2 != 0:
            psnr = compute_psnr(f_g.astype(np.float32), f_c.astype(np.float32))
            psnr_list.append(psnr)

        frame_idx += 1

    if len(psnr_list) > 0:
        avg_psnr = np.mean(psnr_list)
        print(f"-> Extrapolated Frame Quality (PSNR): {avg_psnr:.2f} dB")
    else:
        print("-> Evaluation Failed (No frames compared)")


if __name__ == '__main__':
    evaluate('C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_gaming_60fps_gt.mp4',
             'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_gaming_60fps_pred.mp4',
             'GAMING FAST-PAN SCENARIO')
    evaluate('C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_shmup_60fps_gt.mp4',
             'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\dataset_shmup_60fps_pred.mp4',
             'BULLET HELL SCENARIO')
