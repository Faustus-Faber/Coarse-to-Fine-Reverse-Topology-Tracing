import csv
import numpy as np

rows = list(csv.DictReader(
    open(r'C:\Users\farha\OneDrive\Desktop\Planning\dataset\results.csv')))

cats = {
    'FPS Camera (1-10)': range(1, 11),
    'FPS Combat (11-18)': range(11, 19),
    'Character Motion (19-25)': range(19, 26),
    'HUD Stress (26-35)': range(26, 36),
    'Racing (36-44)': range(36, 45),
    'Platformer (45-50)': range(45, 51),
    'Battle Royale (51-57)': range(51, 58),
    'Space Combat (58-65)': range(58, 66),
    'Fight/Sport (66-75)': range(66, 76),
    'Stress Tests (76-84)': range(76, 85),
    'Extended FPS (85-100)': range(85, 101),
}

print("Category | N | Mean PSNR | Std PSNR | Mean SSIM | Std SSIM")
print("-" * 70)
for name, ids in cats.items():
    cat_rows = [r for r in rows if int(r['id']) in ids]
    psnrs = [float(r['avg_psnr']) for r in cat_rows]
    ssims = [float(r['avg_ssim']) for r in cat_rows]
    print(f"{name:25s} | {len(cat_rows):2d} | {np.mean(psnrs):6.2f} | {np.std(psnrs):5.2f} | {np.mean(ssims):.4f} | {np.std(ssims):.4f}")

all_sorted = sorted(rows, key=lambda r: float(r['avg_psnr']))
print("\nBottom 5 PSNR:")
for r in all_sorted[:5]:
    print(
        f"  {r['name']:30s} {float(r['avg_psnr']):6.2f} dB  SSIM={float(r['avg_ssim']):.4f}")
print("\nTop 5 PSNR:")
for r in all_sorted[-5:]:
    print(
        f"  {r['name']:30s} {float(r['avg_psnr']):6.2f} dB  SSIM={float(r['avg_ssim']):.4f}")
