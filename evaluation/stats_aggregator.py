import csv
import numpy as np
import scipy.stats as stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


rows = list(csv.DictReader(open('dataset/results.csv')))
psnrs = [float(r['avg_psnr']) for r in rows]

# Overall 95% CI
mean_overall, h_overall = mean_confidence_interval(psnrs)
print(f"Overall PSNR: {mean_overall:.2f} ± {h_overall:.2f} dB (95% CI)")

# Per category
cats = {
    'Space Combat': range(58, 66),
    'Platformer': range(45, 51),
    'Extended FPS': range(85, 101),
    'Battle Royale': range(51, 58),
    'Fight/Sport': range(66, 76),
    'Racing': range(36, 45),
    'FPS Combat': range(11, 19),
    'HUD Stress': range(26, 36),
    'Character': range(19, 26),
    'FPS Camera': range(1, 11),
    'Stress Base': range(76, 85),
}

print("\nPer-Category 95% CI:")
print(f"{'Category':<15} {'Mean PSNR':>10}   {'95% CI':>8}")
print("-" * 40)
cat_data = {}
for cat_name, ids in cats.items():
    cat_psnrs = [float(r['avg_psnr']) for r in rows if int(r['id']) in ids]
    if not cat_psnrs:
        continue
    m, h = mean_confidence_interval(cat_psnrs)
    cat_data[cat_name] = cat_psnrs
    print(f"{cat_name:<15} {m:>10.2f} ± {h:>5.2f}")

# Paired t-test between top performing category and second best
top = cat_data['Space Combat']
second = cat_data['Platformer']
t_stat, p_val = stats.ttest_ind(top, second)
print(f"\nIndependent t-test (Space vs Platformer): p-value = {p_val:.4f}")
if p_val < 0.05:
    print("Statistically significant difference.")
else:
    print("No statistically significant difference.")

with open('dataset/statistics.txt', 'w') as f:
    f.write(
        f"Overall PSNR: {mean_overall:.2f} ± {h_overall:.2f} dB (95% CI)\n")
