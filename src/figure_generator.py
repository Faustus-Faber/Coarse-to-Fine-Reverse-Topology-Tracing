"""
Visualization Agent — Publication-Quality Figure Generation
Generates figures for the IEEE paper using matplotlib.
Colorblind-safe palette, APA 7.0 / IEEE standards, 300 DPI.
"""
import os
import csv
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# APA 7.0 / IEEE figure settings
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette
CB_PALETTE = ['#0077BB', '#33BBEE', '#009988', '#EE7733',
              '#CC3311', '#EE3377', '#BBBBBB', '#000000']

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
rows = list(csv.DictReader(
    open(os.path.join(os.path.dirname(__file__), 'dataset', 'results.csv'))))

# ========================================================================
# FIGURE 1: Pipeline Architecture Diagram
# ========================================================================


def fig1_pipeline_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(6.9, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.4',
                     facecolor='#E8F4FD', edgecolor='#0077BB', linewidth=1.5)
    box_style2 = dict(boxstyle='round,pad=0.4',
                      facecolor='#FFF3E0', edgecolor='#EE7733', linewidth=1.5)
    box_out = dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                   edgecolor='#009988', linewidth=2)
    box_hud = dict(boxstyle='round,pad=0.4', facecolor='#FCE4EC',
                   edgecolor='#CC3311', linewidth=1.5)
    arrow = dict(arrowstyle='->', color='#333333', lw=1.5)

    # Input frames
    ax.text(1.5, 6, '$I_0$ (Frame 0)', ha='center', va='center',
            fontsize=9, fontweight='bold', bbox=box_style)
    ax.text(5.5, 6, '$I_1$ (Frame 1)', ha='center', va='center',
            fontsize=9, fontweight='bold', bbox=box_style)

    # Proxy synthesis
    ax.text(3.5, 4.5, 'Proxy Frame Synthesis\n$P_t = I_0(1{-}t) + I_1 \\cdot t$',
            ha='center', va='center', fontsize=9, bbox=box_style2)

    # Arrows to proxy
    ax.annotate('', xy=(3.0, 4.9), xytext=(1.5, 5.6), arrowprops=arrow)
    ax.annotate('', xy=(4.0, 4.9), xytext=(5.5, 5.6), arrowprops=arrow)

    # Optical flow blocks
    ax.text(1.8, 3.0, 'Dense Optical Flow\n$\\vec{V}_{t \\to 0}$',
            ha='center', va='center', fontsize=8, bbox=box_style)
    ax.text(5.2, 3.0, 'Dense Optical Flow\n$\\vec{V}_{t \\to 1}$',
            ha='center', va='center', fontsize=8, bbox=box_style)

    # Arrows from proxy to flows
    ax.annotate('', xy=(2.2, 3.5), xytext=(3.0, 4.1), arrowprops=arrow)
    ax.annotate('', xy=(4.8, 3.5), xytext=(4.0, 4.1), arrowprops=arrow)

    # Exponential fading
    ax.text(3.5, 1.6, 'Exponential Magnitude Fading\n$W_k = \\exp(-\\alpha ||\\vec{V}||) \\cdot w_k$',
            ha='center', va='center', fontsize=8, bbox=box_style2)

    # Arrows from flows to fading
    ax.annotate('', xy=(2.8, 2.0), xytext=(1.8, 2.5), arrowprops=arrow)
    ax.annotate('', xy=(4.2, 2.0), xytext=(5.2, 2.5), arrowprops=arrow)

    # Output
    ax.text(3.5, 0.4, 'Output Frame $\\hat{I}_t$',
            ha='center', va='center', fontsize=10, fontweight='bold', bbox=box_out)
    ax.annotate('', xy=(3.5, 0.8), xytext=(3.5, 1.2), arrowprops=arrow)

    # HUD mask branch
    ax.text(9.5, 4.5, 'Static HUD Mask\n$M = [|I_0 - I_1| = 0]$',
            ha='center', va='center', fontsize=8, bbox=box_hud)
    ax.text(9.5, 2.5, 'Bypass Warp\n(Copy Static Pixels)',
            ha='center', va='center', fontsize=8, bbox=box_hud)
    ax.annotate('', xy=(9.5, 2.9), xytext=(9.5, 4.0), arrowprops=arrow)

    # Connect I0/I1 to HUD
    ax.annotate('', xy=(8.5, 4.5), xytext=(6.5, 5.8),
                arrowprops=dict(arrowstyle='->', color='#CC3311', lw=1.2, linestyle='dashed'))

    # Connect HUD to output
    ax.annotate('', xy=(5.0, 0.5), xytext=(9.0, 2.1),
                arrowprops=dict(arrowstyle='->', color='#CC3311', lw=1.2, linestyle='dashed'))

    # Phase labels
    ax.text(12.5, 6.0, 'Phase 1:\nInput', ha='center',
            va='center', fontsize=7, color='gray')
    ax.text(12.5, 4.5, 'Phase 2:\nProxy', ha='center',
            va='center', fontsize=7, color='gray')
    ax.text(12.5, 3.0, 'Phase 3:\nFlow', ha='center',
            va='center', fontsize=7, color='gray')
    ax.text(12.5, 1.6, 'Phase 4:\nBlend', ha='center',
            va='center', fontsize=7, color='gray')
    ax.text(12.5, 0.4, 'Phase 5:\nOutput', ha='center',
            va='center', fontsize=7, color='gray')

    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_pipeline.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_pipeline.pdf'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Figure 1: Pipeline Architecture saved")


# ========================================================================
# FIGURE 2: Per-Category PSNR Bar Chart (Horizontal) with Error Bars
# ========================================================================
def fig2_psnr_bar_chart():
    cats = {
        'FPS Camera': (range(1, 11),),
        'FPS Combat': (range(11, 19),),
        'Character Motion': (range(19, 26),),
        'HUD Stress': (range(26, 36),),
        'Racing': (range(36, 45),),
        'Platformer': (range(45, 51),),
        'Battle Royale': (range(51, 58),),
        'Space Combat': (range(58, 66),),
        'Fight/Sport': (range(66, 76),),
        'Stress Tests': (range(76, 85),),
        'Extended FPS': (range(85, 101),),
    }

    names, means, stds = [], [], []
    for name, (ids,) in cats.items():
        cat_rows = [r for r in rows if int(r['id']) in ids]
        psnrs = [float(r['avg_psnr']) for r in cat_rows]
        names.append(name)
        means.append(np.mean(psnrs))
        stds.append(np.std(psnrs))

    # Sort by mean PSNR
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    y_pos = np.arange(len(names))

    colors = [CB_PALETTE[4] if m < 25 else CB_PALETTE[3]
              if m < 32 else CB_PALETTE[0] for m in means]
    bars = ax.barh(y_pos, means, xerr=stds, height=0.65, color=colors,
                   edgecolor='white', linewidth=0.5, capsize=3, error_kw={'linewidth': 1})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Mean PSNR (dB)')
    ax.set_xlim(0, 50)
    ax.axvline(x=33.29, color='#333333',
               linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(33.29 + 0.5, len(names) - 0.5, f'Overall Mean\n(33.29 dB)',
            fontsize=7, color='#333333', va='top')

    # Value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.5, i, f'{m:.1f}',
                va='center', fontsize=7, fontweight='bold')

    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_psnr_categories.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_psnr_categories.pdf'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Figure 2: PSNR Bar Chart saved")


# ========================================================================
# FIGURE 3: PSNR vs SSIM Scatter Plot
# ========================================================================
def fig3_psnr_ssim_scatter():
    # Assign categories
    cat_map = {}
    cat_labels = {
        'FPS': range(1, 26),
        'HUD': range(26, 36),
        'Racing': range(36, 45),
        'Platformer': range(45, 51),
        'Battle Royale': range(51, 58),
        'Space': range(58, 66),
        'Fight/Sport': range(66, 76),
        'Stress': range(76, 85),
        'Extended FPS': range(85, 101),
    }
    for label, ids in cat_labels.items():
        for i in ids:
            cat_map[i] = label

    palette_map = {
        'FPS': '#0077BB',
        'HUD': '#33BBEE',
        'Racing': '#009988',
        'Platformer': '#EE7733',
        'Battle Royale': '#CC3311',
        'Space': '#EE3377',
        'Fight/Sport': '#BBBBBB',
        'Stress': '#000000',
        'Extended FPS': '#0077BB',
    }
    markers = {
        'FPS': 'o', 'HUD': 's', 'Racing': 'D', 'Platformer': '^',
        'Battle Royale': 'v', 'Space': 'P', 'Fight/Sport': 'X',
        'Stress': '*', 'Extended FPS': 'h',
    }

    fig, ax = plt.subplots(figsize=(6.9, 4.5))

    for label in cat_labels:
        ids = cat_labels[label]
        cat_rows = [r for r in rows if int(r['id']) in ids]
        psnrs = [float(r['avg_psnr']) for r in cat_rows]
        ssims = [float(r['avg_ssim']) for r in cat_rows]
        ax.scatter(psnrs, ssims, c=palette_map[label], marker=markers[label],
                   s=40, alpha=0.8, label=label, edgecolors='white', linewidths=0.3)

    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('SSIM')
    ax.set_xlim(8, 48)
    ax.set_ylim(0.15, 1.05)
    ax.legend(loc='lower right', ncol=2, framealpha=0.9, fontsize=7)

    # Add quadrant lines
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axvline(x=30, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_psnr_ssim_scatter.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_psnr_ssim_scatter.pdf'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Figure 3: PSNR vs SSIM Scatter saved")


# ========================================================================
# FIGURE 4: PSNR Distribution Boxplot by Category
# ========================================================================
def fig4_psnr_boxplot():
    cat_labels_ordered = [
        ('Space', range(58, 66)),
        ('Platformer', range(45, 51)),
        ('Ext. FPS', range(85, 101)),
        ('Battle Royale', range(51, 58)),
        ('Fight/Sport', range(66, 76)),
        ('Racing', range(36, 45)),
        ('FPS Combat', range(11, 19)),
        ('HUD Stress', range(26, 36)),
        ('Char Motion', range(19, 26)),
        ('FPS Camera', range(1, 11)),
        ('Stress Tests', range(76, 85)),
    ]

    data = []
    labels = []
    for label, ids in cat_labels_ordered:
        cat_rows = [r for r in rows if int(r['id']) in ids]
        psnrs = [float(r['avg_psnr']) for r in cat_rows]
        data.append(psnrs)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    bp = ax.boxplot(data, vert=False, labels=labels, patch_artist=True,
                    medianprops=dict(color='#CC3311', linewidth=1.5),
                    whiskerprops=dict(color='#333333'),
                    capprops=dict(color='#333333'),
                    flierprops=dict(marker='o', markerfacecolor='#BBBBBB', markersize=4))

    colors_box = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(data))]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('PSNR (dB)')
    ax.axvline(x=33.29, color='#333333',
               linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(33.5, 0.6, 'Mean: 33.29 dB', fontsize=7, color='#333333')

    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_psnr_boxplot.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_psnr_boxplot.pdf'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Figure 4: PSNR Boxplot saved")


# ========================================================================
# Generate all figures
# ========================================================================
if __name__ == '__main__':
    print("Generating publication-quality figures...")
    fig1_pipeline_architecture()
    fig2_psnr_bar_chart()
    fig3_psnr_ssim_scatter()
    fig4_psnr_boxplot()
    print("\nAll figures saved to:", FIGURES_DIR)
