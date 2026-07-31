import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy import stats
import pandas as pd

# ============================================================================
# SETUP
# ============================================================================

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11

COLORS = {
    'primary':   '#2A9D8F',
    'secondary': '#264653',
    'accent':    '#E76F51',
    'warning':   '#F4A261',
    'light':     '#B8DDD6',
    'original':  '#8AB6D6',
}

def set_figure_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(direction='out', length=4, width=1.2)
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)

# ============================================================================
# DATA
# ============================================================================

compounds = ['PCE', 'TCE', 'cis-DCE', '1,1-DCE', 'trans-DCE']
mean_k2 = [0.0346, 0.853, 1.498, 2.060, 48.34]
sd_k2 = [0.00207, 0.0819, 0.0517, 0.129, 0.713]
cv_percent = [5.99, 9.60, 3.45, 6.28, 1.47]
sd_ln_k2 = [0.0603, 0.0974, 0.0349, 0.0651, 0.0148]

temps_C = [10, 15, 20, 25]
temps_K = [283, 288, 293, 298]
inv_T = [1000/T for T in temps_K]

k2_data = {
    'PCE':       [0.021, 0.030, 0.0355, 0.051],
    'TCE':       [0.53,  0.72,  0.94,   1.19],
    'cis-DCE':   [0.82,  1.06,  1.445,  1.78],
    '1,1-DCE':   [1.30,  1.61,  2.08,   2.47],
    'trans-DCE': [33.7,  40.0,  47.7,   56.8]
}

arrhenius_results = {
    'PCE':       {'Ea': 9.49, 'dEa': 0.926, 'chi2': 1.024,
                  'Ea_orig': 9.3, 'dEa_orig': 0.9},
    'TCE':       {'Ea': 9.04, 'dEa': 1.462, 'chi2': 0.036,
                  'Ea_orig': 8.9, 'dEa_orig': 0.3},
    '1,1-DCE':   {'Ea': 7.32, 'dEa': 0.976, 'chi2': 0.117,
                  'Ea_orig': 6.9, 'dEa_orig': 0.7},
    'cis-DCE':   {'Ea': 8.84, 'dEa': 0.524, 'chi2': 0.585,
                  'Ea_orig': 8.8, 'dEa_orig': 0.5},
    'trans-DCE': {'Ea': 5.84, 'dEa': 0.221, 'chi2': 0.138,
                  'Ea_orig': 5.8, 'dEa_orig': 0.1}
}

# All ln(k2) values at 20°C, grouped by compound
ln_k2_by_compound = {
    'PCE':       [-3.324, -3.352, -3.411, -3.442, -3.297],
    'TCE':       [-0.051, -0.083, -0.174, -0.301, -0.248, -0.117],
    'cis-DCE':   [ 0.425,  0.385,  0.351,  0.432,  0.425],
    '1,1-DCE':   [ 0.732,  0.610,  0.765,  0.728,  0.770],
    'trans-DCE': [ 3.894,  3.871,  3.859,  3.875,  3.892]
}

# ============================================================================
# FIGURE 1: CV CONSISTENCY TEST
# ============================================================================

def make_figure1():
    compounds_reordered = ['PCE', 'TCE', '1,1-DCE', 'cis-DCE', 'trans-DCE']
    mean_k2_reordered   = [0.0346, 0.853, 2.060, 1.498, 48.34]
    sd_k2_reordered     = [0.00207, 0.0819, 0.129, 0.0517, 0.713]
    cv_percent_reordered = [5.99, 9.60, 6.28, 3.45, 1.47]
    sd_ln_k2_reordered  = [0.0603, 0.0974, 0.0651, 0.0349, 0.0148]

    fig, axes = plt.subplots(3, 1, figsize=(10, 13))

    # Panel A: SD(k2)
    ax = axes[0]
    ax.bar(compounds_reordered, sd_k2_reordered, color=COLORS['accent'],
           alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(r'$\sigma_{k_2}$ (M⁻¹s⁻¹)', fontsize=22, fontweight='bold')
    ax.set_xlabel('Compound', fontsize=22, fontweight='bold')
    ax.set_title('(A) Absolute Standard Deviation', fontsize=24, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1)
    ax.set_xlim(-0.75, len(compounds_reordered) - 0.25)

    # Panel B: CV%
    ax = axes[1]
    ax.bar(compounds_reordered, cv_percent_reordered, color=COLORS['warning'],
           alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('CV (%)', fontsize=22, fontweight='bold')
    ax.set_xlabel('Compound', fontsize=22, fontweight='bold')
    ax.set_title('(B) Coefficient of Variation', fontsize=24, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_ylim(0, 12)
    ax.set_xlim(-0.75, len(compounds_reordered) - 0.25)

    # Panel C: SD[ln(k2)]
    ax = axes[2]
    ax.bar(compounds_reordered, sd_ln_k2_reordered, color=COLORS['primary'],
           alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(r'$\sigma_{\ln(k_2)}$', fontsize=22, fontweight='bold')
    ax.set_xlabel('Compound', fontsize=22, fontweight='bold')
    ax.set_title('(C) Standard Deviation in Logarithmic Space', fontsize=24, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_ylim(0, 0.12)
    ax.set_xlim(-0.75, len(compounds_reordered) - 0.25)

    plt.tight_layout(h_pad=3.5)
    plt.savefig('CV_consistency.png', dpi=400, bbox_inches='tight')
    return fig

# ============================================================================
# FIGURE 2: Q-Q PLOTS
# ============================================================================

def make_figure2():
    #Normalise each compound's ln(k2) values by subtracting the per-compound
    #mean before pooling. This removes between-compound location differences
    #and isolates the distributional shape, making the Q-Q comparison meaningful.
    # --- Build normalised arrays ---
    normalised_ln_k2 = []
    for compound, values in ln_k2_by_compound.items():
        mu = np.mean(values)
        normalised_ln_k2.extend([v - mu for v in values])

    # Back-transform for panel A: exp(normalised ln k2) centres each compound at 1
    normalised_k2 = [np.exp(v) for v in normalised_ln_k2]



    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: normalised k2
    ax = axes[0]
    stats.probplot(normalised_k2, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(COLORS['accent'])
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(7)
    ax.get_lines()[0].set_markeredgewidth(1.5)
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[1].set_linestyle('--')
    ax.get_lines()[1].set_linewidth(1.5)
    ax.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Quantiles (normalised k₂)', fontsize=12, fontweight='bold')
    ax.set_title('A. Rate Constants', fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(axis='both', labelsize=11, width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)


    # Panel B: normalised ln(k2) (should be normal)
    ax = axes[1]
    stats.probplot(normalised_ln_k2, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(COLORS['primary'])
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(7)
    ax.get_lines()[0].set_markeredgewidth(1.5)
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[1].set_linestyle('--')
    ax.get_lines()[1].set_linewidth(1.5)
    ax.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Quantiles (normalised ln k₂)', fontsize=12, fontweight='bold')
    ax.set_title('B. Log-Transformed Rate Constants', fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(axis='both', labelsize=11, width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)


    plt.tight_layout()
    plt.savefig('QQ_plots.png', dpi=400, bbox_inches='tight')
    return fig

# ============================================================================
# ARRHENIUS PLOT
# ============================================================================

def plot_arrhenius(ax, compound, color, highlight=False):
    k2_vals = k2_data[compound]
    ln_k2 = [np.log(k) for k in k2_vals]

    z = np.polyfit(inv_T, ln_k2, 1)
    p = np.poly1d(z)
    ln_k2_fit = p(inv_T)

    errorbar = sd_ln_k2[compounds.index(compound)]
    ax.errorbar(inv_T, ln_k2, yerr=errorbar, fmt='o', markersize=8,
                color=color, markeredgecolor='black', markeredgewidth=1.5,
                elinewidth=2, capsize=4, capthick=2, label='Data', zorder=3)

    x_smooth = np.linspace(min(inv_T), max(inv_T), 100)
    ax.plot(x_smooth, p(x_smooth), '--', color='black', linewidth=1.5,
            alpha=0.7, label='Linear fit', zorder=2)

    residuals = np.array(ln_k2) - np.array(ln_k2_fit)

    res = arrhenius_results[compound]
    chi2 = res['chi2']

    if chi2 > 2:
        box_color = COLORS['accent']
        status = 'Poor fit'
    elif chi2 < 0.2:
        box_color = COLORS['warning']
        status = 'Suspicious'
    else:
        box_color = COLORS['primary']
        status = 'Good fit'

    text = f"χ² = {chi2:.2f}\n{status}\nEₐ = {res['Ea']:.2f} ± {res['dEa']:.2f} kcal/mol"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.3))

    ax.set_xlabel('1000/T (K⁻¹)', fontsize=13, fontweight='bold')
    ax.set_ylabel('ln(k₂)', fontsize=13, fontweight='bold')
    ax.set_title(f'{compound}', fontsize=14, fontweight='bold')
    set_figure_style(ax)

    if highlight and chi2 > 2:
        for i, (x, y, r) in enumerate(zip(inv_T, ln_k2, residuals)):
            y_fit = ln_k2_fit[i]
            ax.plot([x, x], [y, y_fit], color=COLORS['accent'], linewidth=2, alpha=0.5, zorder=1)

# ============================================================================
# FIGURE 6: ERROR BAR COMPARISON
# ============================================================================

def make_figure6():
    compounds_reordered = ['PCE', 'TCE', '1,1-DCE', 'cis-DCE', 'trans-DCE']

    x = np.arange(len(compounds_reordered))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    dEa_orig = [arrhenius_results[c]['dEa_orig'] for c in compounds_reordered]
    dEa_new  = [arrhenius_results[c]['dEa']      for c in compounds_reordered]

    ax.bar(x - width/2, dEa_orig, width, label='Huang et al.',
           color=COLORS['accent'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, dEa_new,  width, label='This work',
           color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Uncertainty in Ea (kcal mol⁻¹)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Compound', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(compounds_reordered, fontsize=12)
    ax.tick_params(axis='y', labelsize=12, width=1.5, length=6)
    ax.legend(frameon=False, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('Error_comparison.png', dpi=400, bbox_inches='tight')
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    make_figure1()
    make_figure2()
    make_figure6()

    plt.show()