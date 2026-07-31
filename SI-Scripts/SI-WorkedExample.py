"""
Worked Example: Error Reporting for A + B → C
==============================================
Reaction: A + B → C,   rate = k[A]^m[B]^n
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

np.random.seed(42)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
rcParams['font.size'] = 13
rcParams['axes.linewidth'] = 1.4
rcParams['xtick.major.width'] = 1.4
rcParams['ytick.major.width'] = 1.4
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['axes.labelsize'] = 13
rcParams['legend.fontsize'] = 11

COLORS = {
    'primary':   '#2A9D8F',
    'secondary': '#264653',
    'accent':    '#E76F51',
    'warning':   '#F4A261',
    'light':     '#B8DDD6',
    'original':  '#8AB6D6',
}

def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.4)
    ax.spines['bottom'].set_linewidth(1.4)
    ax.tick_params(direction='out', length=5, width=1.4)

SEP = "=" * 65

# ===========================================================================
# STAGE 0: INSTRUMENT ERROR SPECIFICATION
# ===========================================================================

m_A_nom, m_B_nom = 15.0, 20.0
delta_bal = 0.3
V_P1, V_P2, V_P3 = 500.0, 500.0, 4000.0
V_tot   = V_P1 + V_P2 + V_P3
pip_tol = 0.01
sig_NMR = 0.02

bal_A   = delta_bal / m_A_nom
bal_B   = delta_bal / m_B_nom
pip_rxn = pip_tol
pip_buf = (V_P3 / V_tot) * pip_tol

sig_rel_A = np.sqrt(bal_A**2 + pip_rxn**2 + pip_buf**2)
sig_rel_B = np.sqrt(bal_B**2 + pip_rxn**2 + pip_buf**2)
max_rel_A = bal_A + pip_rxn + pip_buf
max_rel_B = bal_B + pip_rxn + pip_buf

print(f"""
  σ_[A] (quadrature): {sig_rel_A*100:.3f}%    δ_[A] (worst-case): {max_rel_A*100:.3f}%
  σ_[B] (quadrature): {sig_rel_B*100:.3f}%    δ_[B] (worst-case): {max_rel_B*100:.3f}%
  σ_NMR: {sig_NMR*100:.1f}%
""")

# ===========================================================================
# STAGE 1: SYNTHETIC DATA
# ===========================================================================
K_TRUE, M_TRUE, N_TRUE = 0.050, 1.0, 1.0
SIG_LN = 0.6
N_REP  = 3
N_COND = 5

A_VAR   = np.array([1.0, 2.0,  4.0,  8.0, 16.0])
B_FIXED = 5.0
B_VAR   = np.array([1.0, 2.0,  4.0,  8.0, 16.0])
A_FIXED = 5.0
A_K     = np.array([2.0, 4.0,  6.0,  8.0, 10.0])
B_K     = np.array([2.0, 4.0,  6.0,  8.0, 10.0])

def true_rate(A, B):
    return K_TRUE * np.asarray(A)**M_TRUE * np.asarray(B)**N_TRUE

def simulate(A, B):
    mu    = true_rate(np.atleast_1d(A), np.atleast_1d(B))
    noise = np.exp(np.random.normal(0, SIG_LN, (len(mu), N_REP)))
    return mu[:, None] * noise

R1 = simulate(A_VAR, np.full(N_COND, B_FIXED))
R2 = simulate(np.full(N_COND, A_FIXED), B_VAR)
R3 = simulate(A_K, B_K)

# ===========================================================================
# STAGE 2: RATE STATISTICS
# ===========================================================================
def rate_stats(R):
    ln_R    = np.log(R)
    mean_r  = R.mean(axis=1)
    sd_r    = R.std(axis=1, ddof=1)
    mean_ln = ln_R.mean(axis=1)
    sd_ln   = ln_R.std(axis=1, ddof=1)
    cv      = sd_r / mean_r
    geom    = np.exp(mean_ln)
    # Floor range by ±2% NMR tolerance
    r_min   = np.minimum(R.min(axis=1), mean_r * (1 - sig_NMR))
    r_max   = np.maximum(R.max(axis=1), mean_r * (1 + sig_NMR))
    return dict(mean=mean_r, sd=sd_r, mean_ln=mean_ln, sd_ln=sd_ln,
                cv=cv, geom=geom, r_min=r_min, r_max=r_max)

S1, S2, S3 = rate_stats(R1), rate_stats(R2), rate_stats(R3)

# ===========================================================================
# STAGE 3: ORDER DETERMINATION
# ===========================================================================
def rep_slopes(x, R):
    return np.array([np.polyfit(np.log(x), np.log(R[:, j]), 1)[0]
                     for j in range(R.shape[1])])

def wls_loglog(x, y_mean, sd_ln):
    lx, ly = np.log(x), np.log(y_mean)
    def _wls(w):
        S   = w.sum();         Sx  = (w*lx).sum()
        Sxx = (w*lx**2).sum(); Sy  = (w*ly).sum()
        Sxy = (w*lx*ly).sum(); D   = S*Sxx - Sx**2
        m   = (S*Sxy - Sx*Sy) / D
        b   = (Sy - m*Sx) / S
        res = ly - (m*lx + b)
        c2  = (w * res**2).sum() / (len(x) - 2)
        se  = np.sqrt(max(c2, 1.0) * S / D)
        return m, b, se, c2
    m, b, se, c2_rep = _wls(N_REP / sd_ln**2)
    _, _, se_nmr, c2_nmr = _wls(np.full(len(x), 1./sig_NMR**2))
    return m, b, se, c2_rep, c2_nmr, se_nmr

slopes_A = rep_slopes(A_VAR, R1)
m_fit, int_A, se_m, chi2_m, chi2_m_nmr, se_m_nmr = wls_loglog(
    A_VAR, S1['mean'], S1['sd_ln'])
m_min, m_max = slopes_A.min(), slopes_A.max()

slopes_B = rep_slopes(B_VAR, R2)
n_fit, int_B, se_n, chi2_n, chi2_n_nmr, se_n_nmr = wls_loglog(
    B_VAR, S2['mean'], S2['sd_ln'])
n_min, n_max = slopes_B.min(), slopes_B.max()

print(f"""
  ORDER IN A:  m = {m_fit:.3f} ± {se_m:.3f}   χ²_red = {chi2_m:.2f}   range [{m_min:.2f}, {m_max:.2f}]
  ORDER IN B:  n = {n_fit:.3f} ± {se_n:.3f}   χ²_red = {chi2_n:.2f}   range [{n_min:.2f}, {n_max:.2f}]
χ²_red (NMR σ):  order in A = {chi2_m_nmr:.2f}   order in B = {chi2_n_nmr:.2f}
""")

# ===========================================================================
# STAGE 4: RATE CONSTANT k
# ===========================================================================
k_all    = np.array([R3[i, j] / (A_K[i]**m_fit * B_K[i]**n_fit)
                     for i in range(N_COND) for j in range(N_REP)])
ln_k     = np.log(k_all)
mu_ln    = ln_k.mean()
sig_ln_k = ln_k.std(ddof=1)

# Statistical track: geometric mean ± σ_ln back-transformed
k_geom = np.exp(mu_ln)
k_hi   = k_geom * (np.exp( sig_ln_k) - 1)   # δ₊ = k_geom(exp(+σ_ln) − 1)
k_lo   = k_geom * (1 - np.exp(-sig_ln_k))    # δ₋ = k_geom(1 − exp(−σ_ln))

# Range track: 32-combination exhaustive grid
r_lo = S3['r_min'];            r_hi = S3['r_max']
A_lo = A_K * (1 - max_rel_A);  A_hi = A_K * (1 + max_rel_A)
B_lo = B_K * (1 - max_rel_B);  B_hi = B_K * (1 + max_rel_B)

k_extreme = [
    r / (a**m * b**n)
    for i in range(N_COND)
    for r in [r_lo[i], r_hi[i]]
    for a in [A_lo[i],  A_hi[i]]
    for b in [B_lo[i],  B_hi[i]]
    for m in [m_min, m_max]
    for n in [n_min, n_max]
]
k_rng_lo, k_rng_hi = min(k_extreme), max(k_extreme)

print(f"""
  k STATISTICAL:  {k_geom:.5f} +{k_hi:.5f}/-{k_lo:.5f}  mM⁻¹ min⁻¹  (σ_ln = {sig_ln_k:.3f})
  k RANGE:        [{k_rng_lo:.5f}, {k_rng_hi:.5f}]  mM⁻¹ min⁻¹
  True k = {K_TRUE:.5f}  ✓
""")

# ===========================================================================
# FIGURE 1: CV CONSISTENCY CHECK
# ===========================================================================
fig1, axes1 = plt.subplots(3, 1, figsize=(8, 12))
fig1.subplots_adjust(hspace=0.45)

series_data   = [S1, S2, S3]
series_labels = ['series 1 (vary [A])', 'series 2 (vary [B])', 'series 3 (mixed)']
series_colors = ['#E76F51', '#F4A261', '#2A9D8F']

bar_width = 0.26
x       = np.arange(N_COND)
offsets = [-bar_width, 0, bar_width]

metrics = [
    ('sd',    'absolute σ (mM min⁻¹)', '(A)  absolute σ'),
    ('cv',    'CV (%)',                  '(B)  coefficient of variation'),
    ('sd_ln', r'$\sigma_{\ln}$',       '(C)  log-normal σ'),
]

for ax, (key, ylabel, title) in zip(axes1, metrics):
    style(ax)
    for S, label, col, offset in zip(series_data, series_labels, series_colors, offsets):
        vals = S[key] * 100 if key == 'cv' else S[key]
        ax.bar(x + offset, vals, width=bar_width,
               color=col, alpha=0.85, edgecolor='black', label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(N_COND)], fontsize=12)
    ax.set_xlabel('condition index', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=15, pad=8)

handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='lower center', ncol=3,
            fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.025))

plt.tight_layout()
plt.savefig('cv_consistency.png', bbox_inches='tight', dpi=400)
plt.show()

# ===========================================================================
# FIGURE 3: ORDER IN A | ORDER IN B | RATES
# ===========================================================================
REP_COLORS = ['#E76F51', '#F4A261', '#2A9D8F']
fig3, (ax_A, ax_B, ax_C) = plt.subplots(3, 1, figsize=(8, 12))
fig3.subplots_adjust(hspace=0.45)

xfit_ln = np.linspace(np.log(min(A_VAR)) - 0.3, np.log(max(A_VAR)) + 0.3, 200)

def _draw_order_panel(ax, x, R, slopes, m, intc, se, chi2, xlabel, title, panel_label):
    style(ax)
    lx     = np.log(x)
    idx_lo = slopes.argmin()
    idx_hi = slopes.argmax()
    _, c_lo = np.polyfit(lx, np.log(R[:, idx_lo]), 1)
    _, c_hi = np.polyfit(lx, np.log(R[:, idx_hi]), 1)

    # Range track: band between min and max replicate OLS slopes
    ax.fill_between(xfit_ln,
                    slopes[idx_lo] * xfit_ln + c_lo,
                    slopes[idx_hi] * xfit_ln + c_hi,
                    alpha=0.15, color='#264653', zorder=1,
                    label='min/max OLS')

    # Statistical track: ±1σ teal band (χ²_red-corrected SE)
    ax.fill_between(xfit_ln,
                    (m - se) * xfit_ln + intc,
                    (m + se) * xfit_ln + intc,
                    alpha=0.30, color='#6c9a8b', zorder=2,
                    label=f'WLS')

    ax.plot(xfit_ln, m * xfit_ln + intc, color='#264653', lw=2, zorder=3)

    for j in range(R.shape[1]):
        ax.scatter(lx, np.log(R[:, j]),
                   color=REP_COLORS[j], s=30, zorder=4,
                   alpha=0.95, label=f'rep {j+1}')

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('ln(rate / mM min⁻¹)', fontsize=14)
    ax.set_title(f'{panel_label}  {title}', fontsize=15, pad=8)
    ax.legend(fontsize=11, loc='upper left', frameon=False)

_draw_order_panel(ax_A, A_VAR, R1, slopes_A, m_fit, int_A, se_m, chi2_m,
                  'ln([A] / mM)', 'order in A', '(A)')
_draw_order_panel(ax_B, B_VAR, R2, slopes_B, n_fit, int_B, se_n, chi2_n,
                  'ln([B] / mM)', 'order in B', '(B)')

# --- Panel C: rates with both tracks ---
style(ax_C)
x_pos       = np.arange(N_COND)
cond_labels = [f'({int(a)},{int(b)})' for a, b in zip(A_K, B_K)]

# Outer: absolute range (grey bars with caps)
for i in range(N_COND):
    ax_C.plot([i, i], [S3['r_min'][i], S3['r_max'][i]],
              color='#264653', lw=1.8, alpha=0.55, zorder=2)
    for y in [S3['r_min'][i], S3['r_max'][i]]:
        ax_C.plot([i - 0.14, i + 0.14], [y, y],
                  color='#264653', lw=1.8, alpha=0.55)

# Inner: geometric mean ± σ_ln back-transformed asymmetrically
# δ₊ = r_geom × (exp(+σ_ln) − 1)
# δ₋ = r_geom × (1 − exp(−σ_ln))
hi_sig = S3['geom'] * (np.exp( S3['sd_ln']) - 1)
lo_sig = S3['geom'] * (1 - np.exp(-S3['sd_ln']))

ax_C.errorbar(x_pos, S3['geom'], yerr=[lo_sig, hi_sig],
              fmt='o', color='#6c9a8b', ms=8,
              capsize=6, capthick=2, lw=2, zorder=5,
              label=r'geometric mean ± $\sigma_{\ln}$')
ax_C.plot([], [], color='#264653', lw=1.8, alpha=0.55,
          label='absolute range')

ax_C.set_xticks(x_pos)
ax_C.set_xticklabels(cond_labels, fontsize=12)
ax_C.set_xlabel('([A],[B]) condition (mM)', fontsize=14)
ax_C.set_ylabel('rate (mM min⁻¹)', fontsize=14)
ax_C.legend(fontsize=11, loc='upper left', frameon=False)
ax_C.set_title('(C)  rates', fontsize=15, pad=8)

plt.tight_layout()
plt.savefig('orders_rates.png', bbox_inches='tight', dpi=400)
plt.show()