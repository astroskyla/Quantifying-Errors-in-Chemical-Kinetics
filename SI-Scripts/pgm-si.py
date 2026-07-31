import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Circle
import numpy as np
import os

for pat in [
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/helvetica.ttf",
]:
    if os.path.exists(pat):
        try:
            fm.fontManager.addfont(pat)
        except Exception:
            pass

plt.rcParams['text.usetex']      = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Nimbus Sans', 'Helvetica', 'Arial',
                                     'Liberation Sans', 'DejaVu Sans']

# Canvas matches figure aspect: 780 wide x 820 tall → 7.8 x 8.2 inches
# Pixel-to-data: x_data = x_px/100,  y_data = (820 - y_px)/100
W, H = 7.8, 8.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

R  = 0.40
FS = 8.0

def px(x_px, y_px):
    """Pixel (top-left origin) → data coords."""
    return x_px / 100.0, (820 - y_px) / 100.0

def node(x, y, label, observed=False, r=R, fs=9.5):
    fc = '#c8c8c8' if observed else 'white'
    ax.add_patch(Circle((x, y), r, ec='black', fc=fc,
                        lw=0.9, zorder=5, clip_on=False))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fs, zorder=6, clip_on=False)

def arrow(x1, y1, x2, y2, r=R):
    dx, dy = x2 - x1, y2 - y1
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    ax.annotate(
        '', xy=(x2 - ux*r, y2 - uy*r),
        xytext=(x1 + ux*r, y1 + uy*r),
        arrowprops=dict(arrowstyle='->', color='black',
                        lw=0.9, mutation_scale=10),
        zorder=4,
    )

def plate(x, y, w, h, label, fs=FS):
    ax.add_patch(Rectangle((x, y), w, h, lw=0.8, ec='black',
                            fc='none', zorder=1))
    ax.text(x + w - 0.10, y + 0.10, label,
            ha='right', va='bottom', fontsize=fs, zorder=2)

# Top-row globals  (y_px ≈ 90)
lnk = px(150,  90)   # ln(k)
m   = px(270,  90)   # m
n   = px(400,  90)   # n
sig = px(605,  90)   # σ_ln

# Per-condition latents (inside outer plate)
A   = px(175, 310)   # ln[A]_i   — left side, upper
B   = px(285, 420)   # ln[B]_i   — slightly right & below A
rtr = px(445, 310)   # ln(r_true,i)

# Per-replicate observed (inside inner plate)
obs = px(415, 560)   # ln(r_obs,i,j)

# ── Plates ─────────────────────────────────────────────────────────────
# Outer plate: pixel corners (105,190)→(705,720)
ox,  oy  = px(105, 720)        # bottom-left in data coords
ow = (705 - 105) / 100.0       # width  in data units
oh = (720 - 190) / 100.0       # height in data units
plate(ox, oy, ow, oh, r'conditions  $i = 1,\ldots,I$')

# Inner plate: pixel corners (295,480)→(685,655)
ix,  iy  = px(295, 655)
iw = (685 - 295) / 100.0
ih = (655 - 480) / 100.0
plate(ix, iy, iw, ih, r'replicates  $j = 1,\ldots,N_i$')

# ── Arrows ─────────────────────────────────────────────────────────────
arrow(*lnk, *rtr)   # ln(k)   → r_true
arrow(*m,   *rtr)   # m       → r_true
arrow(*n,   *rtr)   # n       → r_true
arrow(*A,   *rtr)   # ln[A]_i → r_true
arrow(*B,   *rtr)   # ln[B]_i → r_true
arrow(*rtr, *obs)   # r_true  → r_obs
arrow(*sig, *obs)   # σ_ln    → r_obs  (long diagonal from top-right)

# ── Nodes ──────────────────────────────────────────────────────────────
node(*lnk, r'$k$',                                    fs=9.5)
node(*m,   r'$m$',                                         fs=9.5)
node(*n,   r'$n$',                                         fs=9.5)
node(*sig, r'$\sigma$',                              fs=8.5)
node(*A,   r'$[A]_i$',      observed=True,             fs=8.5)
node(*B,   r'$[B]_i$',      observed=True,             fs=8.5)
node(*rtr, r'$r_{\mathrm{true},i}$',                 fs=7.8)
node(*obs, r'$r_{\mathrm{obs},i,j}$', observed=True, fs=7.8)

plt.savefig('pgm-si.png', dpi=300, bbox_inches='tight', facecolor='white')