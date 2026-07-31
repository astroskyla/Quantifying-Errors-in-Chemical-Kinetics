import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Circle
import numpy as np
import os

# ── Register Nimbus Sans (Helvetica clone) if available ──────────────
for pat in [
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",  # Linux
    "/Library/Fonts/Helvetica.ttc",                                  # macOS
    "C:/Windows/Fonts/helvetica.ttf",                                # Windows
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

# ── Canvas ────────────────────────────────────────────────────────────
W, H = 5.4, 6.8
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W);  ax.set_ylim(0, H)
ax.set_aspect('equal');  ax.axis('off')
fig.patch.set_facecolor('white')

R   = 0.38   # node radius
FS  = 7.5    # plate label font size

# ── Helpers ───────────────────────────────────────────────────────────
def node(x, y, label, observed=False, r=R, fs=9):
    fc = '#c8c8c8' if observed else 'white'
    ax.add_patch(Circle((x, y), r, ec='black', fc=fc,
                        lw=0.9, zorder=5, clip_on=False))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fs, zorder=6, clip_on=False)

def arrow(x1, y1, x2, y2, r=R):
    dx, dy = x2-x1, y2-y1
    d = np.hypot(dx, dy); ux, uy = dx/d, dy/d
    ax.annotate('', xy=(x2-ux*r, y2-uy*r), xytext=(x1+ux*r, y1+uy*r),
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=0.9, mutation_scale=10), zorder=4)

def plate(x, y, w, h, label, fs=FS):
    ax.add_patch(Rectangle((x, y), w, h, lw=0.8, ec='black',
                            fc='none', zorder=1))
    ax.text(x+w-0.10, y+0.10, label, ha='right', va='bottom',
            fontsize=fs, zorder=2)

# ── Node positions ────────────────────────────────────────────────────
lnA = (1.05, 5.8)
Ea  = (2.70, 5.8)
sig = (4.35, 5.8)
T   = (1.05, 4.10)
lnk = (2.70, 4.10)
obs = (2.70, 2.30)

# ── Plates (fixed, generous padding around every node) ────────────────
#
#  inner  (replicates):  obs lives here, label row at bottom inside box
#  middle (temperatures): T, lnk + inner plate; label row below inner box
#  outer  (compounds):  top-row nodes + middle plate; label row at bottom

# inner plate — obs at (2.70, 2.30); give 0.55 clearance each side,
#               0.60 below for the label row
plate(1.35, 1.45, 2.70, 1.50,
      r'replicates  $i = 1,\ldots,N_{t,j}$')

# middle plate — T at (1.05,4.10), top of inner plate at 2.95
#                give 0.55 left of T, 0.55 right of sig, 0.65 below inner
plate(0.38, 0.72, 4.64, 3.90,
      r'temperatures  $t = 1,\ldots,T_j$')

# outer plate — top of lnA/Ea/sig nodes at 6.18; bottom at 0.38
plate(0.10, 0.38, 5.20, 5.92,
      r'compounds  $j = 1,\ldots,J$')

# ── Arrows ────────────────────────────────────────────────────────────
arrow(*lnA, *lnk)
arrow(*Ea,  *lnk)
arrow(*T,   *lnk)
arrow(*lnk, *obs)
arrow(*sig, *obs)

# ── Nodes ─────────────────────────────────────────────────────────────
node(*lnA, r'$A_j$',                       fs=8.5)
node(*Ea,  r'$E_{a,j}$',                         fs=9)
node(*sig, r'$\sigma_{j}$',                  fs=8)
node(*T,   r'$T_{t,j}$',    observed=True,       fs=9)
node(*lnk, r'$k_{t,j}$',                    fs=8)
node(*obs, r'$k_{\mathrm{obs},i,t,j}$',    observed=True, fs=7)

plt.savefig('pgm-main.png', dpi=300, bbox_inches='tight', facecolor='white')
