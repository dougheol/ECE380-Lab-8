"""
Lab 8: Transformer B-H Hysteresis Plots
========================================
Reads oscilloscope CSV data and produces:
  1) Major B-H hysteresis loop (full amplitude, showing saturation)
  2) Minor B-H hysteresis loop (scaled approximation)
  3) Combined plot with both loops
  4) Raw waveforms for reference

Data columns (Tektronix 3-block CSV):
  Block 1: TIME, CH1 (current sense, 10 V/A), CH2 (secondary voltage)
  Block 2: TIME, MATH<Intg(CH2)> (oscilloscope integral of secondary)
  Block 3: TIME, REF1 (reference copy of CH1)

Physical conversions:
  I       = CH1 / 10  (A)
  Phi     = Intg(CH2) / N2  (Wb)
  B       = Phi / A  (T)
  H       = N1 * I / l  (A/m)

Core parameters (N87 MnZn ferrite toroid):
  N1 = 29 turns  (primary from Lab 7)
  N2 = 10 turns  (secondary)
  le = 36.75 mm  (mean magnetic path length)
  Ae = 15.78 mm2 (cross-sectional area)
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
time1, ch1, ch2 = [], [], []
time2, intg_raw = [], []

with open('tek0018(in).csv', 'r') as f:
    reader = csv.reader(f)
    for _ in range(21):
        next(reader)
    for row in reader:
        time1.append(float(row[0]))
        ch1.append(float(row[1]))
        ch2.append(float(row[2]))
        time2.append(float(row[4]))
        intg_raw.append(float(row[5]))

t = np.array(time1)
ch1 = np.array(ch1)
ch2 = np.array(ch2)
intg_raw = np.array(intg_raw)

# ---------------------------------------------------------------------------
# 2. Clean signals
# ---------------------------------------------------------------------------
# Remove DC offsets
ch1_centered = ch1 - np.mean(ch1)

# Detrend the oscilloscope integral (linear drift from DC offset in CH2)
trend = np.polyfit(t, intg_raw, 1)
intg_detrended = intg_raw - (trend[0] * t + trend[1])
intg_centered = intg_detrended - np.mean(intg_detrended)

# ---------------------------------------------------------------------------
# 3. Physical quantities
# ---------------------------------------------------------------------------
N1 = 29
N2 = 10
l = 0.03675        # mean magnetic path length (m)
A = 15.78e-6       # cross-sectional area (m2)

I = ch1_centered / 10.0
Phi = intg_centered / N2
B = Phi / A
H = N1 * I / l

# ---------------------------------------------------------------------------
# 4. Smooth for clean loop plots
# ---------------------------------------------------------------------------
window = 301
polyorder = 3
B_smooth = savgol_filter(B, window, polyorder)
H_smooth = savgol_filter(H, window, polyorder)

# ---------------------------------------------------------------------------
# 5. Extract one full cycle for the major loop
# ---------------------------------------------------------------------------
h_sign = np.sign(H_smooth)
zero_cross = np.where(np.diff(h_sign) > 0)[0]

if len(zero_cross) >= 2:
    cycle_start = zero_cross[0]
    cycle_end = zero_cross[1]
else:
    cycle_start = len(H_smooth) // 4
    cycle_end = 3 * len(H_smooth) // 4

H_major = H_smooth[cycle_start:cycle_end]
B_major = B_smooth[cycle_start:cycle_end]

# ---------------------------------------------------------------------------
# 6. Construct minor loop (scaled approximation)
# ---------------------------------------------------------------------------
minor_scale = 0.5
H_minor = H_major * minor_scale
B_minor = B_major * minor_scale

# ---------------------------------------------------------------------------
# 7. Estimate Bs, Br, Hc from the major loop
# ---------------------------------------------------------------------------
dB_dH = np.abs(np.gradient(B_major, H_major))
linear_slope = np.max(dB_dH)
saturation_mask = dB_dH < 0.1 * linear_slope
if np.any(saturation_mask):
    Bs = np.max(np.abs(B_major[saturation_mask]))
else:
    Bs = np.max(np.abs(B_major))

h_zero_crossings = np.where(np.diff(np.sign(H_major)))[0]
Br_values = []
for idx in h_zero_crossings:
    if 0 < idx < len(B_major):
        Br_values.append(abs(B_major[idx]))
Br = np.mean(Br_values) if Br_values else 0.0

b_zero_crossings = np.where(np.diff(np.sign(B_major)))[0]
Hc_values = []
for idx in b_zero_crossings:
    if 0 < idx < len(H_major):
        Hc_values.append(abs(H_major[idx]))
Hc = np.mean(Hc_values) if Hc_values else 0.0

print("=== B-H Loop Parameters ===")
print("H range:  %.1f to %.1f A/m" % (np.min(H_major), np.max(H_major)))
print("B range:  %.4f to %.4f T" % (np.min(B_major), np.max(B_major)))
print("Bs (est): %.4f T" % Bs)
print("Br (est): %.4f T" % Br)
print("Hc (est): %.1f A/m" % Hc)
print()

# ---------------------------------------------------------------------------
# 8. Plot 1: Major B-H hysteresis loop
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(H_smooth, B_smooth, 'lightblue', alpha=0.3, linewidth=0.5,
         label='raw data')
ax1.plot(H_major, B_major, 'b-', linewidth=2, label='major loop')

ax1.axhline(y=+Bs, color='r', linestyle='--', alpha=0.5,
            label='Bs = %.4f T' % Bs)
ax1.axhline(y=-Bs, color='r', linestyle='--', alpha=0.5)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

ax1.set_xlabel('H (A/m)', fontsize=14)
ax1.set_ylabel('B (T)', fontsize=14)
ax1.set_title('Major B-H Hysteresis Loop', fontsize=16)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)

textstr = ('Bs = %.4f T\nBr = %.4f T\nHc = %.1f A/m' % (Bs, Br, Hc))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('bh_major_loop.png', dpi=150, bbox_inches='tight')
print("Saved: bh_major_loop.png")
plt.close(fig1)

# ---------------------------------------------------------------------------
# 9. Plot 2: Minor B-H hysteresis loop
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.plot(H_minor, B_minor, 'g-', linewidth=2,
         label='minor loop (scaled, factor=%.1f)' % minor_scale)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

ax2.set_xlabel('H (A/m)', fontsize=14)
ax2.set_ylabel('B (T)', fontsize=14)
ax2.set_title('Minor B-H Hysteresis Loop', fontsize=16)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

Hm_minor = np.max(np.abs(H_minor))
Bm_minor = np.max(np.abs(B_minor))
textstr2 = ('H_max = %.1f A/m\nB_max = %.4f T' % (Hm_minor, Bm_minor))
props2 = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=props2)

plt.tight_layout()
plt.savefig('bh_minor_loop.png', dpi=150, bbox_inches='tight')
print("Saved: bh_minor_loop.png")
plt.close(fig2)

# ---------------------------------------------------------------------------
# 10. Plot 3: Combined major + minor loops
# ---------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(8, 6))

ax3.plot(H_major, B_major, 'b-', linewidth=2, label='major loop')
ax3.plot(H_minor, B_minor, 'g-', linewidth=2, label='minor loop (scaled)')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)

ax3.set_xlabel('H (A/m)', fontsize=14)
ax3.set_ylabel('B (T)', fontsize=14)
ax3.set_title('Major and Minor Hysteresis Loops', fontsize=16)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bh_combined_loops.png', dpi=150, bbox_inches='tight')
print("Saved: bh_combined_loops.png")
plt.close(fig3)

# ---------------------------------------------------------------------------
# 11. Plot 4: Raw waveforms (for reference)
# ---------------------------------------------------------------------------
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t_ms = t * 1000

ax4a.plot(t_ms, ch1, 'b-', linewidth=0.5)
ax4a.set_ylabel('CH1 (V)\n10 V/A scale', fontsize=12)
ax4a.set_title('Oscilloscope Waveforms', fontsize=14)
ax4a.grid(True, alpha=0.3)

ax4b.plot(t_ms, intg_centered * 1e6, 'r-', linewidth=0.5)
ax4b.set_xlabel('Time (ms)', fontsize=12)
ax4b.set_ylabel('Intg(CH2) (uVs)', fontsize=12)
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('raw_waveforms.png', dpi=150, bbox_inches='tight')
print("Saved: raw_waveforms.png")
plt.close(fig4)

print("\nDone. All plots saved.")
