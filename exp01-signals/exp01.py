"""Experiment 1: time and frequency domain, and the pendulum."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from student import SEED, start_run, stamp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
start_run("exp01-signals")
out = {}


def save(fig, name):
    stamp(fig)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=150)
    plt.close(fig)


def amp_spectrum(x, fs):
    """One-sided amplitude spectrum in the units of x."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=1 / fs)
    a = np.abs(X) * 2.0 / x.size
    a[0] /= 2.0
    return f, a


# ---- Part 1: build a signal and read its spectrum --------------------------
fs, T = 500.0, 2.0                      # sampling rate (Hz), record length (s)
t = np.arange(0, T, 1 / fs)
N = t.size
rng = np.random.default_rng(SEED)       # personal seed -> my own noise
comps = [(5, 1.00), (20, 0.50), (50, 0.25)]
x = sum(A * np.sin(2 * np.pi * f0 * t) for f0, A in comps) + 0.20 * rng.standard_normal(N)
f, amp = amp_spectrum(x, fs)
out["sampling"] = dict(fs=fs, T=T, N=N, df=fs / N, nyquist=fs / 2)
meas = []
for f0, A in comps:
    band = (f > f0 - 2) & (f < f0 + 2)
    k = np.argmax(np.where(band, amp, 0))
    meas.append(dict(set_f=f0, meas_f=float(f[k]), set_a=A, meas_a=float(amp[k])))
out["peaks"] = meas
print("N =", N, "df =", fs / N, "Hz, Nyquist =", fs / 2, "Hz")
for m in meas:
    print(f"  set {m['set_f']} Hz / {m['set_a']:.2f}  ->  measured {m['meas_f']:.1f} Hz / {m['meas_a']:.3f}")

fig, ax = plt.subplots(figsize=(8, 3))
m = t < 0.5
ax.plot(t[m], x[m], lw=0.8, color="0.15")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (arb. units)")
ax.set_title("Constructed signal (5 + 20 + 50 Hz + noise), first 0.5 s")
ax.grid(alpha=0.3)
save(fig, "fig1_1_time.png")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(f, amp, lw=0.8, color="0.15")
for mm in meas:
    ax.annotate(f"{mm['meas_f']:.1f} Hz\n{mm['meas_a']:.2f}", (mm["meas_f"], mm["meas_a"]),
                textcoords="offset points", xytext=(8, -4), fontsize=8)
ax.set_xlim(0, 100)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude (arb. units)")
ax.set_title("One-sided amplitude spectrum (0-100 Hz shown)")
ax.grid(alpha=0.3)
save(fig, "fig1_2_spectrum.png")

# spectral leakage: same signal, T = 1.7 s (peaks fall between bins)
tl = np.arange(0, 1.7, 1 / fs)
xl = sum(A * np.sin(2 * np.pi * f0 * tl) for f0, A in comps)
fl, al = amp_spectrum(xl, fs)
leak = []
for f0, A in comps:
    band = (fl > f0 - 2) & (fl < f0 + 2)
    k = np.argmax(np.where(band, al, 0))
    leak.append(dict(set_f=f0, meas_f=float(fl[k]), meas_a=float(al[k])))
out["leakage_T1_7"] = dict(df=fs / tl.size, peaks=leak)
print("Leakage (T=1.7 s, df=%.3f Hz):" % (fs / tl.size),
      [(round(l_['meas_f'], 2), round(l_['meas_a'], 3)) for l_ in leak])

# ---- Part 2: aliasing --------------------------------------------------------
fs_a = 8.0
t_a = np.arange(0, 2.0, 1 / fs_a)
x_a = np.sin(2 * np.pi * 5 * t_a)
f_a, a_a = amp_spectrum(x_a, fs_a)
alias = float(f_a[np.argmax(a_a)])
out["alias_peak_hz"] = alias
print("5 Hz sampled at 8 Hz -> peak at", alias, "Hz")

tf = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(tf, np.sin(2 * np.pi * 5 * tf), "k--", lw=0.8, label="true 5 Hz sine")
ax.plot(tf, -np.sin(2 * np.pi * 3 * tf), "r-", lw=0.9, label=f"apparent {alias:.0f} Hz alias")
m = t_a <= 1
ax.plot(t_a[m], x_a[m], "bo", ms=5, label="samples at 8 Hz")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (arb. units)")
ax.set_title("Aliasing: 5 Hz sine sampled at 8 Hz")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
save(fig, "fig1_3_aliasing.png")

# ---- Part 3: pendulum --------------------------------------------------------
g, L = 9.81, 1.0
T0 = 2 * np.pi * np.sqrt(L / g)
out["T0"] = T0


def pendulum(t, y, b):
    theta, omega = y
    return [omega, -b * omega - (g / L) * np.sin(theta)]


def simulate(theta0_deg, b, T=20.0, n=4000):
    te = np.linspace(0, T, n)
    sol = solve_ivp(pendulum, [0, T], [np.deg2rad(theta0_deg), 0.0], args=(b,),
                    t_eval=te, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1]


def period_zero_cross(tt, th):
    """Mean spacing of upward zero crossings (linear interpolation)."""
    s = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    tc = tt[s] - th[s] * (tt[s + 1] - tt[s]) / (th[s + 1] - th[s])
    return float(np.mean(np.diff(tc)))


def period_spectrum(tt, th):
    f_, a_ = amp_spectrum(th - th.mean(), 1 / (tt[1] - tt[0]))
    return float(1 / f_[1:][np.argmax(a_[1:])])


rows = []
for a in (5, 45, 95, 135, 175):
    tt, th, _ = simulate(a, 0.0, T=60.0, n=24000)
    pz, ps = period_zero_cross(tt, th), period_spectrum(tt, th)
    rows.append(dict(angle=a, zc=pz, spec=ps, ratio=pz / T0))
    print(f"{a:>4} deg: T_zc = {pz:.3f} s, T_spec = {ps:.3f} s, ratio = {pz / T0:.2f}")
out["periods"] = rows

dampings = (0.0, 0.3, 2.0)
out["dampings"] = dampings
fig, ax = plt.subplots(figsize=(8, 3))
for b, lab in zip(dampings, ("undamped", "light", "heavy")):
    tt, th, _ = simulate(40, b)
    ax.plot(tt, np.rad2deg(th), lw=0.9, label=f"{lab}, b = {b} s$^{{-1}}$")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle θ (deg)")
ax.set_title("Pendulum released from 40°: three damping values")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
save(fig, "fig1_4_damping.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for b in (0.0, 0.3):
    _, th, om = simulate(40, b)
    axes[0].plot(np.rad2deg(th), om, lw=0.9, label=f"b = {b} s$^{{-1}}$")
axes[0].set_xlabel("Angle θ (deg)"); axes[0].set_ylabel("Angular velocity ω (rad/s)")
axes[0].set_title("Phase portrait (θ0 = 40°)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
amps = [5, 20, 45, 70, 95, 120, 135, 150, 165, 175]
per = [period_zero_cross(*simulate(a, 0.0, T=60.0, n=24000)[:2]) for a in amps]
axes[1].plot(amps, per, "o-", ms=4, label="measured (zero crossings)")
axes[1].axhline(T0, color="r", ls="--", label=f"small-angle T0 = {T0:.3f} s")
axes[1].set_xlabel("Initial amplitude θ0 (deg)"); axes[1].set_ylabel("Period (s)")
axes[1].set_title("Period against initial amplitude"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
save(fig, "fig1_5_phase_period.png")

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("T0 =", round(T0, 4), "s; results saved to results.json")
