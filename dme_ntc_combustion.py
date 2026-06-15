"""
================================================================================
 Two-stage autoignition and Negative Temperature Coefficient (NTC) behaviour
 of dimethyl ether (DME, CH3OCH3) / air mixtures.
================================================================================
 Computational Methods in Combustion (MKWS) project.

 This single script reproduces all results and figures in the report:
   1. Ignition-delay-time (IDT) S-curves showing the NTC region for
      p = 10, 20, 40 bar (stoichiometric).
   2. Effect of equivalence ratio (phi = 0.5, 1.0, 2.0) at 20 bar.
   3. Time-resolved two-stage (cool-flame) ignition history.
   4. Species evolution revealing the low-temperature RO2 chemistry.
   5. First-stage (cool-flame) vs total ignition delay.

 Model: 0-D adiabatic constant-pressure homogeneous reactor (Cantera).
 Kinetics: NUI Galway 2015 hierarchical mechanism shipped with Cantera as
 example data ('example_data/n-hexane-NUIG-2015.yaml', 1268 species,
 5336 reactions). It contains the complete low-temperature DME sub-mechanism
 (CH3OCH2O2 / CH3OCH2O2H / ketohydroperoxide pathway), so DME (CH3OCH3) is
 used directly as the fuel.

 Requirements:  cantera, numpy, scipy, pandas, matplotlib
 Run:           python dme_ntc_combustion.py
================================================================================
"""

import warnings
from multiprocessing import Pool

import numpy as np
import cantera as ct
from scipy.signal import find_peaks
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------- configuration --------------------------------
MECH = "example_data/n-hexane-NUIG-2015.yaml"   # ships with Cantera
FUEL = "CH3OCH3"                                  # dimethyl ether
OXID = "O2:1.0, N2:3.76"                          # air
TEMPS = np.linspace(625, 1300, 28)               # initial temperatures [K]
NPROC = 2                                         # parallel workers
SPEC = ["CH3OCH3", "O2", "CH2O", "CH3OCH2O2", "OH", "CO", "CO2"]

# ----------------------------- core kinetics --------------------------------
def ignition(T0, p_bar, phi, history=False):
    """Constant-pressure adiabatic reactor; return ignition metrics.

    The total ignition delay is the time of maximum dT/dt; the first-stage
    (cool-flame) delay is the first significant dT/dt peak before it.
    """
    gas = ct.Solution(MECH)
    gas.set_equivalence_ratio(phi, FUEL, OXID)
    gas.TP = T0, p_bar * 1e5
    reac = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reac])
    net.atol, net.rtol = 1e-15, 1e-6

    tmax = 20.0 if T0 < 700 else (8.0 if T0 < 800 else 2.0)
    idx = {s: gas.species_index(s) for s in SPEC}
    t_arr, T_arr = [], []
    hist = {s: [] for s in SPEC}

    t = 0.0
    while t < tmax:
        t = net.step()
        t_arr.append(t)
        T_arr.append(reac.T)
        if history:
            X = reac.thermo.X
            for s in SPEC:
                hist[s].append(X[idx[s]])
        if reac.T > T0 + 600 and len(T_arr) > 5:
            break

    t_arr, T_arr = np.asarray(t_arr), np.asarray(T_arr)
    dTdt = np.gradient(T_arr, t_arr)
    i_main = int(np.argmax(dTdt))
    idt_total = t_arr[i_main]

    idt_first = np.nan
    if i_main > 3:
        peaks, _ = find_peaks(dTdt[:i_main],
                              prominence=max(50.0, 0.02 * dTdt[i_main]))
        if len(peaks):
            idt_first = t_arr[peaks[0]]

    out = dict(T0=T0, p_bar=p_bar, phi=phi,
               idt_total=float(idt_total), idt_first=float(idt_first))
    if history:
        out["t"], out["T"] = t_arr, T_arr
        for s in SPEC:
            out["X_" + s] = np.asarray(hist[s])
    return out


# helpers for parallel sweeps -------------------------------------------------
def _worker(args):
    return ignition(*args)


def sweep(p_bar, phi):
    tasks = [(T, p_bar, phi, False) for T in TEMPS]
    with Pool(NPROC) as pool:
        rows = pool.map(_worker, tasks)
    df = pd.DataFrame(rows)
    df["invT"] = 1000.0 / df.T0
    df["idt_total_ms"] = df.idt_total * 1e3
    df["idt_first_ms"] = df.idt_first * 1e3
    return df.sort_values("T0").reset_index(drop=True)


# ----------------------------- main study -----------------------------------
def main():
    g = ct.Solution(MECH)
    print(f"Cantera {ct.__version__} | mechanism: "
          f"{g.n_species} species, {g.n_reactions} reactions")

    # (1) pressure effect, stoichiometric
    sweeps_p = {p: sweep(p, 1.0) for p in (10, 20, 40)}
    # (2) equivalence-ratio effect at 20 bar
    sweeps_phi = {0.5: sweep(20, 0.5), 1.0: sweeps_p[20], 2.0: sweep(20, 2.0)}
    # (3-4) time histories
    hist = {tag: ignition(T0, 20, 1.0, history=True)
            for tag, T0 in [("twostage", 700), ("ntc", 850), ("single", 1200)]}

    _plot_pressure(sweeps_p)
    _plot_phi(sweeps_phi)
    _plot_history(hist["ntc"])
    _plot_species(hist["ntc"])
    _plot_first_vs_total(sweeps_p[20])
    base, rows = sensitivity()
    _plot_sensitivity(base, rows)
    _plot_validation(sweeps_p)
    print("Figures written: fig1..fig7 (*.png)")


# ----------------------------- plotting -------------------------------------
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 150, "savefig.bbox": "tight",
                     "lines.linewidth": 2, "lines.markersize": 5})
_TOPAX = (lambda x: 1000.0 / np.where(x == 0, np.nan, x),
          lambda T: 1000.0 / np.where(T == 0, np.nan, T))


def _add_T_axis(ax):
    sec = ax.secondary_xaxis("top", functions=_TOPAX)
    sec.set_xlabel("Temperature  (K)")


def _plot_pressure(s):
    fig, ax = plt.subplots(figsize=(7, 5))
    for p, c, m in [(10, "#1f77b4", "o"), (20, "#d62728", "s"), (40, "#2ca02c", "^")]:
        d = s[p]
        ax.semilogy(d.invT, d.idt_total_ms, m + "-", color=c, label=f"{p} bar")
    ax.set_xlabel("1000 / T  (1/K)"); ax.set_ylabel("Ignition delay time  (ms)")
    ax.set_title("DME / air ignition delay - pressure effect (phi = 1.0)")
    ax.legend(title="Pressure"); _add_T_axis(ax)
    ax.annotate("NTC region", xy=(1.10, 1.0), xytext=(1.18, 4.0),
                arrowprops=dict(arrowstyle="->"))
    fig.savefig("fig1_pressure_ntc.png"); plt.close(fig)


def _plot_phi(s):
    fig, ax = plt.subplots(figsize=(7, 5))
    for phi, c, m in [(0.5, "#1f77b4", "o"), (1.0, "#d62728", "s"), (2.0, "#2ca02c", "^")]:
        d = s[phi]
        ax.semilogy(d.invT, d.idt_total_ms, m + "-", color=c, label=f"phi = {phi}")
    ax.set_xlabel("1000 / T  (1/K)"); ax.set_ylabel("Ignition delay time  (ms)")
    ax.set_title("DME / air ignition delay - equivalence-ratio effect (20 bar)")
    ax.legend(); _add_T_axis(ax)
    fig.savefig("fig2_equivalence_ratio.png"); plt.close(fig)


def _plot_history(h):
    t = h["t"] * 1e3
    dT = np.gradient(h["T"], h["t"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    a1.plot(t, h["T"], color="#d62728"); a1.set_ylabel("Temperature  (K)")
    a1.set_title("Two-stage autoignition of DME/air (T0 = 850 K, 20 bar, phi = 1.0)")
    im = int(np.argmax(dT)); pk, _ = find_peaks(dT[:im], prominence=max(50, 0.02 * dT[im]))
    if len(pk):
        a1.axvline(t[pk[0]], ls="--", color="gray")
        a1.annotate("1st stage\n(cool flame)", xy=(t[pk[0]], h["T"][pk[0]]),
                    xytext=(t[pk[0]] + 0.03, 1050), arrowprops=dict(arrowstyle="->"))
    a1.axvline(t[im], ls="--", color="black")
    a1.annotate("2nd stage\n(main ignition)", xy=(t[im], h["T"][im]),
                xytext=(t[im] - 0.30, 1500), arrowprops=dict(arrowstyle="->"))
    a2.semilogy(t, dT, color="#1f77b4")
    a2.set_ylabel("dT/dt  (K/s)"); a2.set_xlabel("Time  (ms)"); a2.set_xlim(0, 1.0)
    fig.savefig("fig3_twostage_history.png"); plt.close(fig)


def _plot_species(h):
    t = h["t"] * 1e3
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, h["X_CH3OCH3"], label="CH3OCH3 (DME)", color="black")
    ax.plot(t, h["X_O2"], label="O2", color="#7f7f7f")
    ax.plot(t, h["X_CH2O"], label="CH2O", color="#1f77b4")
    ax.plot(t, h["X_CH3OCH2O2"] * 50, label="CH3OCH2O2 (x50)", color="#9467bd")
    ax.plot(t, h["X_OH"] * 20, label="OH (x20)", color="#d62728")
    ax.plot(t, h["X_CO"], label="CO", color="#2ca02c")
    ax.set_xlim(0, 1.0); ax.set_xlabel("Time  (ms)"); ax.set_ylabel("Mole fraction")
    ax.set_title("Species evolution - low-T (RO2) chemistry then main ignition")
    ax.legend(fontsize=9, ncol=2)
    fig.savefig("fig4_species.png"); plt.close(fig)


def _plot_first_vs_total(d):
    first = d.idt_first_ms.where(d.idt_first_ms < 0.8 * d.idt_total_ms)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(d.invT, d.idt_total_ms, "s-", color="#d62728", label="Total ignition delay")
    ax.semilogy(d.invT, first, "o--", color="#1f77b4", label="1st-stage (cool-flame) delay")
    ax.set_xlabel("1000 / T  (1/K)"); ax.set_ylabel("Delay time  (ms)")
    ax.set_title("First-stage vs total ignition delay (DME/air, 20 bar, phi = 1.0)")
    ax.legend(); _add_T_axis(ax)
    fig.savefig("fig5_first_vs_total.png"); plt.close(fig)


# ----------------------------- sensitivity ----------------------------------
DME_SPECIES = ["CH3OCH3", "CH3OCH2", "CH3OCH2O2", "CH3OCH2O2H", "CH2OCH2O2H",
               "O2CH2OCH2O2H", "HO2CH2OCHO", "CH3OCH2O"]
KEY_BRANCH = ["H2O2 (+M) <=> 2 OH (+M)", "2 HO2 <=> H2O2 + O2",
              "CH2O + HO2 <=> H2O2 + HCO"]


def _idt_simple(gas, T0, P_bar, phi, tmax=2.0):
    """Total ignition delay for an already-configured gas (multipliers kept)."""
    gas.set_equivalence_ratio(phi, FUEL, OXID)
    gas.TP = T0, P_bar * 1e5
    reac = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reac])
    net.atol, net.rtol = 1e-15, 1e-6
    t_arr, T_arr, t = [], [], 0.0
    while t < tmax:
        t = net.step()
        t_arr.append(t); T_arr.append(reac.T)
        if reac.T > T0 + 600 and len(T_arr) > 5:
            break
    t_arr, T_arr = np.asarray(t_arr), np.asarray(T_arr)
    return t_arr[int(np.argmax(np.gradient(T_arr, t_arr)))]


def sensitivity(T0=825.0, P_bar=20.0, phi=1.0, factor=2.0, top=12):
    """Brute-force normalised sensitivity S = dln(tau)/dln(k) of the ignition
    delay to the DME sub-mechanism reactions, at an NTC point."""
    gas = ct.Solution(MECH)
    cand = [i for i, R in enumerate(gas.reactions())
            if any(k in R.equation for k in DME_SPECIES) or R.equation in KEY_BRANCH]
    gas.set_multiplier(1.0)
    base = _idt_simple(gas, T0, P_bar, phi)
    rows = []
    for i in cand:
        gas.set_multiplier(1.0)
        gas.set_multiplier(factor, i)
        tau = _idt_simple(gas, T0, P_bar, phi)
        rows.append((gas.reaction(i).equation, np.log(tau / base) / np.log(factor)))
    gas.set_multiplier(1.0)
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return base, rows[:top]


def arrhenius_Ea(df, Tmin=1050.0):
    """Apparent activation energy (kJ/mol) of the high-temperature branch."""
    hi = df[df.T0 >= Tmin]
    sl, inter = np.polyfit(1.0 / hi.T0, np.log(hi.idt_total), 1)
    return sl * 8.314 / 1000.0, sl, inter


def _plot_sensitivity(base, rows):
    eqs = [r[0] for r in rows][::-1]
    S = [r[1] for r in rows][::-1]
    colors = ["#2166ac" if s < 0 else "#b2182b" for s in S]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(range(len(S)), S, color=colors)
    ax.set_yticks(range(len(S))); ax.set_yticklabels(eqs, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"Normalized sensitivity  $S = d(\ln \tau)/d(\ln k)$")
    ax.set_title(r"Ignition-delay sensitivity of DME/air (825 K, 20 bar, $\phi$=1.0)")
    ax.text(0.97, 0.05, "S < 0: promotes ignition\nS > 0: inhibits ignition",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    fig.savefig("fig6_sensitivity.png"); plt.close(fig)


def _plot_validation(sweeps_p):
    d20, d40 = sweeps_p[20], sweeps_p[40]
    Ea, sl, inter = arrhenius_Ea(d20)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.axvspan(1000 / 1050, 1000 / 800, color="orange", alpha=0.12)
    ax.text(1000 / 920, 8.0, "NTC region\n(Pfahl et al. 1996:\n800-1050 K)",
            ha="center", va="top", fontsize=9, color="#a05a00")
    ax.semilogy(d20.invT, d20.idt_total_ms, "s-", color="#d62728", label="This work, 20 bar")
    ax.semilogy(d40.invT, d40.idt_total_ms, "^-", color="#2ca02c", label="This work, 40 bar")
    xx = np.linspace(1000 / 1300, 1000 / 1050, 50)
    ax.semilogy(xx, np.exp(inter + sl * (xx / 1000.0)) * 1e3, "k--", lw=1.5,
                label=f"High-T Arrhenius fit\n$E_a$ = {Ea:.0f} kJ/mol")
    ax.set_xlabel("1000 / T  (1/K)"); ax.set_ylabel("Ignition delay time  (ms)")
    ax.set_title("Validation: NTC window and high-T activation energy vs literature")
    ax.legend(fontsize=9, loc="upper left"); _add_T_axis(ax)
    fig.savefig("fig7_validation.png"); plt.close(fig)


if __name__ == "__main__":
    main()
