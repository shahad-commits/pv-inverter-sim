"""figure catalogue:
Fig 1: PV panel I-V and P-V curves at five irradiance levels
Fig 2: MPP locus overlay on I-V family
Fig 3: MPPT algorithm comparison: tracked power vs time (step profile)
Fig 4: MPPT algorithm comparison: tracked power vs time (cloudy profile)
Fig 5: boost converter: duty cycle and efficiency vs V_in
Fig 6: inverter output waveforms: V_AB, V_out, I_out (3 cycles)
Fig 7: FFT harmonic spectrum (unfiltered vs filtered)
Fig 8: LC filter Bode plot (magnitude + phase)
Fig 9: THD vs filter capacitance (design iteration sweep)
Fig 10: IEEE 1547 harmonic bar chart with compliance limits"""

from __future__ import annotations
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg") #non-interactive backend for CI / headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines  import Line2D
from matplotlib.ticker import LogFormatter, MultipleLocator, AutoMinorLocator
from typing import Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning)

STYLE = {
    #Background & surfaces
    "fig_face":   "#FBF9F2", 
    "axes_face":  "#F5F3EA", 
    "grid_color": "#DDD9C8",
    "spine":      "#B8B4A0",
    #text
    "text": "#2E2B22", 
    "C0": "#9688CC",
    "C1": "#C47FAF",
    "C2": "#6BA58E",
    "C3": "#8F87B7",
    "C4": "#B8965A",
    "C5": "#7DB4C0",
    #semantic colours 
    "mpp":   "#B8965A",
    "limit": "#C0524A",
}
IRRADIANCE_COLORS = {
    200:  "#BFB6E0",
    400:  "#9688CC",
    600:  "#6BA58E",
    800:  "#B8965A",
    1000: "#C47FAF",
}
FONT_TITLE  = dict(fontsize=11, fontweight="bold", color=STYLE["text"])
FONT_LABEL  = dict(fontsize=9,  color=STYLE["text"])
FONT_TICK   = dict(labelsize=8, colors=STYLE["text"])
FONT_LEGEND = dict(fontsize=8,  facecolor=STYLE["fig_face"],
                   edgecolor=STYLE["spine"], labelcolor=STYLE["text"])

FIG_DPI  = 300
FIG_EXT  = ["png", "pdf"]

def _apply_style(ax, xlabel="", ylabel="", title="", grid=True):
    ax.set_facecolor(STYLE["axes_face"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["spine"])
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", **FONT_TICK)
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])
    if xlabel: ax.set_xlabel(xlabel, **FONT_LABEL)
    if ylabel: ax.set_ylabel(ylabel, **FONT_LABEL)
    if title:  ax.set_title(title,  **FONT_TITLE)
    if grid:
        ax.grid(True, color=STYLE["grid_color"], linewidth=0.6, alpha=0.9,
                linestyle="-")
        ax.set_axisbelow(True)


def _new_fig(nrows=1, ncols=1, figsize=(10, 6), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              facecolor=STYLE["fig_face"], **kw)
    fig.patch.set_facecolor(STYLE["fig_face"])
    return fig, axes

def _save(fig, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in FIG_EXT:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {name}.{{png,pdf}}")

#Fig 1 & 2, PV I-V / P-V family + MPP locus
def plot_iv_curves(panel_data: dict, out_dir: str):
    """panel_data: output of pv_model.characterise()"""
    fig, (ax1, ax2) = _new_fig(1, 2, figsize=(12, 5))
    fig.suptitle("PV Panel I–V and P–V Characteristics",
                 fontsize=13, fontweight="bold", color=STYLE["text"], y=1.01)
    mpp_V, mpp_I, mpp_P = [], [], []
    for G, data in sorted(panel_data.items()):
        c = IRRADIANCE_COLORS.get(G, STYLE["C0"])
        lbl = f"G = {G} W/m²"
        ax1.plot(data["V"], data["I"], color=c, lw=1.8, label=lbl)
        ax2.plot(data["V"], data["P"], color=c, lw=1.8, label=lbl)
        mpp_V.append(data["V_mpp"])
        mpp_I.append(data["I_mpp"])
        mpp_P.append(data["P_mpp"])
    #MPP locus
    ax1.plot(mpp_V, mpp_I, "--o", color=STYLE["mpp"], ms=5,
             lw=1.2, label="MPP locus", zorder=5)
    ax2.plot(mpp_V, mpp_P, "--o", color=STYLE["mpp"], ms=5,
             lw=1.2, label="MPP locus", zorder=5)

    _apply_style(ax1, "Terminal Voltage V [V]", "Current I [A]", "I–V Characteristic")
    _apply_style(ax2, "Terminal Voltage V [V]", "Power P [W]",   "P–V Characteristic")
    for ax in (ax1, ax2):
        ax.legend(**FONT_LEGEND)

    fig.tight_layout()
    _save(fig, "fig01_pv_iv_curves", out_dir)
#Fig 2, MPP locus: V_mpp and P_mpp vs irradiance (parametric)
def plot_mpp_locus(panel_data: dict, out_dir: str):
    """standalone MPP locus figure — V_mpp(G) and P_mpp(G) vs irradiance"""
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(__file__))
    from pv_model import PVPanel
    panel = PVPanel()
    G_arr = np.array(sorted(panel_data.keys())) # use same G values as fig01
    #compute MPP at T=25°C (STC) and T=35°C (operating condition)
    temps = [(25, "#58a6ff", "T = 25 °C  (STC)"),
             (35, "#ffa657", "T = 35 °C  (operating)")]

    fig, (ax1, ax2) = _new_fig(1, 2, figsize=(11, 5))
    fig.suptitle("MPP Locus: V_mpp and P_mpp vs Irradiance",
                 fontsize=13, fontweight="bold", color=STYLE["text"], y=1.01)

    for T, color, label in temps:
        V_mpp_arr = np.array([panel.mpp(G=G, T_cell=T)[0] for G in G_arr])
        P_mpp_arr = np.array([panel.mpp(G=G, T_cell=T)[2] for G in G_arr])
        ax1.plot(G_arr, V_mpp_arr, "o-", color=color, lw=1.8, ms=5, label=label)
        ax2.plot(G_arr, P_mpp_arr, "o-", color=color, lw=1.8, ms=5, label=label)
    #mark the simulation operating point: G=1000, T=35°C
    V_op, _, P_op = panel.mpp(G=1000, T_cell=35)
    ax1.axhline(V_op, color=STYLE["C2"], lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(P_op, color=STYLE["C2"], lw=0.8, ls="--", alpha=0.6)
    ax1.annotate(f"V_mpp = {V_op:.2f} V",
                 xy=(1000, V_op), xytext=(600, V_op + 0.3),
                 fontsize=8, color=STYLE["C2"],
                 arrowprops=dict(arrowstyle="->", color=STYLE["C2"], lw=0.8))
    ax2.annotate(f"P_mpp = {P_op:.1f} W",
                 xy=(1000, P_op), xytext=(600, P_op + 8),
                 fontsize=8, color=STYLE["C2"],
                 arrowprops=dict(arrowstyle="->", color=STYLE["C2"], lw=0.8))

    _apply_style(ax1, "Irradiance G [W/m²]", "V_mpp [V]",
                 "MPP Voltage vs Irradiance")
    _apply_style(ax2, "Irradiance G [W/m²]", "P_mpp [W]",
                 "MPP Power vs Irradiance")
    for ax in (ax1, ax2):
        ax.legend(**FONT_LEGEND)
    fig.tight_layout()
    _save(fig, "fig02_mpp_locus", out_dir)

#Fig 3 & 4, MPPT benchmark
def plot_mppt_comparison(mppt_data: dict, profile_name: str,
                          fig_num: int, out_dir: str):
    """mppt_data: one profile's dict from run_mppt_benchmark()"""
    results = mppt_data["results"]
    t       = mppt_data["t"]
    G       = mppt_data["G"]
    algo_colors = {"P&O": STYLE["C0"], "InCond": STYLE["C1"]}
    fig = plt.figure(figsize=(12, 8), facecolor=STYLE["fig_face"])
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1.2, 2, 1], hspace=0.05,
                             figure=fig)

    ax_G   = fig.add_subplot(gs[0])
    ax_P   = fig.add_subplot(gs[1], sharex=ax_G)
    ax_err = fig.add_subplot(gs[2], sharex=ax_G)
    #irradiance
    ax_G.fill_between(t, G, alpha=0.25, color=STYLE["C4"])
    ax_G.plot(t, G, color=STYLE["C4"], lw=1.5)
    _apply_style(ax_G, "", "G [W/m²]", f"MPPT Benchmark — '{profile_name}' profile")
    ax_G.tick_params(labelbottom=False)
    #power
    r_ref = list(results.values())[0]
    ax_P.fill_between(t, r_ref["P_mpp"], alpha=0.1, color=STYLE["C2"])  # no label — line carries it
    ax_P.plot(t, r_ref["P_mpp"], "--", color=STYLE["C2"], lw=1.8,
              label="P_MPP (theoretical)", zorder=5)  # zorder=5 → drawn on top of algo lines

    for name, r in results.items():
        ax_P.plot(t, r["P_actual"], color=algo_colors[name], lw=1.5,
                  label=f"{name}  η={r['eta_overall']*100:.2f}%")

    _apply_style(ax_P, "", "Power [W]", "")
    ax_P.legend(**FONT_LEGEND)
    ax_P.tick_params(labelbottom=False)
    #tracking error (normalised)
    for name, r in results.items():
        err = 100.0 * (r["P_mpp"] - r["P_actual"]) / np.maximum(r["P_mpp"], 1e-3)
        ax_err.plot(t, err, color=algo_colors[name], lw=1.2, label=name)

    ax_err.axhline(0, color=STYLE["spine"], lw=0.8, ls="--")
    _apply_style(ax_err, "Time [s]", "Tracking error [%]", "")
    ax_err.legend(**FONT_LEGEND)
    #adaptive y-limits: step profile has small steady-state oscillation (<1%)
    #that needs a tight scale to be visible; cloudy has large transient spikes (≤20%)
    if profile_name == "step":
        ax_err.set_ylim(-1, 5) #shows P&O ±0.1-0.5% oscillation clearly
    else:
        ax_err.set_ylim(-2, 22) #accommodates ±20% transient spikes during clouds
    plt.setp(ax_G.get_xticklabels(), visible=False)
    plt.setp(ax_P.get_xticklabels(), visible=False)
    _save(fig, f"fig0{fig_num}_mppt_{profile_name}", out_dir)

#Fig 5, boost converter characteristics
def plot_boost_characteristics(out_dir: str):
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(__file__))
    from boost_converter import BoostConverter, BoostParams
    from pv_model import PVPanel
    bc = BoostConverter()
    V_in_arr = np.linspace(15, 45, 200)
    V_out    = 400.0
#Use the nominal operating point (G=1000, T=35°C) for the vertical marker,
    _panel = PVPanel()
    V_mpp_nom, I_mpp_nom, P_mpp_nom = _panel.mpp(G=1000.0, T_cell=35.0)

    D_arr   = np.array([bc.duty_cycle(v, V_out) for v in V_in_arr])
    I_in_arr = P_mpp_nom / V_in_arr           # constant-power: I = P/V
    eta_arr  = np.array([bc.efficiency(v, i, V_out)
                         for v, i in zip(V_in_arr, I_in_arr)])
    dIL_arr = np.array([bc.inductor_ripple(v, bc.duty_cycle(v, V_out)) for v in V_in_arr])
    fig, axes = _new_fig(1, 3, figsize=(14, 4.5))
    for ax in axes:
        ax.set_facecolor(STYLE["axes_face"])
    axes[0].plot(V_in_arr, D_arr * 100, color=STYLE["C0"], lw=2)
    axes[0].axvline(V_mpp_nom, color=STYLE["mpp"], ls="--", lw=1.2,
                    label=f"V_mpp = {V_mpp_nom:.2f} V (nom.)")
    #annotate the 95% clip region (V_in < 21V is physically outside panel range)
    axes[0].axhline(95.0, color=STYLE["spine"], ls=":", lw=0.8)
    axes[0].text(15.5, 95.15, "D clipped at 95%\n(V_in < 21 V outside\npanel operating range)",
                 color=STYLE["text"], fontsize=6.5, va="bottom")
    axes[0].legend(**FONT_LEGEND)
    _apply_style(axes[0], "V_in [V]", "Duty Cycle D [%]", "Duty Cycle vs Input Voltage")

    axes[1].plot(V_in_arr, eta_arr * 100, color=STYLE["C1"], lw=2)
    axes[1].axvline(V_mpp_nom, color=STYLE["mpp"], ls="--", lw=1.2,
                    label=f"V_mpp = {V_mpp_nom:.2f} V")
    axes[1].legend(**FONT_LEGEND)
    #set ylim from just below the minimum efficiency value (avoid wasted whitespace)
    eta_min = float(np.min(eta_arr)) * 100
    axes[1].set_ylim(max(90.0, eta_min - 1.0), 100.0)
    _apply_style(axes[1], "V_in [V]", "Efficiency η [%]",
                 "Boost Converter Efficiency\n(constant-power locus, P = {:.0f} W)".format(P_mpp_nom))

    axes[2].plot(V_in_arr, dIL_arr, color=STYLE["C2"], lw=2)
    axes[2].axvline(V_mpp_nom, color=STYLE["mpp"], ls="--", lw=1.2,
                    label=f"V_mpp = {V_mpp_nom:.2f} V")
    axes[2].legend(**FONT_LEGEND)
    _apply_style(axes[2], "V_in [V]", "ΔI_L [A]", "Inductor Current Ripple")

    fig.tight_layout()
    _save(fig, "fig05_boost_characteristics", out_dir)


#Fig 6, inverter waveforms

def plot_inverter_waveforms(pq_result: dict, out_dir: str):
    r    = pq_result["inv_result"]
    t    = r["t"]
    f_g  = pq_result["inv_params"].f_grid
    T_g  = 1.0 / f_g
    #show last 3 cycles
    dt   = t[1] - t[0]
    n3   = int(3 * T_g / dt)
    sl   = slice(len(t) - n3, len(t))

    t3   = t[sl] - t[sl][0]

    fig, axes = _new_fig(3, 1, figsize=(12, 8),
                          gridspec_kw={"hspace": 0.08})

    axes[0].plot(t3 * 1e3, r["V_AB"][sl],  color=STYLE["C3"], lw=0.8,
                 label="V_AB (switching)")
    axes[0].plot(t3 * 1e3, r["V_out"][sl], color=STYLE["C0"], lw=2.0,
                 label="V_out (filtered)")
    _apply_style(axes[0], "", "Voltage [V]", "Inverter Output Waveforms")
    axes[0].legend(**FONT_LEGEND, loc="upper right")
    axes[0].tick_params(labelbottom=False)

    #inset: 2 ms zoom showing 3-level unipolar SPWM 
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(axes[0], width="28%", height="42%",
                        loc="lower left", borderpad=1.5)

    t_zoom_start_ms = 4.0 #start of zoom window within displayed t3 [ms]
    n_zoom  = int(2e-3 / dt) #2 000 samples = 2 ms
    idx0    = int(t_zoom_start_ms * 1e-3 / dt) #sample offset within sl
    sl_zoom = slice(len(t) - n3 + idx0,
                    len(t) - n3 + idx0 + n_zoom)
    t_zoom  = (r["t"][sl_zoom] - r["t"][sl_zoom][0]) * 1e3  # 0..2 ms

    ax_ins.plot(t_zoom, r["V_AB"][sl_zoom],  color=STYLE["C3"], lw=0.9)
    ax_ins.plot(t_zoom, r["V_out"][sl_zoom], color=STYLE["C0"], lw=1.4)
    #add horizontal reference lines at the three SPWM levels
    for level, ls in [(400, ":"), (0, "--"), (-400, ":")]:
        ax_ins.axhline(level, color=STYLE["spine"], lw=0.5, ls=ls, alpha=0.6)

    ax_ins.set_xlim(0, 2)
    ax_ins.set_ylim(-460, 460)
    ax_ins.set_yticks([-400, 0, 400])
    ax_ins.set_yticklabels(["-400", "0", "+400"], fontsize=5.5)
    ax_ins.set_xticks([0, 1, 2])
    ax_ins.set_xticklabels(["0", "1", "2"], fontsize=5.5)
    ax_ins.set_facecolor(STYLE["axes_face"])
    ax_ins.tick_params(labelsize=5.5, colors=STYLE["text"], length=2)
    for sp in ax_ins.spines.values():
        sp.set_edgecolor(STYLE["C0"])
        sp.set_linewidth(1.0)
    ax_ins.set_xlabel("Time [ms]", fontsize=6, color=STYLE["text"], labelpad=1)
    ax_ins.set_ylabel("V [V]", fontsize=6, color=STYLE["text"], labelpad=1)
    ax_ins.set_title("2 ms zoom — 3-level SPWM", fontsize=6.5,
                     color=STYLE["text"], pad=2, fontweight="bold")

    axes[1].plot(t3 * 1e3, r["I_L"][sl],   color=STYLE["C1"], lw=1.5,
                 label="I_L (inductor)")
    axes[1].plot(t3 * 1e3, r["I_out"][sl], color=STYLE["C2"], lw=1.5,
                 label="I_out (load)")
    _apply_style(axes[1], "", "Current [A]", "")
    axes[1].legend(**FONT_LEGEND)
    axes[1].tick_params(labelbottom=False)

    #instantaneous power
    p_inst = r["V_out"][sl] * r["I_out"][sl]
    axes[2].plot(t3 * 1e3, p_inst, color=STYLE["C4"], lw=1.5)
    axes[2].axhline(np.mean(p_inst), color=STYLE["C2"], ls="--", lw=1.2,
                    label=f"P_avg = {np.mean(p_inst):.1f} W")
    _apply_style(axes[2], "Time [ms]", "Power [W]", "")
    axes[2].legend(**FONT_LEGEND)

    fig.tight_layout()
    _save(fig, "fig06_inverter_waveforms", out_dir)


#Fig 7,Harmonic spectrum
#panel 1: continuous log-frequency spectrum (Hz axis 50 Hz→20 kHz): V_AB (unfiltered) and V_out (filtered) on the same plot
#the dominant 2·f_sw switching cluster and the low-order region are both visible; the filter's attenuation is immediately obvious
#panel 2: Low-order bar chart H2–H29 (filtered V_out only, log y): resolves the small residual harmonics and their compliance margin

def plot_harmonic_spectrum(pq_result: dict, out_dir: str):
    s_unfilt   = pq_result["spec_unfiltered"] #V_AB,  n_harmonics=400
    s_filt     = pq_result["spec_filtered"] #V_out, n_harmonics=200
    inv_params = pq_result["inv_params"]
    fund_ref = max(s_unfilt.fundamental, 1e-9)
    f_grid   = float(s_unfilt.f_fund) #50 Hz
    f_sw     = float(inv_params.f_sw) #10000 Hz
    f_c      = float(pq_result["filter_design"].f_c) #1000 Hz
    #continuous frequency arrays (% of V₁)
    freq_u    = s_unfilt.harmonics * f_grid # H1..H400 -> 50..20 000 Hz
    mag_u_pct = s_unfilt.magnitudes / fund_ref * 100

    freq_f    = s_filt.harmonics * f_grid # H1..H200 -> 50..10 000 Hz
    mag_f_pct = s_filt.magnitudes / fund_ref * 100
    fig, (ax1, ax2) = _new_fig(2, 1, figsize=(13, 8),
                                gridspec_kw={"height_ratios": [2, 1.4], "hspace": 0.12})

    #panel 1: full continuous spectrum (log x, log y) 
    ax1.semilogy(freq_u, np.maximum(mag_u_pct, 1e-3),
                 color=STYLE["C3"], lw=1.0, alpha=0.75,
                 label=f"V_AB unfiltered   THD = {s_unfilt.THD:.1f}%")
    ax1.semilogy(freq_f, np.maximum(mag_f_pct, 1e-3),
                 color=STYLE["C0"], lw=1.8,
                 label=f"V_out filtered    THD = {s_filt.THD:.2f}%")

    ax1.axvline(f_c,    color=STYLE["C2"],    ls="--", lw=1.2,
                label=f"f_c = {f_c:.0f} Hz (filter corner)")
    ax1.axvline(2*f_sw, color=STYLE["mpp"],   ls=":",  lw=1.2,
                label=f"2·f_sw = {2*f_sw/1e3:.0f} kHz (unipolar SPWM cluster)")
    ax1.axhline(5.0,    color=STYLE["limit"], ls="--", lw=1.2,
                label="5% voltage THD limit (IEC 61727 / IEEE 1547-2018 §6.4)")
    ax1.set_xscale("log")
    ax1.set_xlim(40, 21000)
    ax1.set_ylim(1e-3, 200)
    #annotate the dominant SPWM sideband if it is within the array range
    if len(mag_u_pct) >= 399:
        h_dom_pct = mag_u_pct[398] #H399 = 2·f_sw − f_grid
        f_dom     = 399 * f_grid #19950 Hz
        if h_dom_pct > 1.0:
            ax1.annotate(
                f"2·f_sw − f_grid\n({f_dom:.0f} Hz, {h_dom_pct:.1f}%)",
                xy=(f_dom, h_dom_pct),
                xytext=(5000, max(h_dom_pct * 0.15, 0.05)),
                fontsize=7.5, color=STYLE["C3"],
                arrowprops=dict(arrowstyle="->", color=STYLE["C3"], lw=0.8),
            )

    _apply_style(ax1,
                 "",
                 "Harmonic magnitude (% of V₁, log scale)",
                 "Voltage Harmonic Spectrum — V_AB (unfiltered) vs V_out (filtered)\n"
                 "Unipolar SPWM: all switching energy lives at 2·f_sw sidebands; "
                 "LC filter removes them")
    ax1.legend(**FONT_LEGEND, loc="lower left")
    ax1.tick_params(labelbottom=False)

    #panel 2: low-order bar chart, filtered only (H2–H29)
    n_bar      = min(29, s_filt.n_harmonics)
    orders_bar = s_filt.harmonics[1:n_bar].astype(int) # H2..H29 (exclude H1)
    mag_bar    = np.maximum(s_filt.magnitudes[1:n_bar] / fund_ref * 100, 1e-3)

    x = np.arange(len(orders_bar))
    ax2.bar(x, mag_bar, width=0.7, color=STYLE["C0"], alpha=0.85,
            label=f"Filtered V_out (H2–H{n_bar}),  THD = {s_filt.THD:.2f}%")
    ax2.axhline(5.0, color=STYLE["limit"], ls="--", lw=1.2,
                label="5% voltage THD limit (IEC 61727 / IEEE 1547-2018 §6.4)")

    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 20)
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels(orders_bar[::2])
    ax2.set_xlim(-1, len(orders_bar))

    _apply_style(ax2,
                 "Harmonic Order",
                 "Magnitude [%] — log scale",
                 "Low-Order Residual Harmonics in Filtered Output (H2–H29)")
    ax2.legend(**FONT_LEGEND)

    fig.tight_layout()
    _save(fig, "fig07_harmonic_spectrum", out_dir)

# Fig 8, bode plot
def plot_bode(pq_result: dict, out_dir: str):
    inv = pq_result["inverter"]
    f, mag, phase = inv.bode(f_min=10, f_max=1e6)

    f_c  = pq_result["filter_design"].f_c
    f_sw = pq_result["inv_params"].f_sw

    fig, (ax1, ax2) = _new_fig(2, 1, figsize=(10, 7),
                                 gridspec_kw={"hspace": 0.08})

    ax1.semilogx(f, mag, color=STYLE["C0"], lw=2, label="|H(jω)|")
    ax1.axvline(f_c,  color=STYLE["C2"],    ls="--", lw=1.2, label=f"f_c = {f_c:.0f} Hz")
    ax1.axvline(f_sw, color=STYLE["limit"], ls="--", lw=1.2, label=f"f_sw = {f_sw/1e3:.0f} kHz")
    ax1.axhline(-40,  color=STYLE["C5"], ls=":",  lw=1.2,
                label="-40 dB reference")
    _apply_style(ax1, "", "Magnitude [dB]", "LC Filter Transfer Function H(jω) = V_out / V_AB")
    ax1.legend(**FONT_LEGEND)
    ax1.tick_params(labelbottom=False)
    ax2.semilogx(f, phase, color=STYLE["C1"], lw=2)
    ax2.axvline(f_c,  color=STYLE["C2"],    ls="--", lw=1.2)
    ax2.axvline(f_sw, color=STYLE["limit"], ls="--", lw=1.2)
    ax2.yaxis.set_major_locator(MultipleLocator(45))
    _apply_style(ax2, "Frequency [Hz]", "Phase [°]", "")

    fig.tight_layout()
    _save(fig, "fig08_bode_plot", out_dir)


#Fig 9, THD vs filter capacitance
def plot_thd_sweep(pq_result: dict, out_dir: str):
    sweep = pq_result["filter_sweep"]
    if sweep is None:
        return
    C_uF    = sweep["C"] * 1e6
    thd     = sweep["THD"]
    C_chose = sweep["C_chosen"] * 1e6

    fig, ax = _new_fig(figsize=(9, 5))
    ax.semilogx(C_uF, thd, color=STYLE["C0"], lw=2)
    ax.axhline(5.0, color=STYLE["limit"], ls="--", lw=1.5,
               label="5% voltage THD limit (IEC 61727 / IEEE 1547-2018 §6.4)")
    ax.axvline(C_chose, color=STYLE["mpp"], ls="--", lw=1.5,
               label=f"Design choice  C = {C_chose:.1f} µF")
    ax.fill_between(C_uF, thd, 5.0,
                    where=(thd < 5.0), alpha=0.15, color=STYLE["C2"],
                    label="Compliant region")

    _apply_style(ax, "Filter Capacitance C_f [µF]",
                 "Output Voltage THD [%]",
                 "LC Filter Design Iteration, THD vs Capacitance")
    ax.legend(**FONT_LEGEND)
    #ylim must reach at least 5% so the compliance limit line is visible
    ax.set_ylim(0, max(5.5, thd.max() * 1.2))

    fig.tight_layout()
    _save(fig, "fig09_thd_vs_capacitance", out_dir)

#Fig 10, IEEE 1547 harmonic bar chart
def plot_ieee1547_bar(pq_result: dict, out_dir: str):
    result = pq_result["ieee1547"]
    actuals = result.harmonic_actuals
    limits  = result.harmonic_limits
    orders  = sorted(actuals.keys())[:25]
    act_pct = np.array([actuals[h] for h in orders])
    lim_pct = np.array([limits.get(h, 5.0) for h in orders])
    colors = [STYLE["C1"] if a > l else STYLE["C0"]
              for a, l in zip(act_pct, lim_pct)]
    fig, ax = _new_fig(figsize=(13, 5))
    x = np.arange(len(orders))
    ax.bar(x, act_pct, color=colors, alpha=0.85, width=0.6,
           label="Measured harmonic current [% of rated]")
    ax.step(np.append(x - 0.5, x[-1] + 0.5),
            np.append(lim_pct, lim_pct[-1]),
            where="post", color=STYLE["limit"], lw=1.8, label="IEEE 1547-2018 limit")

    status = "PASS ✓" if result.compliant else "FAIL ✗"
    col    = STYLE["C2"] if result.compliant else STYLE["limit"]
    ax.text(0.98, 0.95, f"THD = {result.thd_actual:.2f}%\n{status}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=STYLE["axes_face"],
                      edgecolor=col, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in orders], rotation=45, ha="right", fontsize=7)
    _apply_style(ax, "Harmonic Order",
                 "Current Harmonic [% of rated I₁]",
                 "IEEE 1547-2018 Compliance, Individual Current Harmonics")
    ax.legend(**FONT_LEGEND)
    #set ylim to 4.5%, the highest limit step is 4.0% (H3-H9)... without explicit ylim matplotlib auto-scales to 4.0%, leaving no headroom and making the tallest limit bars appear cut off at the top edge
    ax.set_ylim(0, 4.5)
    fig.tight_layout()
    _save(fig, "fig10_ieee1547_compliance", out_dir)


#generation function
def generate_all_figures(sim_results: dict, out_dir: str = "../figures"):
    """generate all 10 figures from full simulation results"""
    from pv_model import characterise

    print(f"\n{'─'*50}")
    print(f"  Generating figures → {os.path.abspath(out_dir)}")
    print(f"{'─'*50}")
    #Fig 1, I-V curves
    panel_data = characterise(G_values=[200, 400, 600, 800, 1000])
    plot_iv_curves(panel_data, out_dir)
    #Fig 2, MPP locus (V_mpp and P_mpp vs G, two temperatures)
    plot_mpp_locus(panel_data, out_dir)
    #Fig 3 & 4, MPPT
    for fig_num, (profile, data) in enumerate(
        sim_results["mppt_results"].items(), start=3
    ):
        plot_mppt_comparison(data, profile, fig_num, out_dir)

    #Fig 5, Boost
    plot_boost_characteristics(out_dir)
    #Figs 6–10, Power quality
    pq = sim_results["pq_result"]
    plot_inverter_waveforms(pq,  out_dir)
    plot_harmonic_spectrum(pq,   out_dir)
    plot_bode(pq,                out_dir)
    plot_thd_sweep(pq,           out_dir)
    plot_ieee1547_bar(pq,        out_dir)
    print(f"{'─'*50}")
    print("  All figures generated.")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation import run_full_system
    results = run_full_system(verbose=True)
    generate_all_figures(results, out_dir="../figures")