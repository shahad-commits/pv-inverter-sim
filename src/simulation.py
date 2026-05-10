"""top-level simulation runner for the Solar PV Grid-Tied Inverter system....
runs the full conversion chain end-to-end: PV Panel → boost Converter (MPPT) → H-Bridge Inverter → LC Filter → grid
all results are returned as structured dicts and can be passed directly to the plotting module (plots.py) """

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from pv_model        import PVPanel, PVPanelParams
from boost_converter import BoostConverter, BoostParams, design_boost
from mppt            import (PerturbObserve, IncrementalConductance,
                              PandOParams, InCondParams,
                              benchmark_mppt, step_profile, cloudy_profile,
                              ramp_profile)
from inverter        import Inverter, InverterParams, design_lc_filter
from power_quality   import PowerQualityAnalyser, print_harmonic_table, print_ieee1547

#system configuration
@dataclass
class SystemConfig:
    """top-level system parameters"""
    #PV array
    n_panels_series:   int   = 1
    n_panels_parallel: int   = 1
    #grid
    V_grid_rms:  float = 230.0 
    f_grid:      float = 50.0
    #DC bus (full H-bridge: V_dc > V_ac_peak, M_a = V_ac_peak/V_dc < 1)
    V_dc:        float = 400.0
    #switching
    f_sw_boost:  float = 20e3
    f_sw_inv:    float = 10e3
    #simulation
    dt_mppt:     float = 0.05 # s, MPPT update interval
    dt_inverter: float = 1e-6 # s, inverter time step
    #20 full grid cycles (400 ms) ensures LC filter transients have fully settled before steady-state analysis begins...the filter τ ≈ 0.16 ms;
    # 5pi = 0.8 ms, so even 12 cycles (0.24 s) would suffice, 20 cycles provides a comfortable margin and more FFT resolution.
    t_inv_sim: float = 0.40 # s, inverter simulation duration (was 0.12 s → 6 cycles)
    #thermal
    T_cell_default: float = 35.0  # °C

#individual simulation modes
def run_mppt_benchmark(
    config: SystemConfig = None,
    profile_name: str = "step",
    verbose: bool = True,
) -> Dict:
    """benchmark P&O vs Incremental Conductance under a variable irradiance profile """
    cfg = config or SystemConfig()
    panel = PVPanel()
    #select irradiance profile
    dt = cfg.dt_mppt
    if profile_name == "step":
        G_profile = step_profile(dt=dt)
    elif profile_name == "cloudy":
        G_profile = cloudy_profile(dt=dt)
    elif profile_name == "ramp":
        t_ramp = len(step_profile(dt=dt)) * dt
        G_profile = ramp_profile(t_end=t_ramp, dt=dt)
    else:
        raise ValueError(f"Unknown profile: {profile_name}")
    t = np.arange(len(G_profile)) * dt
    if verbose:
        print(f"\n{'─'*50}")
        print(f"  MPPT Benchmark: '{profile_name}' profile")
        print(f"  Duration: {t[-1]:.1f} s  |  Steps: {len(G_profile)}")
        print(f"{'─'*50}")
    algorithms = {
        "P&O":    PerturbObserve(PandOParams(delta_V=0.5)),
        "InCond": IncrementalConductance(InCondParams(delta_V=0.3)),
    }
    t0 = time.perf_counter()
    results = benchmark_mppt(panel, G_profile, T_cell=cfg.T_cell_default,
                              dt=dt, algorithms=algorithms)
    elapsed = time.perf_counter() - t0
    if verbose:
        for name, r in results.items():
            print(f"  {name:8s}  η_overall = {r['eta_overall']*100:.2f}%  "
                  f"η_ss = {r['eta_ss']*100:.2f}%")
        print(f"  Completed in {elapsed:.2f} s")
    #boost converter efficiency at each step (informational)
    bc = BoostConverter(BoostParams(f_sw=cfg.f_sw_boost))
    boost_eta = np.array([
        bc.efficiency(
            V_in=results["P&O"]["V_ref"][k],
            I_in=max(results["P&O"]["P_actual"][k] / max(results["P&O"]["V_ref"][k], 1), 0.01),
            V_out=cfg.V_dc,
        )
        for k in range(len(G_profile))
    ])

    return {
        "profile":      profile_name,
        "t":            t,
        "G":            G_profile,
        "results":      results,
        "boost_eta":    boost_eta,
        "panel_params": panel.p,
    }

def run_power_quality(
    config: SystemConfig = None,
    G: float = 1000.0,
    filter_iterations: bool = True,
    verbose: bool = True,
) -> Dict:
    """simulate inverter output and perform FFT-based THD analysis"""
    cfg = config or SystemConfig()
    #design LC filter
    panel = PVPanel()
    _, _, P_pv = panel.mpp(G=G, T_cell=cfg.T_cell_default)
    filt = design_lc_filter(
        f_sw=cfg.f_sw_inv,
        f_grid=cfg.f_grid,
        V_dc=cfg.V_dc,
        P_rated=P_pv,
    )

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  LC Filter Design (f_sw = {cfg.f_sw_inv/1e3:.0f} kHz)")
        print(f"  L_f = {filt.L_f*1e3:.2f} mH  |  C_f = {filt.C_f*1e6:.2f} µF")
        print(f"  f_corner = {filt.f_c:.1f} Hz  |  Atten = {filt.attenuation_dB:.1f} dB @ f_sw")
        print(f"{'─'*50}")

    #inverter simulation
    inv_params = InverterParams(
        V_dc    = cfg.V_dc,
        f_grid  = cfg.f_grid,
        f_sw    = cfg.f_sw_inv,
        L_f     = filt.L_f,
        C_f     = filt.C_f,
        V_ac_peak = cfg.V_grid_rms * np.sqrt(2),
        R_load  = (cfg.V_grid_rms ** 2) / max(P_pv, 1.0),
    )
    inv = Inverter(inv_params)

    if verbose:
        print(f"  Running inverter simulation ({cfg.t_inv_sim*1e3:.0f} ms)...")
    t0 = time.perf_counter()
    inv_result = inv.simulate(t_end=cfg.t_inv_sim, dt=cfg.dt_inverter)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  P_out = {inv_result['P_out']:.2f} W  |  "
              f"V_rms = {inv_result['V_rms']:.2f} V  |  done in {elapsed:.2f} s")

    #power quality analysis
    #both analysers use the same n_harmonics so the resulting spectra are directly comparable in plots
    # n_harmonics=200 covers up to H200 = 10 kHz which captures all significant PWM sidebands for a 10 kHz switching frequency, the unfiltered V_AB analyser uses 400 harmonics for a complete picture
    #of the raw switching spectrum; the filtered/current analysers use 200 so that comparison plots share the same harmonic axis.
    pq_vab  = PowerQualityAnalyser(f_grid=cfg.f_grid, n_harmonics=400) # full V_AB spectrum
    pq_vout = PowerQualityAnalyser(f_grid=cfg.f_grid, n_harmonics=200) # filtered, comparable
    pq_curr = PowerQualityAnalyser(f_grid=cfg.f_grid, n_harmonics=200) # current, IEEE 1547
    dt_sim = inv_result["t"][1] - inv_result["t"][0]
    spec_unfiltered = pq_vab.analyse(inv_result["V_AB"],  dt_sim)
    spec_filtered   = pq_vout.analyse(inv_result["V_out"], dt_sim)
    spec_current    = pq_curr.analyse(inv_result["I_out"], dt_sim)
    # IEEE 1547-2018 check uses the current spectrum
    I_rated = inv_result["I_rms"] if inv_result["I_rms"] > 0 else 1.0
    ieee1547 = pq_curr.ieee1547_check(spec_current, I_rated=I_rated)
    if verbose:
        print(f"\n  THD (unfiltered V_AB): {spec_unfiltered.THD:.2f}%")
        print(f"  THD (filtered V_out):  {spec_filtered.THD:.2f}%")
        print_ieee1547(ieee1547)
    #optional: sweep filter capacitance
    filter_sweep = None
    if filter_iterations:
        C_sweep = np.logspace(
            np.log10(filt.C_f * 0.1),
            np.log10(filt.C_f * 20.0),
            40,
        )
        #Use pq_vab (n_harmonics=400) for the sweep so all f_sw sidebands up to H400=20 kHz are captured — pq_vout (n_harmonics=200) would truncate at H200=10 kHz, missing harmonics above f_sw..
        #pass the actual inverter R_L so the analytical filter model is consistent with the simulated design.
        _, thd_sweep = pq_vab.thd_vs_capacitance(
            inv_result["V_AB"],
            dt=cfg.dt_inverter,
            L_f=filt.L_f,
            C_values=C_sweep,
            R_load=inv_params.R_load,
            R_L=inv_params.R_L,
        )
        filter_sweep = {"C": C_sweep, "THD": thd_sweep, "C_chosen": filt.C_f}

    return {
        "inv_result":       inv_result,
        "spec_unfiltered":  spec_unfiltered,
        "spec_filtered":    spec_filtered,
        "spec_current":     spec_current,
        "ieee1547":         ieee1547,
        "filter_sweep":     filter_sweep,
        "filter_design":    filt,
        "inv_params":       inv_params,
        "inverter":         inv,
    }

def run_full_system(
    config: SystemConfig = None,
    irradiance_profiles: tuple = ("step", "cloudy"),
    verbose: bool = True,
) -> Dict:
    """run complete end-to-end system simulation and return all results"""
    cfg = config or SystemConfig()
    print("\n" + "═"*60)
    print("  Solar PV Grid-Tied Inverter Simulation")
    print("  Full Conversion Chain Analysis")
    print("═"*60)
    #MPPT benchmarks
    mppt_results = {}
    for profile in irradiance_profiles:
        mppt_results[profile] = run_mppt_benchmark(cfg, profile_name=profile,
                                                    verbose=verbose)
    #power quality
    pq_result = run_power_quality(cfg, G=1000.0,
                                   filter_iterations=True, verbose=verbose)
    #summary
    summary = _build_summary(cfg, mppt_results, pq_result)
    if verbose:
        _print_summary(summary)
    return {
        "mppt_results": mppt_results,
        "pq_result":    pq_result,
        "config":       cfg,
        "summary":      summary,
    }

#summary builder
def _build_summary(cfg, mppt_results, pq_result) -> dict:
    panel = PVPanel()
    #note: this is the nominal operating point (G=1000, T=T_cell_default=35°C), not strict STC (T=25°C)..so keys are named from *_stc to *_nom accordingly
    V_mpp, I_mpp, P_mpp = panel.mpp(G=1000.0, T_cell=cfg.T_cell_default)
    #sse the same switching frequency as the system config, not the default
    bc = BoostConverter(BoostParams(f_sw=cfg.f_sw_boost))
    boost_eta = bc.efficiency(V_in=V_mpp, I_in=I_mpp, V_out=cfg.V_dc)
    mppt_eta = {}
    for profile, res in mppt_results.items():
        for algo, r in res["results"].items():
            key = f"{algo} ({profile})"
            mppt_eta[key] = r["eta_overall"] * 100.0
    return {
        "P_pv_nom": P_mpp, # W  at G=1000, T=T_cell_default (was P_pv_stc)
        "V_mpp_nom": V_mpp, # V
        "I_mpp_nom": I_mpp, # A
        "boost_eta_pct": boost_eta * 100.0,
        "mppt_eta": mppt_eta,
        "THD_unfiltered": pq_result["spec_unfiltered"].THD,
        "THD_filtered": pq_result["spec_filtered"].THD,
        "ieee1547_pass": pq_result["ieee1547"].compliant,
        "V_rms_out": pq_result["inv_result"]["V_rms"],
        "P_out": pq_result["inv_result"]["P_out"],
        "f_c_Hz": pq_result["filter_design"].f_c,
        "L_f_mH": pq_result["filter_design"].L_f * 1e3,
        "C_f_uF": pq_result["filter_design"].C_f * 1e6,
    }

def _print_summary(s: dict):
    print("\n" + "═"*60)
    print("  SIMULATION SUMMARY")
    print("═"*60)
    print(f"  PV Panel (nom.):  P = {s['P_pv_nom']:.1f} W  "
          f"V_mpp = {s['V_mpp_nom']:.2f} V  I_mpp = {s['I_mpp_nom']:.3f} A")
    print(f"  Boost efficiency: {s['boost_eta_pct']:.2f}%")
    print()
    for k, v in s["mppt_eta"].items():
        print(f"  MPPT η  {k:22s}: {v:.2f}%")
    print()
    print(f"  THD (unfiltered): {s['THD_unfiltered']:.2f}%")
    print(f"  THD (filtered):   {s['THD_filtered']:.2f}%")
    std = "PASS ✓" if s["ieee1547_pass"] else "FAIL ✗"
    print(f"  IEEE 1547-2018:   {std}")
    print(f"  V_rms output:     {s['V_rms_out']:.2f} V")
    print(f"  P_out:            {s['P_out']:.2f} W")
    print(f"  LC filter:  L={s['L_f_mH']:.2f} mH  C={s['C_f_uF']:.2f} µF  f_c={s['f_c_Hz']:.0f} Hz")
    print("═"*60 + "\n")

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    results = run_full_system(verbose=True)