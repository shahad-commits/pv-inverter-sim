"""average-value and switched model of a DC-DC boost converter suitable for
interfacing a PV panel to a DC bus."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

#converter parameters

@dataclass
class BoostParams:
    """design parameters for the boost stage"""
    L: float = 2.0e-3 #H boost inductor
    C: float = 470e-6 #F output capacitor
    R_L: float = 0.05 #inductor ESR
    R_C: float = 0.02 #capacitor ESR
    f_sw: float = 20e3 #Hz switching frequency
    V_in_nom: float = 30.5 #V nominal input (panel V_mpp)
    V_out_nom: float = 400.0#V nominal output (DC bus)
    V_diode: float = 0.7  #V diode forward drop
    R_sw: float = 0.01 #MOSFET on-resistance

#average-value model
class BoostConverter:
    """average-value model and steady-state analyser for a boost converter"""
    def __init__(self, params: BoostParams = None):
        self.p = params or BoostParams()
    #steady-state analysis
    def duty_cycle(self, V_in: float, V_out: float) -> float:
        """Compute required duty cycle for given V_in → V_out [CCM, with losses].
        Uses iterative fixed-point solution accounting for diode drop and
        resistive losses..raises ValueError if V_in >= V_out (boost converter cannot step down)"""
        if V_in <= 0:
            raise ValueError(f"V_in must be positive, got {V_in:.3f} V")
        if V_in >= V_out:
            raise ValueError(
                f"Boost converter cannot step down: V_in={V_in:.3f} V >= V_out={V_out:.3f} V. "
                f"Reduce V_in or increase V_out."
            )
        p = self.p
        #first-order lossless estimate
        D = 1.0 - (V_in / V_out)
        D = np.clip(D, 0.05, 0.95)
        # fixed-point iteration with convergence check: from CCM volt-second balance (loss-corrected):
        #V_out * (1-D) = V_in - D*R_L*I_in - (1-D)*V_diode
        #I_in-independent approximation (valid when I_in*R_L << V_in):
        #D = 1 - V_in / (V_out + V_diode + D*R_L)
        D_prev = D
        for iteration in range(20): # up to 20 iterations (was hard-coded 5)
            D_new = 1.0 - V_in / (V_out + p.V_diode + D * p.R_L)
            D = np.clip(D_new, 0.05, 0.95)
            if abs(D - D_prev) < 1e-7:   # converged
                break
            D_prev = D
        return float(D)
    def output_voltage(self, V_in: float, D: float, I_out: float = 0.0) -> float:
        """output voltage given V_in, duty cycle D, and load current
        Accounts for diode drop and inductor ESR"""
        p = self.p
        D_prime = 1.0 - D
        V_out = (V_in - D * p.R_L * I_out / D_prime - D_prime * p.V_diode) / D_prime
        return max(V_out, 0.0)
    def efficiency(self, V_in: float, I_in: float, V_out: float) -> float:
        """compute converter efficiency accounting for conduction losses..
        Loss model (averaged CCM):
        P_sw= D × I_in² × R_sw MOSFET on-state conduction loss
        (switch carries ~I_in for D·T_sw fraction of each cycle)
        P_diode = (1−D) × I_in × V_diode    Diode conduction loss
        average diode current = I_in×(1−D); V_diode is a fixed drop...switching losses (gate drive, reverse recovery) are not modelled here."""
        p = self.p
        P_in = V_in * I_in
        if P_in <= 0:
            return 0.0
        D = self.duty_cycle(V_in, V_out)
        #conduction losses
        P_sw    = D * I_in ** 2 * p.R_sw # MOSFET
        P_diode = (1.0 - D) * I_in * p.V_diode # diode: (1-D)×I_in×V_f
        P_L     = I_in ** 2 * p.R_L # inductor ESR
        P_losses = P_sw + P_diode + P_L
        eta = (P_in - P_losses) / P_in
        return float(np.clip(eta, 0.0, 1.0))

    def inductor_ripple(self, V_in: float, D: float) -> float:
        """peak-to-peak inductor current ripple [A]"""
        p = self.p
        return V_in * D / (p.L * p.f_sw)
    def capacitor_ripple(self, I_out: float, D: float) -> float:
        """peak-to-peak output voltage ripple [V]"""
        p = self.p
        return I_out * D / (p.C * p.f_sw)
    def ccm_boundary_current(self, V_in: float, V_out: float) -> float:
        """critical inductor current at CCM/DCM boundary [A], below this average input current the converter enters DCM"""
        p = self.p
        D = self.duty_cycle(V_in, V_out)
        return V_in * D / (2 * p.L * p.f_sw)

    #small-signal state-space (for controller design)
    def small_signal_model(self, V_in: float, D: float, I_L: float, V_out: float,
                            R_load: float = 200.0):
        """return (A, B, C, E) matrices of the averaged CCM small-signal model"""
        p = self.p
        D_prime = 1.0 - D
        A = np.array([
            [-(p.R_L) / p.L,          -D_prime / p.L      ],
            [ D_prime / p.C,           -1.0 / (R_load * p.C)],  # corrected: R_load*C
        ])
        B = np.array([
            [1.0 / p.L,        V_out / p.L],
            [0.0,              -I_L  / p.C],
        ])

        C_out = np.array([[0.0, 1.0]])
        E     = np.array([[0.0, 0.0]])

        return A, B, C_out, E
    #time-domain switched simulation (euler integration)
    def simulate(
        self,
        V_in_profile: np.ndarray,
        D_profile: np.ndarray,
        t_end: float = None,
        dt: float = None,
        R_load: float = 50.0,
    ) -> dict:
        """simulate switched converter waveforms"""
        p = self.p
        T_sw = 1.0 / p.f_sw
        n_periods = len(V_in_profile)
        dt_sim = dt or T_sw / 100.0
        steps_per_period = max(1, int(round(T_sw / dt_sim)))
        total_steps = n_periods * steps_per_period
        t_arr  = np.zeros(total_steps)
        V_out_arr = np.zeros(total_steps)
        I_L_arr = np.zeros(total_steps)
        V_in_arr = np.zeros(total_steps)
        D_arr = np.zeros(total_steps)
        #initial conditions: use steady-state estimates to minimise startup transient
        #in CCM: I_L,avg = I_out / (1-D) = (V_out_nom/R_load) / (1-D_ss), D_ss ≈ 1 - V_in[0]/V_out_nom (lossless estimate)
        D_ss_init = float(np.clip(1.0 - V_in_profile[0] / p.V_out_nom, 0.05, 0.95))
        I_out_est  = p.V_out_nom / R_load
        I_L  = I_out_est / max(1.0 - D_ss_init, 0.05) #CCM steady-state inductor current
        V_C  = min(p.V_out_nom, V_in_profile[0] / max(1 - D_profile[0], 0.05))
        t    = 0.0
        step = 0
        for period_idx in range(n_periods):
            V_in = float(V_in_profile[period_idx])
            D    = float(np.clip(D_profile[period_idx], 0.05, 0.95))
            t_on = D * T_sw
            t_period = 0.0
            for sub_step in range(steps_per_period):
                if step >= total_steps:
                    break
                #switch state
                sw_on = (t_period < t_on)
                if sw_on:
                    #switch ON: diode OFF
                    dI_L  = (V_in - p.R_L * I_L) / p.L
                    dV_C  = (-V_C / R_load) / p.C
                else:
                    #switch OFF: diode ON
                    V_d   = p.V_diode
                    dI_L  = (V_in - p.R_L * I_L - V_C - V_d) / p.L
                    dV_C  = (I_L - V_C / R_load) / p.C
                I_L  = max(I_L + dI_L * dt_sim, 0.0)
                V_C  += dV_C * dt_sim
                V_out = V_C + p.R_C * (I_L if not sw_on else 0.0)
                t_arr[step]    = t
                V_out_arr[step] = V_out
                I_L_arr[step]  = I_L
                V_in_arr[step]  = V_in
                D_arr[step]    = D
                t         += dt_sim
                t_period  += dt_sim
                step      += 1
        #trim to actual steps computed
        sl = slice(0, step)
        eta = np.where(
            V_in_arr[sl] * I_L_arr[sl] > 1e-3,
            V_out_arr[sl] ** 2 / R_load / np.maximum(V_in_arr[sl] * I_L_arr[sl], 1e-3),
            0.0
        )
        return {
            "t":          t_arr[sl],
            "V_out":      V_out_arr[sl],
            "I_L":        I_L_arr[sl],
            "V_in":       V_in_arr[sl],
            "D":          D_arr[sl],
            "efficiency": np.clip(eta, 0.0, 1.0),
        }

#sesign helper
def design_boost(V_in: float, V_out: float, P_out: float,
                 f_sw: float = 20e3,
                 ripple_I_pct: float = 0.30,
                 ripple_V_pct: float = 0.01) -> dict:
    """Minimum inductor and capacitor values for given peak-to-peak ripple specs"""
    D = 1.0 - V_in / V_out
    I_out = P_out / V_out
    I_in  = I_out / (1.0 - D) # = I_L,avg in CCM (lossless approximation)
    #peak-to-peak ripple magnitudes (DC quantities, no √2 conversion)
    delta_I_L = ripple_I_pct * I_in # A, peak-to-peak inductor current ripple
    delta_V_C = ripple_V_pct * V_out # V, peak-to-peak output voltage ripple
    L_min = V_in * D / (delta_I_L * f_sw)
    C_min = I_out * D / (delta_V_C * f_sw)
    return {
        "D": D,
        "L_min_mH": L_min * 1e3,
        "C_min_uF": C_min * 1e6,
        "I_in_A":   I_in,
        "I_out_A":  I_out,
    }
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(__file__))
    from pv_model import PVPanel as _PVPanel
    _panel = _PVPanel()
    V_mpp_nom, I_mpp_nom, P_mpp_nom = _panel.mpp(G=1000.0, T_cell=35.0)
    print(f"Operating point (G=1000 W/m², T=35°C): V_mpp={V_mpp_nom:.2f}V  "
          f"I_mpp={I_mpp_nom:.3f}A  P_mpp={P_mpp_nom:.1f}W")
    bc = BoostConverter()
    D  = bc.duty_cycle(V_in=V_mpp_nom, V_out=400)
    print(f"Duty cycle D = {D:.4f} ({D*100:.1f}%)")
    print(f"Ripple ΔI_L = {bc.inductor_ripple(V_mpp_nom, D):.3f} A")
    print(f"Ripple ΔV_out = {bc.capacitor_ripple(I_mpp_nom*(1-D), D):.3f} V")
    print(f"Efficiency ≈ {bc.efficiency(V_mpp_nom, I_mpp_nom, 400)*100:.2f}%")
    spec = design_boost(V_mpp_nom, 400, P_mpp_nom)
    print("\nMinimum component values (constant-power locus):")
    for k, v in spec.items():
        print(f"  {k:15s} = {v:.4f}")