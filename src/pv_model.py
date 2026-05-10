import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Tuple

#panel dataclass, STC-rated parameter
@dataclass
class PVPanelParams:
    """electrical parameters for a single PV panel at STC"""
    #STC ratings
    P_mp_stc: float = 250.0 # W, rated maximum power
    V_mp_stc: float = 30.5 # V, voltage at MPP
    I_mp_stc: float = 8.19 # A, current at MPP
    V_oc_stc: float = 37.8 # V, open-circuit voltage
    I_sc_stc: float = 8.72 # A, short-circuit current
    # temperature coefficients
    alpha_I: float = 0.00053 # A/K, dI_sc/dT
    beta_V: float  = -0.12300 # V/K, dV_oc/dT
    mu_Pmp: float  = -0.43e-2 # /K , relative dP_mp/dT (fraction)
    # cell / diode parameters
    N_s: int   = 60 # series cells per panel
    N_p: int   = 1 # parallel strings
    n:   float = 1.1 # diode ideality factor (fitted)

    # series and shunt resistance, fitted phenomenological parameters.. these are curve-fit artifacts from the Villalva et al. iterative method:
    # they are not independently measurable physical quantities but instead absorb all second-order effects (cell interconnect resistance, bypass
    # diode forward drops, distributed cell mismatch, temperature gradients) into two lumped elements so that the single-diode model reproduces the
    # correct I_sc, V_oc, and P_mp at STC..
    R_s:  float = 0.274 # Ω, phenomenological series resistance (fitted)
    R_sh: float = 415.405 # Ω, phenomenological shunt resistance  (fitted)
    # physical constants
    q: float = 1.602e-19 #C
    k: float = 1.381e-23 #J/K
    T_stc: float = 298.15 #K (25 °C)
    G_stc: float = 1000.0 # W/m²
    # Silicon bandgap energy for I_0(T) correction
    E_g:   float = 1.121 # eV, bandgap of crystalline silicon at 25 °C

# Core model
class PVPanel:
    """single-diode PV panel model with irradiance and temperature dependence
    physics:
      -I_ph scales linearly with G and has a small linear T correction.
      -V_oc irradiance dependence comes from the diode equation directly:
        V_oc = V_T × ln(I_ph/I_0 + 1); as G increases I_ph increases,
        raising V_oc logarithmically
      -V_oc temperature dependence is embedded in I_0(T): I_0 rises
        exponentially with T, which lowers V_oc. beta_V is a datasheet
        parameter but is not used directly in this model, it emerges
        automatically from the bandgap-based I_0(T) formula.
      -I_0 follows the standard 3-parameter temperature model:
        I_0(T) = I_0,STC × (T/T_STC)^3 × exp[q·E_g/(n·k) × (1/T_STC − 1/T)]
      -V_mpp increases with G at low-to-moderate irradiance but may peak
        and decline at very high G due to I²·R_s losses compressing the MPP
        voltage.. for this panel (R_s=0.274 Ω) V_mpp peaks near G≈800 W/m²"""

    def __init__(self, params: PVPanelParams = None):
        self.p = params or PVPanelParams()
        self._cache: dict = {}
        # compute I_0 at STC once (used as reference for temperature scaling)
        self._I_0_stc: float = self._compute_I0_stc()

    def _compute_I0_stc(self) -> float:
        """Compute saturation current at STC from V_oc,STC and I_ph,STC. at open-circuit (I=0), the single-diode equation gives:
            0 = I_ph - I_0*(exp(V_oc/V_T) - 1) - V_oc/R_sh
        Solving for I_0:
            I_0,STC = (I_ph,STC - V_oc,STC/R_sh) / (exp(V_oc,STC/V_T,STC) - 1)
        the R_sh term (≈1% of I_ph at V_oc) must be included for consistency with the I-V equation used in current(); omitting it introduces a ~1%
        systematic error in I_0 that propagates to all V and power calculations"""
        p = self.p
        V_T_stc  = p.n * p.N_s * p.k * p.T_stc / p.q
        I_ph_stc = p.I_sc_stc * p.N_p   # at STC: G=G_stc, dT=0
        I_sh_stc = p.V_oc_stc / p.R_sh  # shunt current at V_oc
        I_0_stc  = (I_ph_stc - I_sh_stc) / (np.exp(p.V_oc_stc / V_T_stc) - 1.0)
        return I_0_stc

    # thermal voltage
    def _V_T(self, T: float) -> float:
        """Thermal voltage for N_s cells in series [V]"""
        return self.p.n * self.p.N_s * self.p.k * T / self.p.q

    # Irradiance / temperature adjusted parameters
    def _adjusted_params(self, G: float, T: float) -> Tuple[float, float]:
        """return (I_ph, I_0) adjusted for irradiance G [W/m²] and cell
        temperature T [K].."""
        p = self.p
        dT = T - p.T_stc

        # 1-photo-current: linear in G, small linear T correction
        I_ph = (p.I_sc_stc + p.alpha_I * dT) * (G / p.G_stc) * p.N_p

        # 2-saturation current: temperature-dependent via bandgap model
        #    I_0(T) = I_0,STC × (T/T_STC)^3 × exp[q·E_g/(n·k) × (1/T_STC − 1/T)]
        q_over_nk = p.q / (p.n * p.k)
        I_0 = (self._I_0_stc
               * (T / p.T_stc) ** 3
               * np.exp(q_over_nk * p.E_g * (1.0 / p.T_stc - 1.0 / T)))

        # note: V_oc irradiance dependence is handled by the diode equation itself: V_oc = V_T × ln(I_ph/I_0 + 1), which increases
        # logarithmically with G (via I_ph ∝ G)

        return I_ph, I_0

    # panel current at given terminal voltage
    def current(self, V: float, G: float = 1000.0, T_cell: float = 25.0) -> float:
        """solve I = f(V) for a single (V, G, T_cell) point via Brent's method"""
        if T_cell <= -273.15:
            raise ValueError(f"T_cell must be > -273.15°C, got {T_cell:.2f}°C")
        T = T_cell + 273.15
        I_ph, I_0 = self._adjusted_params(G, T)
        V_T = self._V_T(T)
        p = self.p
        def residual(I):
            exponent = np.clip((V + I * p.R_s) / V_T, -500, 500)
            return I - I_ph + I_0 * (np.exp(exponent) - 1) + (V + I * p.R_s) / p.R_sh

        # bracket: I in [-I_ph*0.02, I_ph*1.05], lower bound is slightly negative to handle near-V_oc voltages where
        # I_sol ≈ 0 and residual(-0.01) can be slightly positive due to R_sh current, causing brentq to fail with a same-sign bracket error.
        # using -I_ph*0.02 as lower bound ensures residual is always negative there:
        # at I = -I_ph*0.02: residual ≈ -I_ph*0.02 - I_ph + ... ≈ -1.02*I_ph < 0 ✓
        try:
            I_sol = brentq(residual, -I_ph * 0.02, I_ph * 1.05, xtol=1e-8, maxiter=200)
        except ValueError:
            I_sol = 0.0

        return max(I_sol, 0.0)

    # Full I-V and P-V curves
    def iv_curve(
        self,
        G: float = 1000.0,
        T_cell: float = 25.0,
        n_points: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ compute the full I-V and P-V characteristic curves"""
        T = T_cell + 273.15
        I_ph, I_0 = self._adjusted_params(G, T)
        p = self.p

        # estimate V_oc for this condition
        V_T = self._V_T(T)
        V_oc_est = V_T * np.log(I_ph / I_0 + 1)
        V_oc_est = min(V_oc_est, p.V_oc_stc * 1.2)

        V = np.linspace(0, V_oc_est * 0.999, n_points)
        I = np.array([self.current(v, G, T_cell) for v in V])
        P = V * I

        return V, I, P

    # Maximum power point
    def mpp(
        self,
        G: float = 1000.0,
        T_cell: float = 25.0,
    ) -> Tuple[float, float, float]:
        """return (V_mpp, I_mpp, P_mpp), the maximum power operating point
        Uses the P-V curve and golden-section search for accuracy"""
        from scipy.optimize import minimize_scalar

        T = T_cell + 273.15
        I_ph, I_0 = self._adjusted_params(G, T)
        V_T = self._V_T(T)
        V_oc_est = min(V_T * np.log(I_ph / I_0 + 1), self.p.V_oc_stc * 1.2)

        result = minimize_scalar(
            lambda v: -v * self.current(v, G, T_cell),
            bounds=(0.1, V_oc_est * 0.999),
            method="bounded",
        )
        V_mpp = result.x
        I_mpp = self.current(V_mpp, G, T_cell)
        P_mpp = V_mpp * I_mpp

        return V_mpp, I_mpp, P_mpp

    # convenience: panel string / array scaling
    def array_mpp(
        self,
        G: float = 1000.0,
        T_cell: float = 25.0,
        n_series: int = 1,
        n_parallel: int = 1,
    ) -> Tuple[float, float, float]:
        """Scale MPP to an n_series × n_parallel array"""
        V_mpp, I_mpp, P_mpp = self.mpp(G, T_cell)
        return V_mpp * n_series, I_mpp * n_parallel, P_mpp * n_series * n_parallel

# module-level helper..quick IV characterisatio

def characterise(G_values=(200, 400, 600, 800, 1000),
                 T_cell: float = 25.0) -> dict:
    """return a dict of IV curve data keyed by irradiance level.
    Useful for bulk generation of comparison plots"""
    panel = PVPanel()
    results = {}
    for G in G_values:
        V, I, P = panel.iv_curve(G=G, T_cell=T_cell)
        V_mpp, I_mpp, P_mpp = panel.mpp(G=G, T_cell=T_cell)
        results[G] = {
            "V": V, "I": I, "P": P,
            "V_mpp": V_mpp, "I_mpp": I_mpp, "P_mpp": P_mpp,
        }
    return results


if __name__ == "__main__":
    panel = PVPanel()
    print("I-V MPP at T_cell = 25°C (STC temperature):")
    for G in [400, 700, 1000]:
        Vm, Im, Pm = panel.mpp(G=G, T_cell=25)
        print(f"G={G:4d} W/m²  →  V_mpp={Vm:.2f} V  I_mpp={Im:.3f} A  P_mpp={Pm:.2f} W")
    # note: V_mpp peaks near G≈800 W/m² due to I²·R_s compression; slight
    # rollback at G=1000 vs G=700 is physical, not a model defect.
    print("\nNominal operating point (G=1000, T_cell=35°C):")
    Vm, Im, Pm = panel.mpp(G=1000, T_cell=35)
    print(f"  V_mpp={Vm:.2f} V  I_mpp={Im:.3f} A  P_mpp={Pm:.2f} W")