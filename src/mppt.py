"""maximum Power Point Tracking (MPPT) algorithms for PV systems..
two algorithms are implemented and share a common base class interface so they can be benchmarked under identical irradiance profiles:
1-Perturb & Observe (P&O): simple, robust, widely deployed Steady-state oscillation around MPP introduces a small power loss
2-Incremental Conductance (InCond): exploits the condition dP/dV = 0 at MPP expressed as:
dI/dV = -I/V (at MPP)
dI/dV > -I/V (left of MPP)
dI/dV < -I/V (right of MPP)
-reduces steady-state oscillation at constant irradiance
-both algorithms output a reference voltage V_ref that is fed to the boost converter's voltage controller...
-tracking Efficiency: η_MPPT = (1/T) ∫ P_actual dt  /  (1/T) ∫ P_mpp dt """

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

# Base class
class MPPTBase(ABC):
    """abstract base for MPPT controllers"""
    def __init__(self, V_init: float = 25.0):
        self.V_ref = V_init # current reference voltage [V]
        self._history: List[dict] = []
    @abstractmethod
    def step(self, V: float, I: float, **kwargs) -> float:
        """process a new V, I measurement and return updated V_ref"""

    def reset(self, V_init: float = 25.0):
        self.V_ref = V_init
        self._history.clear()

    @property
    def history(self) -> List[dict]:
        return self._history

    def tracking_efficiency(self, P_mpp_profile: np.ndarray) -> float:
        """compute MPPT tracking efficiency η = ΣP_actual / ΣP_mpp ∈ [0, 1], clipped to [0, 1] — transient overshoots (where P_actual momentarily exceeds P_mpp due to discretisation) are physically meaningless and
        would inflate the metric above 1.0 without clipping"""
        if not self._history:
            return 0.0
        P_actual = np.array([h["P"] for h in self._history])
        n = min(len(P_actual), len(P_mpp_profile))
        denom = np.sum(np.abs(P_mpp_profile[:n]))
        if denom < 1e-6:
            return 0.0
        eta = float(np.sum(P_actual[:n]) / denom)
        return float(np.clip(eta, 0.0, 1.0))

#1-perturb & observe
@dataclass
class PandOParams:
    delta_V: float = 0.5 #V perturbation step size
    V_min: float = 5.0 #V minimum reference voltage
    #V_max=40V is sufficient for T_cell ≥ 0°C (V_oc(0°C) ≈ 42V; V_mpp ≈ 34V)
    #for sub-zero cell temperatures, increase V_max to avoid limiting the MPPT, search region: e.g. V_max=45V covers down to T_cell ≈ -10°C
    V_max: float = 40.0 #V maximum reference voltage
    sample_dt: float = 0.05 #s algorithm update interval


class PerturbObserve(MPPTBase):
    """classic Perturb & Observe MPPT..the step size delta_V governs the trade-off between tracking speed and steady-state oscillation power loss """
    def __init__(self, params: PandOParams = None, V_init: float = 25.0):
        super().__init__(V_init)
        self.p = params or PandOParams()
        self._V_prev = V_init
        self._P_prev = 0.0
    def step(self, V: float, I: float, **kwargs) -> float:
        P = V * I
        dP = P  - self._P_prev
        dV = V  - self._V_prev
        if abs(dV) < 1e-6:
            #no perturbation yet applied, just perturb upward
            self.V_ref += self.p.delta_V
        elif dP > 0:
            #power increased, keep moving in same direction
            self.V_ref += self.p.delta_V * np.sign(dV)
        else:
            #power decreased, reverse direction
            self.V_ref -= self.p.delta_V * np.sign(dV)
        self.V_ref = float(np.clip(self.V_ref, self.p.V_min, self.p.V_max))
        self._V_prev = V
        self._P_prev = P
        self._history.append({"V": V, "I": I, "P": P, "V_ref": self.V_ref})
        return self.V_ref
    def reset(self, V_init: float = 25.0):
        super().reset(V_init)
        self._V_prev = V_init
        self._P_prev = 0.0

#2-incremental Conductance
@dataclass
class InCondParams:
    delta_V:float = 0.3 #V voltage step
    V_min: float = 5.0 #V
    #V_max=40V ,same cold-temperature caveat as PandOParams: raise to 45V, if operating below T_cell=0°C to avoid artificially capping the search..
    V_max: float = 40.0 #V
    tol: float = 0.01 #A/V conductance equality tolerance
    sample_dt: float = 0.05 #s


class IncrementalConductance(MPPTBase):
    """incremental Conductance MPPT"""
    def __init__(self, params: InCondParams = None, V_init: float = 25.0):
        super().__init__(V_init)
        self.p = params or InCondParams()
        self._V_prev = V_init
        self._I_prev = 0.0

    def step(self, V: float, I: float, **kwargs) -> float:
        P    = V * I
        dV   = V - self._V_prev
        dI   = I - self._I_prev
        if abs(dV) < 1e-9:
            #vertical I-V, irradiance step, use current sign of dI
            if abs(dI) < 1e-9:
                pass  #at MPP, no change
            elif dI > 0:
                self.V_ref += self.p.delta_V
            else:
                self.V_ref -= self.p.delta_V
        else:
            G_inc = dI / dV #incremental conductance
            G_ins = -I / V if V > 1e-6 else 0.0 #instantaneous
            diff = G_inc - G_ins
            if abs(diff) <= self.p.tol:
                pass  #at MPP
            elif diff > 0:
                self.V_ref += self.p.delta_V
            else:
                self.V_ref -= self.p.delta_V

        self.V_ref = float(np.clip(self.V_ref, self.p.V_min, self.p.V_max))
        self._V_prev = V
        self._I_prev = I
        self._history.append({"V": V, "I": I, "P": P, "V_ref": self.V_ref})
        return self.V_ref
    def reset(self, V_init: float = 25.0):
        super().reset(V_init)
        self._V_prev = V_init
        self._I_prev = 0.0

#benchmark harness
def benchmark_mppt(
    panel,
    irradiance_profile: np.ndarray,
    T_cell: float = 35.0,
    dt: float = 0.05,
    algorithms: dict = None,
) -> dict:
    """run one or more MPPT algorithms over an irradiance time profile and return per-algorithm performance metrics"""
    if algorithms is None:
        algorithms = {
            "P&O":   PerturbObserve(),
            "InCond": IncrementalConductance(),
        }

    n = len(irradiance_profile)
    t = np.arange(n) * dt

    #pre-compute true MPP at each step
    P_mpp_arr = np.zeros(n)
    V_mpp_arr = np.zeros(n)
    for k, G in enumerate(irradiance_profile):
        Vm, Im, Pm = panel.mpp(G=G, T_cell=T_cell)
        P_mpp_arr[k] = Pm
        V_mpp_arr[k] = Vm
    results = {}
    for name, algo in algorithms.items():
        algo.reset(V_init=V_mpp_arr[0])
        V_ref_arr    = np.zeros(n)
        P_actual_arr = np.zeros(n)
        for k, G in enumerate(irradiance_profile):
            #the panel operates at V_ref (assume perfect voltage control)..record V_op BEFORE calling step() so V_ref_arr[k] matches the operating voltage that produced P_actual[k] (not the next-step target)
            V_op = float(np.clip(algo.V_ref, 1.0, 40.0))
            I_op = panel.current(V=V_op, G=G, T_cell=T_cell)
            V_ref_arr[k] = V_op #pre-step: voltage that gave P_actual[k]
            P_actual_arr[k] = V_op * I_op
            algo.step(V_op, I_op) #updates V_ref for next iteration

        eta = algo.tracking_efficiency(P_mpp_arr)
        #steady-state efficiency: exclude transient settling after each irradiance step.. for a step profile the MPPT typically settles
        # within ~10 algorithm steps (0.5 s at dt=0.05 s)...we therefore exclude the first 10 samples of each steady segment, for a cloudy profile there is no true steady state, we fall back
        #to using the last 50% of the run
        settle_steps = 10 #samples to skip after each irradiance change
        ss_mask = np.ones(n, dtype=bool)
        #mark transient samples: detect irradiance steps
        G_arr = np.asarray(irradiance_profile)
        step_indices = np.where(np.abs(np.diff(G_arr)) > 1.0)[0] + 1
        for si in step_indices:
            hi = min(n, si + settle_steps)
            ss_mask[si:hi] = False
        #also exclude the very first settle_steps samples (startup transient)
        ss_mask[:settle_steps] = False

        denom = np.sum(P_mpp_arr[ss_mask])
        if denom < 1e-6:
            #fallback: last 50 % of run
            ss_start = n // 2
            denom = max(np.sum(P_mpp_arr[ss_start:]), 1e-6)
            eta_ss = float(np.sum(P_actual_arr[ss_start:]) / denom)
        else:
            eta_ss = float(np.sum(P_actual_arr[ss_mask]) / denom)

        results[name] = {
            "t": t,
            "V_ref": V_ref_arr,
            "P_actual": P_actual_arr,
            "P_mpp": P_mpp_arr,
            "V_mpp": V_mpp_arr,
            "eta_overall": eta,
            "eta_ss": float(eta_ss),
        }

    return results

#standard irradiance profiles
def ramp_profile(t_end: float = 10.0, dt: float = 0.05,
                 G_start: float = 200.0, G_end: float = 1000.0) -> np.ndarray:
    """linear ramp from G_start to G_end"""
    n = int(t_end / dt)
    return np.linspace(G_start, G_end, n)


def step_profile(dt: float = 0.05) -> np.ndarray:
    """stepped irradiance profile covering standard test cases: 200 → 600 → 1000 → 800 → 400 W/m²"""
    segments = [
        (200,  4.0), # low irradiance
        (600,  3.0), # mid step up
        (1000, 4.0), # full sun
        (800,  3.0), # partial cloud
        (400,  4.0), # heavy cloud
        (1000, 3.0), # recovery
    ]
    parts = []
    for G, duration in segments:
        n = int(duration / dt)
        parts.append(np.full(n, G))
    return np.concatenate(parts)


def cloudy_profile(dt: float = 0.05, seed: int = 42) -> np.ndarray:
    """realistic partial-cloud profile: slow baseline + fast Gaussian transients"""
    rng  = np.random.default_rng(seed)
    n    = int(30.0 / dt)   # 30 s
    t    = np.linspace(0, 30, n)
    base = 700.0 + 200.0 * np.sin(2 * np.pi * t / 30.0)
    #cloud shadows: 3–5 random dips
    clouds = np.zeros(n)
    for _ in range(5):
        t0     = rng.uniform(5, 25)
        width  = rng.uniform(0.5, 2.0)
        depth  = rng.uniform(100, 500)
        clouds -= depth * np.exp(-((t - t0) ** 2) / (2 * width ** 2))
    G = np.clip(base + clouds, 100, 1000)
    return G

if __name__ == "__main__":
    from pv_model import PVPanel
    panel = PVPanel()
    profile = step_profile()
    results = benchmark_mppt(panel, profile)
    for name, r in results.items():
        print(f"{name:8s}: η_overall={r['eta_overall']*100:.2f}%  "
              f"η_ss={r['eta_ss']*100:.2f}%")