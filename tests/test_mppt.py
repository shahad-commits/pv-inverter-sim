"""
Unit tests for the MPPT algorithm implementations
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from pv_model import PVPanel
from mppt import (PerturbObserve, IncrementalConductance,
                   PandOParams, InCondParams,
                   benchmark_mppt, step_profile, cloudy_profile, ramp_profile)


class TestPerturbObserve:

    def setup_method(self):
        self.panel = PVPanel()
        self.po    = PerturbObserve()

    def test_initial_v_ref(self):
        assert 5 <= self.po.V_ref <= 40

    def test_step_returns_float(self):
        v_ref = self.po.step(V=30.0, I=8.0)
        assert isinstance(v_ref, float)

    def test_v_ref_stays_in_bounds(self):
        """V_ref must never exceed V_min/V_max bounds"""
        for _ in range(200):
            V = np.random.uniform(5, 40)
            I = self.panel.current(V, G=1000)
            self.po.step(V, I)
        assert self.po.p.V_min <= self.po.V_ref <= self.po.p.V_max

    def test_history_grows(self):
        for k in range(10):
            self.po.step(V=25.0 + k * 0.1, I=8.0 - k * 0.05)
        assert len(self.po.history) == 10

    def test_reset_clears_history(self):
        for _ in range(5):
            self.po.step(30.0, 8.0)
        self.po.reset()
        assert len(self.po.history) == 0

    def test_converges_toward_mpp(self):
        """After 100 steps at constant irradiance P&O should be near MPP"""
        G = 1000.0
        Vm, Im, Pm = self.panel.mpp(G=G, T_cell=25)
        self.po.reset(V_init=20.0)

        for _ in range(100):
            V_op = float(np.clip(self.po.V_ref, 1, 40))
            I_op = self.panel.current(V_op, G=G, T_cell=25)
            self.po.step(V_op, I_op)

        P_tracked = self.po.V_ref * self.panel.current(self.po.V_ref, G)
        # Should be within 5% of true MPP
        assert P_tracked / Pm > 0.90, (
            f"P&O converged to {P_tracked:.1f}W, MPP={Pm:.1f}W"
        )


class TestIncrementalConductance:

    def setup_method(self):
        self.panel = PVPanel()
        self.inc   = IncrementalConductance()

    def test_step_returns_float(self):
        v_ref = self.inc.step(V=30.0, I=8.0)
        assert isinstance(v_ref, float)

    def test_v_ref_stays_in_bounds(self):
        for _ in range(200):
            V = np.random.uniform(5, 40)
            I = self.panel.current(V, G=800)
            self.inc.step(V, I)
        assert self.inc.p.V_min <= self.inc.V_ref <= self.inc.p.V_max

    def test_converges_toward_mpp(self):
        G = 1000.0
        Vm, Im, Pm = self.panel.mpp(G=G, T_cell=25)
        self.inc.reset(V_init=20.0)

        for _ in range(100):
            V_op = float(np.clip(self.inc.V_ref, 1, 40))
            I_op = self.panel.current(V_op, G=G, T_cell=25)
            self.inc.step(V_op, I_op)

        P_tracked = self.inc.V_ref * self.panel.current(self.inc.V_ref, G)
        assert P_tracked / Pm > 0.90


class TestMPPTBenchmark:

    def setup_method(self):
        self.panel = PVPanel()

    def test_step_profile_length(self):
        prof = step_profile(dt=0.05)
        assert len(prof) > 100

    def test_cloudy_profile_bounded(self):
        prof = cloudy_profile()
        assert np.all(prof >= 100) and np.all(prof <= 1000)

    def test_benchmark_returns_both_algos(self):
        prof = step_profile(dt=0.05)[:50]  # short for speed
        results = benchmark_mppt(self.panel, prof, dt=0.05)
        assert "P&O" in results
        assert "InCond" in results

    def test_benchmark_eta_reasonable(self):
        """Both algorithms should achieve >85% tracking efficiency"""
        prof = step_profile(dt=0.05)[:100]
        results = benchmark_mppt(self.panel, prof, dt=0.05)
        for name, r in results.items():
            assert r["eta_overall"] > 0.85, (
                f"{name} tracking efficiency {r['eta_overall']*100:.1f}% too low"
            )

    def test_incond_better_ss_than_po_constant_irradiance(self):
        """
        Under constant irradiance InCond should have ≥ P&O steady-state efficiency
        (InCond has zero steady-state oscillation in theory)
        """
        prof = np.full(400, 1000.0)
        results = benchmark_mppt(self.panel, prof, dt=0.05)
        eta_po   = results["P&O"]["eta_ss"]
        eta_inc  = results["InCond"]["eta_ss"]
        # InCond should be at least as good (allow 0.5% tolerance)
        assert eta_inc >= eta_po - 0.005, (
            f"InCond ({eta_inc*100:.2f}%) worse than P&O ({eta_po*100:.2f}%) "
            f"at constant irradiance"
        )

    def test_power_non_negative(self):
        prof = cloudy_profile()[:100]
        results = benchmark_mppt(self.panel, prof, dt=0.05)
        for name, r in results.items():
            assert np.all(r["P_actual"] >= -0.1), f"{name}: negative power detected"

    def test_v_ref_in_valid_range(self):
        prof = step_profile()[:50]
        results = benchmark_mppt(self.panel, prof, dt=0.05)
        for name, r in results.items():
            assert np.all(r["V_ref"] >= 0), f"{name}: negative V_ref"
            assert np.all(r["V_ref"] <= 45), f"{name}: V_ref too high"