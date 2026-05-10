"""
Unit tests for the DC-DC boost converter model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from boost_converter import BoostConverter, BoostParams, design_boost


class TestBoostConverter:

    def setup_method(self):
        self.bc = BoostConverter()

    def test_duty_cycle_range(self):
        """Duty cycle must be in (0, 1)"""
        D = self.bc.duty_cycle(V_in=30.5, V_out=400.0)
        assert 0 < D < 1

    def test_duty_cycle_ideal(self):
        """For ideal converter D = 1 - V_in/V_out"""
        # Use very low resistances to approach ideal case
        params = BoostParams(R_L=1e-6, R_sw=1e-6, V_diode=1e-6)
        bc_ideal = BoostConverter(params)
        D = bc_ideal.duty_cycle(V_in=50.0, V_out=200.0)
        D_ideal = 1 - 50.0 / 200.0
        assert abs(D - D_ideal) < 0.05

    def test_efficiency_below_unity(self):
        """Efficiency must always be ≤ 1"""
        for V_in in [20, 30, 40]:
            eta = self.bc.efficiency(V_in=V_in, I_in=8.0, V_out=400.0)
            assert 0 < eta <= 1.0

    def test_efficiency_high_at_nominal(self):
        """Efficiency at nominal operating point should be > 90%"""
        eta = self.bc.efficiency(V_in=30.5, I_in=8.19, V_out=400.0)
        assert eta > 0.90, f"Efficiency {eta*100:.1f}% is below 90%"

    def test_inductor_ripple_positive(self):
        """Current ripple must be positive"""
        D = self.bc.duty_cycle(30.5, 400)
        ripple = self.bc.inductor_ripple(30.5, D)
        assert ripple > 0

    def test_capacitor_ripple_positive(self):
        D = self.bc.duty_cycle(30.5, 400)
        ripple = self.bc.capacitor_ripple(0.6, D)
        assert ripple > 0

    def test_ccm_boundary(self):
        """CCM boundary current should be positive and finite"""
        I_ccm = self.bc.ccm_boundary_current(30.5, 400)
        assert 0 < I_ccm < 5.0

    def test_small_signal_model_dimensions(self):
        """State-space matrices must have correct dimensions"""
        A, B, C, E = self.bc.small_signal_model(30.5, 0.92, 8.0, 400)
        assert A.shape == (2, 2)
        assert B.shape == (2, 2)
        assert C.shape == (1, 2)
        assert E.shape == (1, 2)

    def test_small_signal_model_stable(self):
        """A matrix eigenvalues must have negative real parts (stable)"""
        A, _, _, _ = self.bc.small_signal_model(30.5, 0.92, 8.0, 400)
        eigenvalues = np.linalg.eigvals(A)
        assert np.all(eigenvalues.real < 0), "System must be stable"

    def test_design_boost_reasonable_values(self):
        spec = design_boost(V_in=30.5, V_out=400, P_out=250)
        assert spec["L_min_mH"] > 0
        assert spec["C_min_uF"] > 0
        assert 0 < spec["D"] < 1
        assert spec["I_in_A"] > spec["I_out_A"]  # boost increases current on input

    def test_simulate_output_voltage(self):
        """Switched simulation output voltage should be near V_out target"""
        n = 50
        V_in_profile = np.full(n, 30.5)
        D_profile    = np.full(n, self.bc.duty_cycle(30.5, 400))
        result = self.bc.simulate(V_in_profile, D_profile, R_load=1600)
        # Steady-state: last 10 periods
        V_ss = np.mean(result["V_out"][-len(result["V_out"])//10:])
        # Allow wide tolerance since simple Euler simulation
        assert 200 < V_ss < 600, f"V_out_ss = {V_ss:.1f} V seems unreasonable"

    @pytest.mark.parametrize("V_in", [20, 25, 30, 35, 40])
    def test_duty_cycle_increases_with_step_up(self, V_in):
        """Higher step-up ratio requires higher D"""
        D_400 = self.bc.duty_cycle(V_in=V_in, V_out=400)
        D_200 = self.bc.duty_cycle(V_in=V_in, V_out=200)
        assert D_400 > D_200, f"D(400V) should > D(200V) for V_in={V_in}"