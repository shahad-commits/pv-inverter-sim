"""
Unit tests for the H-bridge inverter model and power quality analyser.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from inverter       import Inverter, InverterParams, SPWMModulator, design_lc_filter
from power_quality  import PowerQualityAnalyser, HarmonicSpectrum



# Inverter tests


class TestSPWMModulator:

    def setup_method(self):
        self.params = InverterParams()
        self.mod    = SPWMModulator(self.params)

    def test_modulation_index_in_range(self):
        assert 0 < self.mod.M_a <= 1.0

    def test_reference_bounded(self):
        t_vals = np.linspace(0, 0.1, 1000)
        refs   = [self.mod.reference(t) for t in t_vals]
        assert max(refs) <= 1.0 + 1e-9
        assert min(refs) >= -1.0 - 1e-9

    def test_carrier_bounded(self):
        t_vals = np.linspace(0, 0.01, 10000)
        cars   = [self.mod.carrier(t) for t in t_vals]
        assert max(cars) <= 1.0 + 1e-9
        assert min(cars) >= -1.0 - 1e-9

    def test_v_ab_three_levels(self):
        """Unipolar SPWM: V_AB ∈ {-V_dc, 0, +V_dc}"""
        t_vals = np.linspace(0, 0.04, 4000)
        V_dc   = self.params.V_dc
        levels = set()
        for t in t_vals:
            v = self.mod.v_ab(t)
            levels.add(round(v / V_dc) * V_dc)
        assert -V_dc in levels
        assert 0 in levels
        assert V_dc in levels

    def test_switch_states_binary(self):
        for t in np.linspace(0, 0.02, 200):
            Sa, Sb = self.mod.switch_states(t)
            assert Sa in (0, 1)
            assert Sb in (0, 1)


class TestInverter:

    def setup_method(self):
        self.inv = Inverter()

    def test_f_corner_positive(self):
        assert self.inv.f_corner > 0

    def test_q_factor_positive(self):
        assert self.inv.Q_factor > 0

    def test_simulate_returns_expected_keys(self):
        res = self.inv.simulate(t_end=0.02, dt=1e-5)
        for key in ["t", "V_AB", "V_out", "I_L", "I_out", "P_out", "V_rms"]:
            assert key in res, f"Missing key: {key}"

    def test_simulation_array_lengths_match(self):
        res = self.inv.simulate(t_end=0.02, dt=1e-5)
        n = len(res["t"])
        for key in ["V_AB", "V_out", "I_L", "I_out"]:
            assert len(res[key]) == n

    def test_v_rms_near_target(self):
        """Output RMS should be within 20% of V_ac_peak/√2"""
        res   = self.inv.simulate(t_end=0.06, dt=1e-5)
        V_exp = self.inv.p.V_ac_peak / np.sqrt(2)
        assert abs(res["V_rms"] - V_exp) / V_exp < 0.20, (
            f"V_rms = {res['V_rms']:.1f} V, expected ≈ {V_exp:.1f} V"
        )

    def test_power_output_positive(self):
        res = self.inv.simulate(t_end=0.04, dt=1e-5)
        assert res["P_out"] >= 0

    def test_bode_correct_shape(self):
        f, mag, phase = self.inv.bode()
        assert len(f) == len(mag) == len(phase)
        assert f[0] > 0

    def test_bode_attenuation_at_high_freq(self):
        """Filter should attenuate at f_sw"""
        f, mag, _ = self.inv.bode(f_max=1e5)
        idx_sw = np.argmin(np.abs(f - self.inv.p.f_sw))
        assert mag[idx_sw] < -20, (
            f"Filter attenuation at f_sw: {mag[idx_sw]:.1f} dB (expected < -20 dB)"
        )

    def test_filter_passes_fundamental(self):
        """Filter should have near 0 dB at grid frequency"""
        f, mag, _ = self.inv.bode()
        idx_f1 = np.argmin(np.abs(f - self.inv.p.f_grid))
        assert mag[idx_f1] > -3.0, (
            f"Filter attenuates fundamental by {-mag[idx_f1]:.1f} dB (should be < 3 dB)"
        )


class TestDesignLCFilter:

    def test_returns_positive_values(self):
        filt = design_lc_filter()
        assert filt.L_f > 0
        assert filt.C_f > 0
        assert filt.f_c > 0

    def test_corner_frequency_relation(self):
        """f_c should equal 1/(2π√LC)"""
        filt = design_lc_filter()
        f_c_check = 1.0 / (2 * np.pi * np.sqrt(filt.L_f * filt.C_f))
        assert abs(f_c_check - filt.f_c) / filt.f_c < 0.05

    def test_corner_below_switching(self):
        filt = design_lc_filter(f_sw=10e3)
        assert filt.f_c < filt.f_sw



# Power quality tests


class TestPowerQualityAnalyser:

    def setup_method(self):
        self.pq = PowerQualityAnalyser(f_grid=50.0, n_harmonics=20)
        dt = 1e-5
        t  = np.arange(0, 0.1, dt)
        # Pure fundamental
        self.V_pure = 230 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
        # With harmonics
        self.V_dist = (230 * np.sqrt(2) * np.sin(2 * np.pi * 50  * t)
                       + 10.0 * np.sin(2 * np.pi * 150 * t)
                       + 5.0  * np.sin(2 * np.pi * 250 * t))
        self.dt = dt

    def test_pure_sine_thd_near_zero(self):
        spec = self.pq.analyse(self.V_pure, self.dt)
        assert spec.THD < 1.0, f"Pure sine THD = {spec.THD:.3f}% (expected < 1%)"

    def test_distorted_thd_detectable(self):
        spec = self.pq.analyse(self.V_dist, self.dt)
        # Known harmonic content: ~√(10² + 5²) / (230√2/√2) ≈ 4.8%
        assert spec.THD > 1.0, f"THD = {spec.THD:.3f}% not detected"

    def test_fundamental_extraction_accurate(self):
        spec = self.pq.analyse(self.V_pure, self.dt)
        expected_rms = 230 * np.sqrt(2) / np.sqrt(2)  # = 230 V
        assert abs(spec.fundamental - expected_rms) / expected_rms < 0.02

    def test_rms_total_correct(self):
        spec = self.pq.analyse(self.V_pure, self.dt)
        rms_np = np.sqrt(np.mean(self.V_pure[-int(0.04/self.dt):] ** 2))
        assert abs(spec.rms_total - rms_np) / rms_np < 0.02

    def test_crest_factor_near_sqrt2(self):
        spec = self.pq.analyse(self.V_pure, self.dt)
        assert abs(spec.crest_factor - np.sqrt(2)) < 0.1

    def test_harmonic_count_correct(self):
        spec = self.pq.analyse(self.V_pure, self.dt)
        assert spec.n_harmonics == 20
        assert len(spec.harmonics) == 20
        assert len(spec.magnitudes) == 20

    def test_ieee1547_pure_sine_compliant(self):
        spec   = self.pq.analyse(self.V_pure, self.dt)
        result = self.pq.ieee1547_check(spec, I_rated=spec.fundamental)
        assert result.compliant, f"Pure sine failed IEEE 1547: {result.violations}"

    def test_sinad_pure_sine_high(self):
        sinad = self.pq.sinad(self.V_pure, self.dt)
        assert sinad > 40, f"SINAD of pure sine = {sinad:.1f} dB (expected > 40 dB)"

    def test_thd_voltage_method(self):
        thd1 = self.pq.analyse(self.V_dist, self.dt).THD
        thd2 = self.pq.thd_voltage(self.V_dist, self.dt)
        assert abs(thd1 - thd2) < 0.1