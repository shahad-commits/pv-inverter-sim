"""
Unit tests for the PV panel single-diode model
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from pv_model import PVPanel, PVPanelParams, characterise


class TestPVPanel:

    def setup_method(self):
        self.panel = PVPanel()

    def test_short_circuit_current(self):
        """I_sc at STC should be close to rated value"""
        I_sc = self.panel.current(V=0.0, G=1000.0, T_cell=25.0)
        assert abs(I_sc - self.panel.p.I_sc_stc) < 0.5, (
            f"I_sc = {I_sc:.3f} A, expected ≈ {self.panel.p.I_sc_stc} A"
        )

    def test_open_circuit_voltage(self):
        """V_oc at STC should be close to rated value"""
        V, I, P = self.panel.iv_curve(G=1000.0, T_cell=25.0)
        # V_oc is where I crosses zero — find last non-negative I
        idx = np.where(I >= 0)[0]
        V_oc_est = V[idx[-1]] if len(idx) else 0.0
        assert abs(V_oc_est - self.panel.p.V_oc_stc) < 3.0, (
            f"V_oc ≈ {V_oc_est:.2f} V, expected ≈ {self.panel.p.V_oc_stc} V"
        )

    def test_mpp_at_stc(self):
        """MPP power at STC should be within 5% of rated"""
        Vm, Im, Pm = self.panel.mpp(G=1000.0, T_cell=25.0)
        assert abs(Pm - self.panel.p.P_mp_stc) / self.panel.p.P_mp_stc < 0.05, (
            f"P_mpp = {Pm:.2f} W, expected ≈ {self.panel.p.P_mp_stc} W"
        )

    def test_mpp_monotone_in_irradiance(self):
        """P_mpp should increase monotonically with irradiance"""
        G_vals = [200, 400, 600, 800, 1000]
        P_vals = [self.panel.mpp(G=G)[2] for G in G_vals]
        for i in range(len(P_vals) - 1):
            assert P_vals[i] < P_vals[i + 1], (
                f"P_mpp not monotone at G={G_vals[i]}→{G_vals[i+1]}"
            )

    def test_mpp_voltage_decreases_with_temperature(self):
        """V_mpp should decrease with rising cell temperature"""
        Vm_25, _, _ = self.panel.mpp(G=1000.0, T_cell=25.0)
        Vm_50, _, _ = self.panel.mpp(G=1000.0, T_cell=50.0)
        assert Vm_50 < Vm_25, "V_mpp should decrease with temperature"

    def test_iv_curve_shape(self):
        """I-V curve should be monotonically decreasing"""
        V, I, P = self.panel.iv_curve(G=1000.0, T_cell=25.0)
        assert np.all(np.diff(I) <= 0.1), "I-V curve should be non-increasing"

    def test_zero_irradiance_gives_zero_power(self):
        """At G=0 the panel should produce no power"""
        _, _, Pm = self.panel.mpp(G=0.001, T_cell=25.0)
        assert Pm < 0.1, f"Expected near-zero power at G≈0, got {Pm:.4f} W"

    def test_current_non_negative(self):
        """Panel current should never be negative"""
        V, I, _ = self.panel.iv_curve(G=800.0, T_cell=35.0)
        assert np.all(I >= -1e-6), "Current should be non-negative"

    def test_characterise_returns_five_levels(self):
        data = characterise(G_values=[200, 400, 600, 800, 1000])
        assert len(data) == 5
        for G, d in data.items():
            assert "V" in d and "I" in d and "P" in d

    def test_array_scaling(self):
        """2S1P array should have double V_mpp and same I_mpp as single panel"""
        Vm1, Im1, Pm1 = self.panel.mpp()
        Vm2, Im2, Pm2 = self.panel.array_mpp(n_series=2, n_parallel=1)
        assert abs(Vm2 - 2 * Vm1) < 1.0
        assert abs(Im2 - Im1) < 0.1

    @pytest.mark.parametrize("G", [100, 300, 500, 700, 900])
    def test_mpp_physically_reasonable(self, G):
        Vm, Im, Pm = self.panel.mpp(G=G, T_cell=25.0)
        assert 5 < Vm < 45, f"V_mpp={Vm:.2f} unreasonable at G={G}"
        assert 0 < Im < 12,  f"I_mpp={Im:.3f} unreasonable at G={G}"
        assert Pm > 0,        f"P_mpp={Pm:.2f} must be positive"