"""

Shared pytest fixtures for the solar inverter simulation test suite.
"""

import sys
import os

# Ensure the src directory is on the path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
import numpy as np


@pytest.fixture(scope="session")
def panel():
    """Session-scoped PV panel instance"""
    from pv_model import PVPanel
    return PVPanel()


@pytest.fixture(scope="session")
def boost():
    """Session-scoped boost converter instance"""
    from boost_converter import BoostConverter
    return BoostConverter()


@pytest.fixture(scope="session")
def inverter():
    """Session-scoped inverter instance"""
    from inverter import Inverter
    return Inverter()


@pytest.fixture(scope="session")
def pq_analyser():
    """Session-scoped power quality analyser"""
    from power_quality import PowerQualityAnalyser
    return PowerQualityAnalyser(f_grid=50.0, n_harmonics=30)


@pytest.fixture
def synthetic_sine():
    """Factory fixture: returns a pure 50 Hz sine waveform"""
    def _make(f0=50.0, V_peak=325.0, dt=1e-5, duration=0.1):
        t = np.arange(0, duration, dt)
        return t, V_peak * np.sin(2 * np.pi * f0 * t), dt
    return _make


@pytest.fixture
def step_irradiance_profile():
    from mppt import step_profile
    return step_profile(dt=0.05)


@pytest.fixture
def cloudy_irradiance_profile():
    from mppt import cloudy_profile
    return cloudy_profile(dt=0.05)