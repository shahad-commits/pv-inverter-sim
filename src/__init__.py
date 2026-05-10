"""solar PV Grid-Tied Inverter Simulation Package
modules:
pv_model — Single-diode PV panel model (I-V curves, MPP)
boost_converter — DC-DC boost converter (average + switched model)
mppt — MPPT algorithms (P&O, Incremental Conductance)
inverter — H-bridge PWM inverter with LC output filter
power_quality — FFT-based THD and IEEE 1547 compliance analysis
simulation — Top-level simulation runner
plots — Publication-quality figure generation"""

from .pv_model        import PVPanel, PVPanelParams, characterise
from .boost_converter import BoostConverter, BoostParams, design_boost
from .mppt            import (PerturbObserve, IncrementalConductance,
                               benchmark_mppt, step_profile, cloudy_profile)
from .inverter        import Inverter, InverterParams, design_lc_filter
from .power_quality   import PowerQualityAnalyser, print_harmonic_table, print_ieee1547
from .simulation      import run_full_system, run_mppt_benchmark, run_power_quality

__version__ = "1.0.0"
__author__  = "Solar Inverter Simulation Project"