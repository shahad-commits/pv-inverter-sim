"""FFT-based power quality analysis of inverter output waveforms...
metrics computed:
-total Harmonic Distortion (THD) of voltage and current
-individual harmonic magnitudes and phases
-power factor (displacement and true)
-crest factor
-SINAD (signal-to-noise-and-distortion)
-IEEE 1547-2018 compliance check
THD Definition: THD_V = sqrt(Σ_{n=2}^{N} V_n²) / V_1  × 100 %
where V_n is the RMS voltage of the nth harmonic and V_1 is the
fundamental.. odd harmonics dominate in symmetric PWM inverter outputs
IEEE 1547-2018 current harmonic limits:
Odd harmonics (3rd–9th): < 4.0 % of rated current
Odd harmonics (11th–15th): < 2.0 %
Odd harmonics (17th–21st): < 1.5 %
Odd harmonics (23rd–33rd): < 0.6 %
Even harmonics: < 25 % of odd limits
Total (THD): < 5.0 % """

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#data structures
@dataclass
class HarmonicSpectrum:
    """FFT harmonic analysis results"""
    f_fund: float # Hz, fundamental frequency
    harmonics: np.ndarray # harmonic orders (1, 2, 3, …, N)
    frequencies: np.ndarray # Hz, harmonic frequencies
    magnitudes: np.ndarray # RMS magnitudes [V or A]
    phases: np.ndarray # degrees
    THD: float # %, total harmonic distortion
    fundamental: float # RMS fundamental component [V or A]
    rms_total: float # RMS of full waveform [V or A]
    crest_factor: float # peak / rms
    n_harmonics:  int # number of harmonics analysed


@dataclass
class IEEE1547Result:
    """IEEE 1547-2018 compliance check results"""
    compliant:         bool
    thd_limit:         float = 5.0          # %
    thd_actual:        float = 0.0          # %
    violations:        List[str] = field(default_factory=list)
    harmonic_limits:   Dict[int, float] = field(default_factory=dict)  # order: limit %
    harmonic_actuals:  Dict[int, float] = field(default_factory=dict)  # order: actual %
#core FFT analysis
class PowerQualityAnalyser:
    """FFT-based power quality analyser for single-phase inverter waveforms """
    # IEEE 1547-2018 current harmonic limits (% of rated current).
    _I_LIMITS: Dict[Tuple[int, int], float] = {
        (2,   9): 4.0, # odd H3–H9 (even get ×0.25 of this)
        (10, 15): 2.0,
        (16, 21): 1.5,
        (22, 33): 0.6,
        (34, 99): 0.3, # was (35,99), gap at H34 now closed
    }
    _THD_LIMIT = 5.0   # %

    def __init__(
        self,
        f_grid: float = 50.0,
        n_harmonics: int = 50,
        window: str = "hann",
    ):
        self.f_grid = f_grid
        self.n_harmonics = n_harmonics
        self.window = window

    # windowing
    def _make_window(self, n: int) -> np.ndarray:
        if self.window == "hann":
            return np.hanning(n)
        elif self.window == "hamming":
            return np.hamming(n)
        elif self.window == "blackman":
            return np.blackman(n)
        else:
            return np.ones(n) # rectangular
    #FFT spectrum extraction
    def _fft_spectrum(
        self, x: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """return (frequencies [Hz], one-sided RMS amplitude spectrum)
        correct processing order:
          1-normalise by sample count:  X = rfft(x * w) / n
          2-convert to one-sided (double non-DC, non-Nyquist bins)
          3-correct for window coherent gain (= mean(window))
          4-peak → RMS: divide by √2"""
        n = len(x)
        win = self._make_window(n)
        coherent_gain = np.mean(win)   # = 0.5 for Hann, 1.0 for rectangular

        #step 1: normalised FFT
        X = np.fft.rfft(x * win) / n
        freq = np.fft.rfftfreq(n, d=dt)
        amp = np.abs(X)
        #step 2: one-sided spectrum, double all bins except DC (index 0)
        #and Nyquist (index n//2 when n is even)
        amp[1:-1] *= 2.0
        #step 3: correct for window coherent gain
        amp = amp / coherent_gain
        #step 4: peak → RMS for sinusoidal components
        rms_amp = amp / np.sqrt(2.0)
        return freq, rms_amp

    #analyse a waveform
    def analyse(
        self,
        waveform: np.ndarray,
        dt: float,
        n_cycles: Optional[int] = None,
        rated: Optional[float] = None,
    ) -> HarmonicSpectrum:
        """perform harmonic analysis on a waveform..."""
        #select analysis window..use exact integer number of cycles
        T_grid = 1.0 / self.f_grid
        samples_per_cycle = T_grid / dt
        if n_cycles is None:
            n_cycles = max(1, int(len(waveform) / samples_per_cycle) - 1)
        n_use = int(n_cycles * samples_per_cycle)
        if n_use > len(waveform):
            n_use = len(waveform)
        x = waveform[-n_use:]   #use end of waveform (steady state)
        freq, rms_amp = self._fft_spectrum(x, dt)
        #extract harmonic bins
        df = freq[1] - freq[0] if len(freq) > 1 else 1.0
        #compute raw (un-windowed) FFT once for phase extraction, O(n log n) × 1
        X_raw = np.fft.rfft(x)
        def get_harmonic_rms(order: int) -> Tuple[float, float]:
            f_target = order * self.f_grid
            idx = int(round(f_target / df))
            idx = np.clip(idx, 0, len(rms_amp) - 1)
            #sse the center bin only (not max of ±1 bins)
            #taking max would amplify spectral leakage...the center bin is the physically correct bin for an integer-cycle-aligned window.
            mag   = rms_amp[idx]
            phase = np.degrees(np.angle(X_raw[idx]))  # phase from pre-computed rfft
            return float(mag), float(phase)
        orders = np.arange(1, self.n_harmonics + 1)
        mags   = np.zeros(len(orders))
        phases = np.zeros(len(orders))
        for k, h in enumerate(orders):
            mags[k], phases[k] = get_harmonic_rms(h)

        V1    = mags[0]   #fundamental RMS
        V_rms = np.sqrt(np.mean(x ** 2))
        # THD: ratio of harmonic content to fundamental
        V_harm_sq = np.sum(mags[1:] ** 2)
        THD = 100.0 * np.sqrt(V_harm_sq) / max(V1, 1e-9)
        # crest factor
        crest = np.max(np.abs(x)) / max(V_rms, 1e-9)
        return HarmonicSpectrum(
            f_fund=self.f_grid,
            harmonics=orders,
            frequencies=orders * self.f_grid,
            magnitudes=mags,
            phases=phases,
            THD=float(THD),
            fundamental=float(V1),
            rms_total=float(V_rms),
            crest_factor=float(crest),
            n_harmonics=self.n_harmonics,
        )
    #IEEE 1547-2018 compliance
    def ieee1547_check(
        self,
        spec: HarmonicSpectrum,
        I_rated: Optional[float] = None,
    ) -> IEEE1547Result:
        """check a *current* harmonic spectrum against IEEE 1547-2018 Table 2 limits"""
        I_base = I_rated or spec.fundamental
        violations = []
        harmonic_limits   = {}
        harmonic_actuals  = {}
        #THD check
        thd_ok = spec.THD <= self._THD_LIMIT
        if not thd_ok:
            violations.append(
                f"THD = {spec.THD:.2f}% exceeds limit of {self._THD_LIMIT}%"
            )
        #individual harmonic checks, skip H1 
        for h, mag in zip(spec.harmonics[1:], spec.magnitudes[1:]):
            h = int(h)
            pct = 100.0 * mag / max(I_base, 1e-9)
            harmonic_actuals[h] = pct
            #find limit for this harmonic order
            limit = None
            for (lo, hi), lim in self._I_LIMITS.items():
                if lo <= h <= hi:
                    limit = lim
                    break
            if limit is None:
                limit = 0.3 # >99th: 0.3 %
            #even harmonics: 25 % of the corresponding odd-harmonic limit
            #per IEEE 1547-2018 Table 2, footnote
            if h % 2 == 0:
                limit *= 0.25
            harmonic_limits[h] = limit
            if pct > limit:
                violations.append(
                    f"H{h}: {pct:.2f}% > limit {limit:.2f}%"
                )
        compliant = len(violations) == 0
        return IEEE1547Result(
            compliant=compliant,
            thd_limit=self._THD_LIMIT,
            thd_actual=spec.THD,
            violations=violations,
            harmonic_limits=harmonic_limits,
            harmonic_actuals=harmonic_actuals,
        )
    # Convenience: voltage THD only
    def thd_voltage(self, V_out: np.ndarray, dt: float) -> float:
        """Quick computation of voltage THD [%]."""
        spec = self.analyse(V_out, dt)
        return spec.THD
    #SINAD
    def sinad(self, waveform: np.ndarray, dt: float) -> float:
        """Signal-to-Noise-and-Distortion ratio [dB] SINAD = 20 * log10(V_rms / sqrt(V_noise² + V_harmonic²))"""
        spec = self.analyse(waveform, dt)
        v_total = spec.rms_total
        v_fund  = spec.fundamental
        noise_distortion = np.sqrt(max(v_total ** 2 - v_fund ** 2, 0.0))
        if noise_distortion < 1e-12:
            return 120.0
        return float(20 * np.log10(v_fund / noise_distortion))

    #frequency sweep: THD vs filter capacitance
    def thd_vs_capacitance(
        self,
        V_AB: np.ndarray,
        dt: float,
        L_f: float,
        C_values: np.ndarray,
        R_load: float = 100.0,
        R_L: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """compute THD for a range of filter capacitor values by applying the LC filter analytically in the frequency domain.
the unfiltered V_AB spectrum is computed once; for each C_f value the LC transfer function is evaluated at every frequency bin and the
resulting filtered spectrum is used to compute THD"""
        freq_in, amp_in = self._fft_spectrum(V_AB, dt)
        df = freq_in[1] - freq_in[0] if len(freq_in) > 1 else 1.0
        thd_arr = np.zeros(len(C_values))

        # fundamental bin index
        i_f1 = int(round(self.f_grid / df))
        i_f1 = np.clip(i_f1, 1, len(amp_in) - 1)

        for i, C_f in enumerate(C_values):
            omega = 2 * np.pi * freq_in
            # compute impedances; handle DC bin (omega[0]=0) explicitly to avoid NaN.
            # At DC: Z_C → ∞ (capacitor blocks DC), Z_par → R_load, H → R_load/(R_L+R_load) ≈ 1
            with np.errstate(divide='ignore', invalid='ignore'):
                Z_L = R_L + 1j * omega * L_f
                # Z_C = 1/(jωC); treat DC bin as open circuit (large real number)
                denom_C = 1j * omega * C_f
                Z_C = np.where(omega == 0, 1e15 + 0j, 1.0 / denom_C)
                # Z_par = Z_C || R_load; at DC Z_C>>R_load so Z_par → R_load
                Z_par = (Z_C * R_load) / (Z_C + R_load)
                H = Z_par / (Z_L + Z_par)

            # guard any residual NaN/Inf before using amplitudes
            H_mag = np.where(np.isfinite(np.abs(H)), np.abs(H), 0.0)
            amp_out = amp_in * H_mag

            V1 = amp_out[i_f1]

            # sum harmonic power (H2: n_harmonics)
            harm_sq = 0.0
            for hn in range(2, self.n_harmonics + 1):
                idx = int(round(hn * self.f_grid / df))
                if 0 < idx < len(amp_out):
                    harm_sq += amp_out[idx] ** 2

            thd_arr[i] = 100.0 * np.sqrt(harm_sq) / max(V1, 1e-9)

        return C_values, thd_arr


def print_harmonic_table(spec: HarmonicSpectrum, max_order: int = 20):
    print(f"\n{'='*55}")
    print(f"  Harmonic Spectrum Analysis  (f_fund = {spec.f_fund} Hz)")
    print(f"{'='*55}")
    print(f"  {'Order':>5}  {'Freq (Hz)':>10}  {'Mag (Vrms)':>12}  {'% of Fund':>10}")
    print(f"  {'-'*48}")
    for h, f, m in zip(spec.harmonics[:max_order],
                        spec.frequencies[:max_order],
                        spec.magnitudes[:max_order]):
        pct = 100.0 * m / max(spec.fundamental, 1e-9)
        marker = " ◄" if h == 1 else ""
        print(f"  {int(h):>5}  {f:>10.1f}  {m:>12.4f}  {pct:>10.3f}%{marker}")
    print(f"\n  Fundamental (V1)   = {spec.fundamental:.4f} Vrms")
    print(f"  RMS (total)        = {spec.rms_total:.4f} Vrms")
    print(f"  THD                = {spec.THD:.3f}%")
    print(f"  Crest Factor       = {spec.crest_factor:.3f}")
    print(f"{'='*55}\n")


def print_ieee1547(result: IEEE1547Result):
    status = "✓ PASS" if result.compliant else "✗ FAIL"
    print(f"\n{'='*45}")
    print(f"  IEEE 1547-2018 Compliance: {status}")
    print(f"  THD: {result.thd_actual:.2f}%  (limit: {result.thd_limit:.1f}%)")
    if result.violations:
        print(f"\n  Violations ({len(result.violations)}):")
        for v in result.violations:
            print(f"    - {v}")
    else:
        print("  No harmonic violations detected.")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    #self-test with a synthetic waveform, known harmonics, verifiable TH.. dt=1e-4 s (100 µs): T_grid/dt = 0.02/1e-4 = 200 exactly (integer),
    #guaranteeing perfect cycle alignment and zero spectral leakage
    #window choice: window='rect' (rectangular) because:
    #1-The data is exactly cycle-aligned (no truncation leakage).
    #2-Hann window introduces inter-bin sidelobe leakage: H1 (230V) leaks ~0.0075% into the H2 bin, giving a false H2 reading of ~0.017V.. with perfectly aligned data this is entirely a Hann artefact, not signal
    #for REAL inverter waveforms (where exact cycle alignment isn't guaranteed), keep the default window='hann'...its better leakage rejection outweighs the
    #sidelobe interference on the typically small harmonic amplitudes present
    f0  = 50.0
    dt  = 1e-4 #100 µs, exact integer cycles
    t   = np.arange(0, 0.1, dt) #1000 samples = 5 full cycles
    V   = (230 * np.sqrt(2) * np.sin(2*np.pi*f0*t)
           + 5.0 * np.sin(2*np.pi*3*f0*t)
           + 3.0 * np.sin(2*np.pi*5*f0*t)
           + 1.5 * np.sin(2*np.pi*7*f0*t))

    #use rectangular window: zero leakage on cycle-aligned data
    pq   = PowerQualityAnalyser(f_grid=f0, window='rect')
    spec = pq.analyse(V, dt)
    print_harmonic_table(spec, max_order=10)

    check = pq.ieee1547_check(spec, I_rated=spec.fundamental)
    print_ieee1547(check)