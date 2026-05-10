# Solar PV Grid-Tied Inverter Simulation

> **A dual-environment simulation framework for the complete energy conversion chain from photovoltaic panel to utility grid, implemented in Python (behavioural) and Simulink (continuous-time switched).**

&nbsp; IEEE 1547-2018 Compliant &nbsp;-&nbsp; **Python:** 3.8+ &nbsp;-&nbsp; **MATLAB:** R2021b+

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Physical System Overview](#2-physical-system-overview)
  - [2.1 Conversion Chain Topology](#21-conversion-chain-topology)
  - [2.2 Operating Conditions](#22-operating-conditions)
- [3. Mathematical Foundations](#3-mathematical-foundations)
  - [3.1 PV Panel  Single-Diode Equivalent Circuit](#31-pv-panel--single-diode-equivalent-circuit)
  - [3.2 Boost DC-DC Converter](#32-boost-dc-dc-converter)
  - [3.3 Maximum Power Point Tracking](#33-maximum-power-point-tracking)
  - [3.4 H-Bridge Inverter and SPWM Modulation](#34-h-bridge-inverter-and-spwm-modulation)
  - [3.5 LC Output Filter and Power Quality](#35-lc-output-filter-and-power-quality)
- [4. Simulink Continuous-Time Model](#4-simulink-continuous-time-model)
  - [4.1 Root-Level Architecture](#41-root-level-architecture)
  - [4.2 Subsystem Block Diagrams](#42-subsystem-block-diagrams)
  - [4.3 Solver Configuration](#43-solver-configuration)
  - [4.4 Scope: Four-Panel Transient Display](#44-scope-four-panel-transient-display)
  - [4.5 Verified Steady-State Results](#45-verified-steady-state-results)
- [5. Python Implementation Architecture](#5-python-implementation-architecture)
  - [5.1 Module Structure](#51-module-structure)
  - [5.2 Data Flow: Irradiance to Grid](#52-data-flow-irradiance-to-grid)
- [6. Validation and Results](#6-validation-and-results)
  - [6.1 Figure Guide and Plot Placement](#61-figure-guide-and-plot-placement)
  - [6.2 System Performance Summary](#62-system-performance-summary)
  - [6.3 Energy Conversion Chain Efficiency](#63-energy-conversion-chain-efficiency)
  - [6.4 Cross-Environment Consistency](#64-cross-environment-consistency)

---

## 1. Executive Summary

This package simulates a **250 W single-phase grid-tied photovoltaic inverter** across two complementary environments:

| Environment | Purpose | Key Tool |
|---|---|---|
| **Python** | Algorithm development, MPPT benchmarking, FFT-based power quality analysis | `simulation.py`, `plots.py` |
| **Simulink** | Continuous-time switched simulation, closed-loop PI voltage control, real dead-time effects | `build_full_system.m` |

Both environments are cross-validated against each other and against analytical transfer-function predictions, with all residuals explained to within 0.2%.

**Verified key results (operating condition: G = 1000 W/m², T = 35 °C):**

| Metric | Python | Simulink | Specification |
|---|---|---|---|
| V_rms output | 230.45 V | **230.0 V** | 230 V nominal |
| P_out | 240.63 W | **238.6 W** | ~240 W |
| THD (filtered) | 1.02% | **1.08%** | < 5 % (IEEE 1547) |
| V_dc (boost) |  | **400.3 V** | 400 V |
| System η |  | **98.35%** | > 96 % |
| MPPT η (P&O) | **99.86%** |  | > 99 % |
| MPPT η (InCond) | **99.95%** |  | > 99 % |

> **Why does Simulink show higher THD (1.08% vs 1.02%)?**
> The Simulink model implements 2 µs switch dead-time that the Python simulation omits. Dead-time adds low-order odd harmonics (principally 3rd and 5th) with a theoretical upper-bound contribution of ≈ 4·t_dead·f_sw / (π·M_a) = 3.1 % before LC attenuation. H3 (150 Hz) and H5 (250 Hz) are below the filter corner (f_c = 1 kHz) so they pass with near-unity gain; the net dead-time contribution is ≈ 0.06 percentage points, matching the 0.06 pp gap observed (1.08 % − 1.02 %). Both values comply with the 5 % limit: Simulink 4.6× margin, Python 4.9× margin.

> **Note on Python V_rms = 230.45 V (target 230.00 V):**
> The 0.45 V (0.20%) offset is caused by discrete SPWM switching-instant quantisation at dt = 1 µs. The 10 kHz carrier period spans exactly 100 time-steps; each switching edge is quantised to the nearest µs, producing a systematic V_AB fundamental amplitude of 325.88 V vs the ideal 325.28 V. This propagates through the filter unchanged (|H(50 Hz)| ≈ 1.0) and raises V_rms by 0.45 V. The effect is fully documented and within the ±1% grid voltage tolerance of EN 50160. It is not a model error.

---

## 2. Physical System Overview

### 2.1 Conversion Chain Topology

Power flows through four sequential conversion stages.
The MPPT algorithm is a **control layer** that adjusts the boost duty cycle  it is not a power stage in its own right.

**Signal conventions:**
- At T = 35 °C, G = 1000 W/m²: V_mpp = 29.48 V, I_mpp = 8.13 A, P_pv = 239.7 W *(operating condition, not STC)*
- MPPT outputs a voltage reference V_ref that the boost PI controller tracks
- The inverter uses a fixed 400 V DC bus constant matched to the boost output

### 2.2 Operating Conditions

Two distinct conditions appear throughout this project and must not be confused:

| Condition | G (W/m²) | T_cell (°C) | V_mpp (V) | I_mpp (A) | P_mpp (W) | Used in |
|---|---|---|---|---|---|---|
| **STC** (Standard Test) | 1000 | 25 | 30.67 | 8.158 | 250.2 | `pv_model.py __main__` |
| **Operating** (nominal sim) | 1000 | 35 | 29.48 | 8.130 | 239.7 | `simulation.py`, Simulink |

The temperature shift from 25 °C to 35 °C reduces V_mpp by approximately 1.19 V (β_V · ΔT = −0.123 × 10 = −1.23 V theoretical; actual model gives −1.19 V due to the bandgap-based I₀(T) model) and reduces P_mpp by ~4.2 % per 10 K, explaining the 250.2 → 239.7 W drop. All electrical design targets (R_load, boost D_ss, PI gains) are derived at the operating condition (T = 35 °C), not STC.

---

## 3. Mathematical Foundations

### 3.1 PV Panel, Single-Diode Equivalent Circuit

#### 3.1.1 Governing Equation

The single-diode model (Villalva et al., 2009) represents a PV cell as a photocurrent source with a parallel diode and resistive leakage:

> **I = I_ph − I₀ · [exp((V + I·R_s) / (n·V_T)) − 1] − (V + I·R_s) / R_sh**

| Symbol | Value | Description |
|---|---|---|
| I_ph | irradiance-scaled | Photogenerated current [A] |
| I₀ | T-dependent | Diode saturation current [A] |
| R_s | 0.274 Ω | Series resistance (contacts, wiring)  Villalva-fitted, 4.6 mΩ/cell × 60 cells |
| R_sh | 415 Ω | Shunt resistance (leakage paths)  Villalva-fitted |
| n | 1.10 | Diode ideality factor (fitted) |
| V_T | n·N_s·k·T/q | Thermal voltage (N_s = 60 cells in series) |
| E_g | 1.121 eV | Silicon bandgap energy (for I₀ temperature scaling) |

> **Note on R_s and R_sh:** These are phenomenological curve-fit parameters from the Villalva iterative method. They are not independently measurable physical quantities but absorb all second-order effects (cell interconnect resistance, distributed mismatch, temperature gradients) so that the single-diode model reproduces the correct I_sc, V_oc, and P_mp at STC. R_s = 0.274 Ω corresponds to 4.6 mΩ/cell, within the physical range of 3–10 mΩ/cell for crystalline silicon.

The equation is **implicit in I**, it cannot be inverted analytically.

#### 3.1.2 Irradiance and Temperature Scaling

Photocurrent scales linearly with irradiance and has a small linear temperature dependence:

> **I_ph(G,T) = (I_sc,stc + α_I·ΔT) · G/G_stc · N_p**,  α_I = 5.3×10⁻⁴ A/K

Saturation current follows the standard bandgap-based three-parameter temperature model:

> **I₀(T) = I₀,STC × (T/T_STC)³ × exp[q·E_g / (n·k) × (1/T_STC − 1/T)]**

where I₀,STC is computed at construction from the open-circuit condition (including the R_sh shunt current):

> **I₀,STC = (I_ph,STC − V_oc,STC/R_sh) / [exp(V_oc,STC / V_T,STC) − 1]**

This correctly captures the exponential increase of dark current with temperature and produces the logarithmic V_oc(G) dependence directly from the diode equation  no separate V_oc temperature-shift formula is required. Raising T by 10 K drops V_mpp ≈ 1.2 V and P_mpp ≈ 4.2 %, consistent with the simulation values (250.2 W → 239.7 W).

#### 3.1.3 Maximum Power Point via Golden-Section Search

The MPP is located by minimising −P(V) = −V·I(V) over V ∈ (0, V_oc):

```python
result = minimize_scalar(
    lambda v: -v * self.current(v, G, T_cell),
    bounds=(0.1, V_oc_est * 0.999),
    method="bounded",          # golden-section / Brent's method
)
V_mpp, I_mpp = result.x, self.current(result.x, G, T_cell)
```

Each I(V) evaluation uses Brent's root-finding on the implicit residual equation, with bracket [−I_ph·0.02, I_ph·1.05] and exponent clipping to [−500, 500] for numerical stability. The lower bracket of −I_ph·0.02 (rather than −0.01 A) ensures a valid sign change even near V_oc where the shunt current can make a fixed −0.01 A bracket fail.

#### 3.1.4 PV Characteristics, Figure 1
<a id="fig1"></a>
<img width="3567" height="1530" alt="fig01_pv_iv_curves" src="https://github.com/user-attachments/assets/8851a40e-c89d-4715-ab12-1aa47f92fc90" />


> **→ Figure 1: `fig01_pv_iv_curves.png`**
> *Left  I-V family (G ∈ {200, 400, 600, 800, 1000} W/m², T = 25 °C). Right  P-V curves with MPP locus.*

**Reading Figure 1:**
The I-V curves exhibit the three classic regions: the flat current-source region where I ≈ I_ph ∝ G, the diode knee near V_oc, and the steep voltage-source drop-off. The orange MPP locus tracks nearly horizontally across all irradiance levels  V_mpp changes by approximately 1.1 V across the five curves (29.6 V at G = 200 to 30.7 V at G = 800 W/m², with a slight rollback at G = 1000 W/m² due to I²·R_s compression at high currents). This confirms that voltage is primarily a logarithmic function of irradiance while current scales linearly. The unimodal maxima is shown, which is a prerequisite for the hill-climbing MPPT algorithms to converge globally. Quantitative check: P_mpp at G = 1000 W/m², T = 25 °C reads 250.2 W against the 250 W datasheet rating (< 0.1 % error).

#### 3.1.5 MPP Locus  Figure 2
<a id="fig2"></a>
<img width="3268" height="1530" alt="fig02_mpp_locus" src="https://github.com/user-attachments/assets/30669705-7cf6-496d-82f3-aa115a356043" />


> **→ Figure 2: `fig02_mpp_locus.png`**
> *Left  V_mpp vs G at T = 25 °C and T = 35 °C. Right  P_mpp vs G at both temperatures. Operating point (G = 1000 W/m², T = 35 °C) annotated on both panels.*

**Reading Figure 2:**
V_mpp rises with irradiance (logarithmic dependence via I_sc scaling) then rolls back slightly above G ≈ 800 W/m² due to I²·R_s compression  this is physically correct and visible as the peak then slight dip in both temperature curves. The T = 35 °C curve sits uniformly ~1.19 V below the T = 25 °C curve across all irradiance levels, consistent with the bandgap-model temperature effect. P_mpp scales near-linearly with G on both curves; the ~4.2 % drop per 10 K (μ_Pmp · ΔT) produces the separation between the STC and operating-condition lines. The annotated marker confirms the simulation operating point: V_mpp = 29.48 V, P_mpp = 239.7 W at G = 1000 W/m², T = 35 °C.

---

### 3.2 Boost DC-DC Converter

#### 3.2.1 Averaged State-Space Model (CCM)

In Continuous Conduction Mode (CCM), the boost converter is described by two first-order ODEs for the averaged inductor current Ī_L and capacitor voltage V̄_C:

> **dĪ_L/dt = (V_in − R_L·Ī_L − (1−D)·V̄_C) / L**

> **dV̄_C/dt = ((1−D)·Ī_L − V̄_C/R_load) / C**

These are the **exact equations integrated in the Simulink averaged model** (`build_boost_averaged_subsystem`). The output is V_out = V̄_C. At steady state both derivatives vanish, giving the standard voltage conversion ratio:

> **M = V_out / V_in = 1 / (1−D)**  (ideal CCM)

With R_L = 0.05 Ω and R_load ≈ 670 Ω, resistive losses shift the actual gain slightly below ideal. Solving at the operating point:

> **D_ss = 1 − V_in/V_out = 1 − 29.48/400 = 0.9263**

This is the value pre-loaded into the PI integrator initial condition so the boost starts at its operating point with zero transient.

#### 3.2.2 CCM Verification

CCM holds when the minimum inductor current exceeds zero:

> **ΔI_L,pp = (V_in·D) / (L·f_sw) = (29.48 × 0.9263) / (2×10⁻³ × 20×10³) = 0.683 A**

The average inductor current is Ī_L = P_in/V_in = 239.7/29.48 = 8.13 A, giving a ripple ratio of 0.683/8.13 = 8.4 % and a minimum of 8.13 − 0.34 = 7.79 A ≫ 0. CCM is guaranteed across the full input voltage range (15 V to 45 V).

#### 3.2.3 Conduction Losses and Efficiency

The primary loss mechanism is inductor copper loss:

> **P_loss,L = R_L · Ī_L² = 0.05 × 8.13² = 3.30 W**

Additional losses (diode, MOSFET on-resistance) are captured in `boost_converter.efficiency()`. The combined model predicts η = 98.2 %, consistent with the terminal output (98.19 %). Note: the efficiency curve in Figure 5 is computed on a **constant-power locus** (I_in = P_mpp/V_in), which correctly reflects the panel operating condition at each V_in value.

#### 3.2.4 Boost Characteristics  Figure 5
<a id="fig5"></a>
<img width="4166" height="1319" alt="fig05_boost_characteristics" src="https://github.com/user-attachments/assets/4fc8aab5-3839-4109-b878-ecf6eefa3f58" />


> **→ Figure 5: `fig05_boost_characteristics.png`**
> *Left  Duty cycle D vs V_in. Centre  Efficiency η vs V_in (constant-power locus, P = 240 W). Right  Inductor current ripple ΔI_L vs V_in.*

**Reading Figure 5:**
The duty cycle curve decreases from 95 % at V_in ≈ 21 V (at lower voltages D is clamped at 95 %, outside the panel's operating range) to ≈ 89 % at V_in = 45 V, with a dashed marker at V_mpp = 29.48 V confirming D ≈ 92.6 %. Efficiency increases monotonically with V_in because I_in = P/V_in decreases, reducing all conduction losses. The ripple panel shows ΔI_L ∝ V_in (linear, as predicted by the ripple formula), reaching 0.683 A at the operating point  8.4 % of rated current, well within the 15 % design specification.

---

### 3.3 Maximum Power Point Tracking

#### 3.3.1 The MPP Optimisation Problem

The PV characteristic P(V) = V · I(V) is a smooth unimodal function with a unique maximum at V_mpp. This maximum shifts with irradiance and temperature and must be tracked continuously without direct measurement of V_mpp. Two algorithms are implemented.

#### 3.3.2 Perturb and Observe (P&O)

At each sampling instant (dt = 0.05 s), the algorithm compares the power before and after a small voltage perturbation δV = 0.5 V:

> if ΔP > 0:   V_ref += δV · sign(ΔV)
> else:        V_ref −= δV · sign(ΔV)

**Steady-state oscillation:** The algorithm never stops perturbing, creating a ±δV limit cycle around V_mpp.. the associated power loss from a quadratic P-V approximation gives:

Δη_P&O ≈ ½·(δV/V_mpp)² ≈ ½·(0.5/29.48)² = 0.014 %
The measured η = 99.86 % implies a 0.14 % loss which is ten times larger than the quadratic estimate. The discrepancy arises because the actual P-V curve has significantly steeper curvature at the MPP than a parabola: measured directly, the average power at V_mpp ± 0.5 V is 0.234 % below P_mpp. The formula gives a lower bound; the 0.09 pp advantage of InCond over P&O on the step profile reflects this real IV-curve-curvature effect.

#### 3.3.3 Incremental Conductance (InCond)

The condition dP/dV = 0 at the MPP yields a conductance criterion:

> **dI/dV = −I/V**  (at MPP);  dI/dV ≶ −I/V  (left / right of MPP)

The algorithm perturbs only when the incremental conductance dI/dV departs from the instantaneous conductance −I/V by more than a tolerance ε = 0.01 A/V. When exactly at the MPP it stops perturbing, eliminating the steady-state oscillation present in P&O. This explains the higher η = 99.95 % under stable irradiance and faster recovery during cloud shadows.

#### 3.3.4 MPPT Benchmarks  Figures 3 and 4
<a id="fig3"></a>
<img width="3016" height="2082" alt="fig03_mppt_step" src="https://github.com/user-attachments/assets/bd668a74-4502-48d0-8e3d-bd6cc346e261" />


> **→ Figure 3: `fig03_mppt_step.png`**
> *Three-panel: (top) stepped irradiance profile; (middle) P_actual vs P_MPP for P&O and InCond; (bottom) normalised tracking error.*

**Reading Figure 3:**
The stepped profile (200 → 600 → 1000 → 800 → 400 → 1000 W/m²) creates sharp power transitions. The P_MPP theoretical reference (green dashed) is plotted on top of the algorithm traces, confirming both converge to within the P&O oscillation band. Settling time after each step is ≈ 0.3–0.5 s for both algorithms. The overall and steady-state efficiencies are 99.86 %/99.86 % (P&O) and 99.95 %/99.96 % (InCond). InCond outperforms P&O by 0.09 percentage points overall on this profile.

<a id="fig4"></a>
<img width="2995" height="2082" alt="fig04_mppt_cloudy" src="https://github.com/user-attachments/assets/a8f93ddf-9b1c-45f6-9b9b-eb967e0c3ff1" />

> **→ Figure 4: `fig04_mppt_cloudy.png`**
> *Three-panel: realistic partial-cloud profile with Gaussian dips; tracking comparison; tracking error.*

**Reading Figure 4:**
The cloudy profile introduces rapid transients (dG/dt up to several hundred W/m²/s) that stress both algorithms. P&O efficiency drops to 98.55 % (overall) while InCond maintains 99.11 %, a 0.56 percentage-point advantage. During the fastest cloud shadows, peak tracking errors reach ±15–20 % for P&O versus ±8–12 % for InCond, and InCond recovers approximately one perturbation cycle faster. The insight is physical: InCond detects slope direction immediately from current and voltage derivatives, whereas P&O can take one additional perturbation cycle to determine which way to move. For systems in regions with frequent partial-cloud conditions, InCond is the preferred choice despite its added implementation complexity.

---

### 3.4 H-Bridge Inverter and SPWM Modulation

#### 3.4.1 Unipolar SPWM

The full H-bridge generates three voltage levels {+V_dc, 0, −V_dc} from the 400 V DC bus. Unipolar SPWM uses two complementary comparison operations:

> **S_A = 1 if r(t) > c(t)**,   **S_B = 1 if −r(t) > c(t)**

where the reference r(t) = M_a·sin(2π·f_g·t) and the carrier c(t) is a triangular wave at f_sw = 10 kHz. The pole voltage is:

> **V_AB(t) = (S_A − S_B) · V_dc ∈ {−400, 0, +400} V**

**Key unipolar property:** All harmonic energy appears at sidebands of **2·f_sw = 20 kHz** (the first harmonic cluster is at 2·f_sw ± n·f_grid, i.e., H397–H401 in the 50 Hz harmonic numbering scheme). This explains why the unfiltered THD = 41.75 % lives entirely outside the H1–H30 range  it is not a filter design failure but the mathematical property of unipolar modulation. The LC filter removes this 20 kHz cluster; residual low-order harmonics (H3–H29, totalling ~1 %) are caused by inverter switching-edge quantisation at the 1 µs time step.

#### 3.4.2 Modulation Index and AC Output

The fundamental of V_AB has peak amplitude M_a · V_dc. For a 230 V RMS target:

> **M_a = V_ac,peak / V_dc = 230·√2 / 400 = 0.8132**  (linear region: M_a < 1)

This gives:

> **V_rms,target = M_a · V_dc / √2 = 0.8132 × 400 / √2 = 230.00 V**

The value M_a = 0.8132 is used in the Python `SPWMModulator` and the Simulink SPWM subsystem. An older approximation M_a = 325/400 = 0.8125 (using the rounded 325 V instead of the exact 230√2 V) creates a 0.19 V systematic error in V_rms and is not used.

#### 3.4.3 Dead-Time and Its Effect on THD

Real gate drivers require a dead-time t_d = 2 µs between turning off one switch and turning on the complementary switch to prevent shoot-through. This is implemented in the Simulink SPWM subsystem. The theoretical harmonic voltage error per switching period is V_dc · t_d · f_sw = 400 × 2×10⁻⁶ × 10⁴ = 8 V, injecting odd harmonics with upper-bound THD contribution:

> **THD_dt ≤ 4·t_d·f_sw / (π·M_a) = 4 × 2×10⁻⁶ × 10⁴ / (π × 0.8132) = 3.1 %**  (before LC attenuation)

The LC filter provides no attenuation at 150 Hz or 250 Hz (both below f_c = 1 kHz). The actual dead-time contribution is ≈ 0.06 percentage points (1.08 % − 1.02 %), far below the 3.1 % bound because that bound assumes worst-case harmonic phase alignment across all switching cycles.

#### 3.4.4 Inverter Waveforms  Figure 6
<a id="fig6"></a>
<img width="3023" height="2082" alt="fig06_inverter_waveforms" src="https://github.com/user-attachments/assets/afa6f26b-bfad-4b3d-96a7-83b5986c268d" />

> **→ Figure 6: `fig06_inverter_waveforms.png`**
> *Three-panel: (top) V_AB and V_out vs time over 3 grid cycles with 2 ms zoom inset showing 3-level switching; (middle) inductor current I_L and output current I_out; (bottom) instantaneous power P(t).*

**Reading Figure 6:**
The top panel shows V_AB (violet) switching between ±400 V with a sinusoidal envelope  the pulse widths growing and shrinking at the 50 Hz grid rate confirms correct SPWM operation. The **2 ms zoom inset** in the upper right explicitly shows the three-level character: the signal dwells at 0 V near zero-crossings of the reference, and reaches ±400 V during peak half-cycles. At the full 60 ms view the individual 100 µs PWM pulses are compressed below pixel width and appear as a solid fill; the inset prevents misreading this as a 2-level square wave. V_out (light blue) is a clean 230 V RMS sinusoid with peak ≈ 325 V, demonstrating the LC filter's attenuation of the 20 kHz switching cluster. The middle panel shows I_L with its high-frequency ripple (unipolar SPWM doubles the effective ripple frequency to 20 kHz) superimposed on the sinusoidal envelope, and I_out is visually identical to I_L's envelope. The bottom panel shows instantaneous power P(t) = V_out·I_out oscillating at 100 Hz around P_avg ≈ 240.6 W.

---

### 3.5 LC Output Filter and Power Quality

#### 3.5.1 Filter State-Space Equations

The LC filter with winding resistance R_L and capacitor ESR R_C is described by:

> **L_f · dI_L/dt = V_AB − R_L·I_L − V_C**

> **C_f · dV_C/dt = I_L − V_C/R_load**

> **V_out = V_C + R_C · (I_L − V_C/R_load)**

The R_C correction in the V_out equation accounts for the ESR voltage drop across the capacitor branch. At 50 Hz (far below the 1 kHz corner), the filter is in its flat passband and the exact transfer function evaluates to |H(j·2π·50)| ≈ 1.00003, confirming near-unity gain at the fundamental.

#### 3.5.2 Filter Design

Corner frequency chosen at f_c = f_sw/10 = 1000 Hz ensures ≥ 40 dB attenuation at the switching frequency. Component values derived from the 15 % current ripple specification on a constant M_a = 1 worst-case basis:

> **L_f = V_dc / (8 · ΔI_L,spec · f_sw) = 400 / (8 × 0.3596 × 10⁴) = 13.91 mH**

> **C_f = 1 / ((2π·f_c)² · L_f) = 1 / ((2π × 10³)² × 0.01391) = 1.82 µF**

where ΔI_L,spec = 0.15 × I_rated × √2, I_rated = P_mpp / (V_dc/(2√2)) = 239.7 / 141.4 = 1.70 A.

Attenuation at f_sw = 10 kHz is exactly −40·log₁₀(f_sw/f_c) = −40·log₁₀(10) = **−40 dB**, confirmed by the simulation summary output. The `design_lc_filter()` function in `inverter.py` implements this formula directly, and its output (L_f = 13.91 mH, C_f = 1.82 µF) is used by `simulation.py` at runtime to configure the InverterParams for the time-domain simulation.

#### 3.5.3 THD Definition

> **THD = √(Σ Vₙ²,  n=2..N) / V₁  ×  100 %**

where Vₙ is the RMS amplitude of the nth harmonic and V₁ is the fundamental RMS. The Python analyser computes this from a windowed FFT of an integer number of grid cycles (20 cycles at dt = 1 µs, using a rectangular window for exact-cycle-aligned data to eliminate Hann sidelobe leakage), ensuring bin-exact harmonic extraction without spectral leakage.

#### 3.5.4 Harmonic Spectrum  Figure 7
<a id="fig7"></a>
<img width="3241" height="2135" alt="fig07_harmonic_spectrum" src="https://github.com/user-attachments/assets/c830cf3a-d087-45ea-91dc-3c502827840b" />

> **→ Figure 7 (two panels): `fig07_harmonic_spectrum.png`**
> *Top  Continuous log-frequency spectrum (50 Hz to 20 kHz) of V_AB and V_out; the 2·f_sw = 20 kHz switching cluster and the LC filter attenuation are both visible. Bottom  Low-order bar chart H2–H29 of filtered V_out showing residual harmonic compliance margin.*

**Reading Figure 7:**
The top panel tells the complete filtering story. Both V_AB (lavender) and V_out (blue) show the fundamental at 50 Hz at the same amplitude (~100% of V₁)  the filter passes the fundamental with near-unity gain. In the low-frequency region (50 Hz to 1 kHz), both spectra are nearly identical (0.06–0.23%) because the filter corner at 1 kHz provides little attenuation below it. The decisive difference is visible at **2·f_sw = 20 kHz**: the V_AB spectrum peaks sharply at H399 = 19 950 Hz (37.6% of V₁, annotated) and H397 = 19 850 Hz (17.9% of V₁)  these are the unipolar SPWM sidebands at 2·f_sw ± n·f_grid that carry the entire 41.75% unfiltered THD. V_out at these same frequencies is attenuated to the noise floor, confirming the LC filter provides its designed ~40 dB attenuation at 20 kHz. The lower panel resolves the residual low-order harmonics of V_out (H2–H29), all below 0.3% and well under the 5% IEC 61727 voltage THD limit.

#### 3.5.5 Bode Plot  Figure 8
<a id="fig8"></a>
<img width="2558" height="1853" alt="fig08_bode_plot" src="https://github.com/user-attachments/assets/fd989816-9836-4221-b896-bbeff1530d28" />

> **→ Figure 8: `fig08_bode_plot.png`**
> *Bode diagram of H(jω) = V_out/V_AB: magnitude (top) and phase (bottom).*

**Reading Figure 8:**
The magnitude plot has three distinct regions. The passband (10 Hz – 1 kHz) is flat at 0 dB: the fundamental at 50 Hz passes with unity gain. The transition at f_c = 1 kHz shows the resonance peak (~+8 dB, consistent with Q = 171), then the characteristic −40 dB/decade 2nd-order slope. At f_sw = 10 kHz the attenuation reads −40 dB exactly, matching the analytical prediction (the sky-blue dotted reference line confirms this). The phase plot is smooth and bounded: −90° at resonance, asymptoting toward −180° at high frequency. The rapid phase transition at f_c is consistent with Q = 171 (very lightly damped; primary damping comes from R_load = 221.7 Ω through the 1/(R_load·C_f) term).

#### 3.5.6 Filter Design Sweep  Figure 9
<a id="fig9"></a>
<img width="2667" height="1451" alt="fig09_thd_vs_capacitance" src="https://github.com/user-attachments/assets/30d6fed8-91a2-4325-8933-bc06e460215e" />

> **→ Figure 9: `fig09_thd_vs_capacitance.png`**
> *THD vs C_f over one decade, with the chosen design point marked and the 5% compliance limit visible.*

**Reading Figure 9:**
The sweep reveals a non-monotonic relationship between C_f and THD. For C_f < 0.3 µF the filter corner is above the main SPWM harmonics and filtering is insufficient. At C_f ≈ 1.8 µF a local minimum appears: this is the design point where f_c = 1 kHz aligns optimally with f_sw/10. For C_f > 3 µF the resonance at lower frequencies interacts with SPWM sidebands at intermediate frequencies, causing the ripple structure in the right region. The design choice C_f = 1.82 µF sits at the left edge of the minimum, where THD ≈ 1.02% and component stress is not excessive. The 5% IEC 61727 compliance limit (red dashed) is shown in full  the entire curve falls well below it, demonstrating robustness across a wide capacitance range.

#### 3.5.7 IEEE 1547-2018 Compliance  Figure 10
<a id="fig10"></a>
<img width="3867" height="1468" alt="fig10_ieee1547_compliance" src="https://github.com/user-attachments/assets/0654b39e-8596-42b4-9bd4-e9c598454a12" />

> **→ Figure 10: `fig10_ieee1547_compliance.png`**
> *Individual current harmonic magnitudes (blue bars) against IEEE 1547-2018 Table 2 limits (red step function).*

**Reading Figure 10:**
This is the compliance proof. The red step function encodes the standard current harmonic limits: 4 % for H3–H9 odd (1 % for even), 2 % for H11–H15, 1.5 % for H17–H21, 0.6 % for H23–H33. Every blue bar falls far below the red step:

| Harmonic | Measured | Limit | Margin |
|---|---|---|---|
| H3 | 0.21 % | 4.00 % | **19×** |
| H5 | ~0.15 % | 4.00 % | **27×** |
| H7–H9 | < 0.25 % | 4.00 % | > 16× |
| H11–H15 | < 0.10 % | 2.00 % | > 20× |
| H23–H33 | < 0.45 % | 0.60 % | > 1.3× |
| Even all | < 0.01 % | ≤ 1.00 % | > 100× |
| **Total THD** | **1.02 %** | **5.00 %** | **4.9×** |

The absence of even harmonics confirms the symmetric unipolar SPWM implementation. The large margins provide headroom for component aging, temperature variation, and partial-load operation.

---


## 4. Simulink Continuous-Time Model

The Simulink model (`full_system.slx`, generated by `build_full_system.m`) is a continuous-time switched simulation that complements the Python behavioural model. It adds four effects that the Python simulation omits: closed-loop PI voltage control with realistic transient dynamics, 2 µs switch dead-time, continuous LP-filtered RMS and THD measurement, and physics-based efficiency from actual copper losses.


```matlab
% Build and simulate
build_full_system        % generates full_system.slx
sim('full_system')       % runs 6-second simulation
```

---

### 4.1 Root-Level Architecture

The model contains six subsystems wired at the root level. Power flows left-to-right through the conversion chain; the PI feedback loop and PQ measurements branch off as indicated.


<img width="1920" height="991" alt="full_system _ " src="https://github.com/user-attachments/assets/c127fdf9-d344-4529-b811-2fdae5797a64" />


> **→ Figure A: `fig_simulink_root.png`**
> *Screenshot of the full_system.slx root canvas showing all six subsystems, signal routing, display blocks, and the efficiency calculation chain.*

**Reading Figure A:**
Working left to right: the PV block receives G and T constants and outputs Vmpp, Impp, and Pmpp. Vmpp feeds into the Boost subsystem as V_in; the Boost output Vdc_out feeds back to the PI voltage loop whose output D drives the Boost duty-cycle input. Separately, the SPWM block generates Sa and Sb which drive the HBridge alongside a fixed 400 V constant (Vdc_inv). The HBridge produces VAB which feeds LCfilt. LCfilt outputs Vout and Iout to the PQ block, which produces Vrms, Pout, and THD. The efficiency block at the bottom right receives Ppv from PV, IL from Boost, and Iout from LCfilt to compute η from actual copper losses.

The signal flow is summarised below:

<img width="3700" height="3110" alt="02_Simulink_Model_Root-Level_Architecture" src="https://github.com/user-attachments/assets/060e0edd-cae4-4eb9-9810-3d9b9200cf43" />

### 4.2 Subsystem Block Diagrams

Each subsystem is built programmatically by `build_full_system.m`. The following figures show the internal block structure of the four computationally significant subsystems.

#### 4.2.1 Boost Averaged-State Subsystem


<img width="1920" height="991" alt="full_system_BOOST" src="https://github.com/user-attachments/assets/bc987cb8-e127-45a7-a593-68a02e4f1c6a" />

<img width="3700" height="1648" alt="05_2_Boost_Subsystem" src="https://github.com/user-attachments/assets/fadb2f80-a606-49b1-a35b-1afbd4c057b5" />

> **→ Figure B: `fig_simulink_boost.png`**
> *Internal block diagram of the Boost subsystem: two integrators (IL_int, VC_int) connected via the averaged CCM state equations, with the one_D complement term shared between both ODEs.*

**Reading Figure B:**
The subsystem implements the two first-order ODEs directly:

> **dĪ_L/dt = (V_in − R_L·Ī_L − (1−D)·V̄_C) / L**
> **dV̄_C/dt = ((1−D)·Ī_L − V̄_C/R_load) / C**

The input ports are V_in (from PV Vmpp) and D (from PI_boost output). The block `c1` (constant = 1) feeds the Sum block `one_D` to compute (1−D), which is then used in both multiplier blocks. The output port Vout_o carries V̄_C (= Vdc); the second output IL_o carries Ī_L for use in the efficiency calculation.

All integrator initial conditions are pre-loaded at the steady-state operating point so the simulation starts without a large step disturbance:

| State | Initial condition | Derivation |
|---|---|---|
| V̄_C (VC_int) | 400 V | Target DC bus voltage |
| Ī_L (IL_int) | 8.127 A | Ppv / Vmpp = 238.61 / 29.36 A |
| PI integrator | 0.9266 | D_ss = 1 − Vmpp / Vdc |

**PI controller design** (outer voltage loop, from small-signal analysis):

The plant DC gain at the operating point is:

> **G_vd,0 = V_in / (1−D_ss)² = 29.36 / 0.0734² = 5449 V/duty**

The boost LC resonance frequency is:

> **f_n = (1−D_ss) / (2π·√(LC)) = 0.0734 / (2π·√(2×10⁻³ × 470×10⁻⁶)) = 12.05 Hz**

The LC is lightly damped, so the PI crossover is set well below f_n to ensure stability:

> **f_c = f_n / 10 = 1.205 Hz**
> **K_p = 0.9 / G_vd,0 = 1.65 × 10⁻⁴**,   **K_i = K_p · 2π·f_c / 5 = 2.50 × 10⁻⁴**

Phase margin ≈ 165°. Small-signal settling time ≈ 5 / (2π × 1.205) = 0.66 s. The TYPE-1 PI gives zero steady-state error for a constant Vdc reference.

---

#### 4.2.2 SPWM Subsystem


<img width="1920" height="991" alt="full_system _ SPWM" src="https://github.com/user-attachments/assets/9daa4c8d-78cc-4ae9-abc7-bc47c353c066" />

<img width="3700" height="1508" alt="06_3_SPWM_Subsystem" src="https://github.com/user-attachments/assets/426a0b45-3dfe-49da-bb2c-5ca698d70c8f" />


> **→ Figure C: `fig_simulink_spwm.png`**
> *Internal block diagram of the SPWM subsystem: sinusoidal reference, triangular carrier, relay comparator, and the two Transport Delay blocks implementing the 2 µs dead-time for each leg.*

**Reading Figure C:**
The subsystem is self-contained  it has no input ports. A Sine Wave block generates the modulation reference r(t) = Ma·sin(2π·50·t) with Ma = 0.8132. A Repeating Sequence generates the triangular carrier c(t) at f_sw = 10 kHz in the range [−1, +1]. The Sum block computes r(t) − c(t) and feeds a Relay block (thresholds both at 0) whose output Comp_A is binary {0, 1}.

Dead-time is implemented as:

```matlab
Sa = delay(Comp_A,     t_dead)   % Leg A: delay rising edge by 2 µs
Sb = delay(1-Comp_A,   t_dead)   % Leg B: complement RAW signal, then delay
```

Complementing the **raw** Comp_A (not the delayed Sa) and then delaying gives exactly one t_dead gap per transition. The `c_one` constant (Value = '1', set explicitly to prevent version-dependent default evaluation) feeds the Sum block that computes `1 − Comp_A`.

---

#### 4.2.3 PQ Measurement Subsystem


<img width="1920" height="991" alt="full_system_PQ" src="https://github.com/user-attachments/assets/2544248d-f11b-49b9-a5a4-98704e4b1f7a" />

<img width="3700" height="2330" alt="09_6_PQ_Block_Subsystem" src="https://github.com/user-attachments/assets/e1025714-e312-4b46-863d-3087012f9e6a" />



> **→ Figure D: `fig_simulink_pq.png`**
> *Internal block diagram of the PQ subsystem: LP filters for Vrms and Pout, BPF at 50 Hz for fundamental extraction, harmonic subtraction chain for THD, and the gated THD output.*

**Reading Figure D:**
The subsystem takes Vout and Iout from LCfilt and produces three outputs.

**Vrms** path: Vout is squared, passed through LP(τ = 100 ms), clamped to ≥ 0, and square-rooted.

> **V_rms = √(LP(V_out²))**

**Pout** path: Vout·Iout is passed through LP(τ = 100 ms).

> **P_out = LP(V_out · I_out)**

**THD** path uses harmonic subtraction (not Vrms² − Vf²) because the 100 Hz LP ripple in LP(V²) is ±4100 V², which is 1400× larger than the THD signal of ≈ 3 V². Subtracting the fundamental before squaring eliminates this noise:

> **V_harm(t) = V_out(t) − BPF(V_out(t))**
> **THD = √(LP(V_harm²)) / max(√(LP(BPF(V_out)²)), 1 V) × 100**

The BPF is a 2nd-order band-pass at ω₀ = 2π×50 with ζ = 0.707:

> H_BPF(s) = 444.22s / (s² + 444.22s + 98696)

At ω₀, H_BPF(jω₀) = 1+0j exactly, so V_harm contains zero fundamental energy. The numerator is stored as the two-element vector `[444.22 0]`  the trailing zero encodes the s⁰ coefficient, making H(s) = 444.22·s/denom. A scalar `[444.22]` would produce a low-pass filter and was the root cause of the earlier THD = 22,850 % failure.

The THD output is multiplied by a Step gate (Before = 0, After = 1, t = 2 s) so that the display reads zero during the transient and only shows the settled value after t = 2 s.

The `vf_sf` Saturation block clamps the denominator to a minimum of 1 V, preventing division by zero at start-up when the fundamental has not yet built up.

**Efficiency** is computed at the root level (not inside PQ) from actual copper losses:

> **η = (P_pv − P_loss,boost − P_loss,inv) / P_pv**
> P_loss,boost = R_L,b · LP(I_L,b²) = 0.05 × 8.127² = 3.30 W
> P_loss,inv = R_L,i · LP(I_out²) = 0.5 × 1.037² = 0.54 W

Computing η as P_out / P_pv would give η ≡ 1 by algebraic identity (R_load was defined as V_rms² / P_pv), making it meaningless. The copper-loss formulation is the only physically correct approach.

---

### 4.3 Solver Configuration

| Parameter | Value | Reason |
|---|---|---|
| Solver | ode23 | Handles Relay zero-crossings correctly; ode23tb fails on switching |
| MaxStep | 1 µs | T_sw / 100 = 10 kHz / 100; resolves dead-time and switching edges |
| RelTol | 1×10⁻⁴ | Balances accuracy and simulation speed |
| StopTime | 6 s | > 9 PI settling time-constants (0.66 s each) |

---

### 4.4 Scope: Four-Panel Transient Display

The built-in Scope block is connected to four signals and configured with tight per-axis Y limits, individual titles, and distinct colours. The channel order groups the two slowly-varying "settling" signals at the top and the two AC/quality signals below.

| Axis | Signal | Source | Y range | Colour |
|---|---|---|---|---|
| 1 (top) | Vdc | Boost/1 | 0 – 700 V | Gold |
| 2 | Vrms | PQ/1 | 0 – 280 V | Orange |
| 3 | Vout | LCfilt/1 | −440 – +440 V | Sky blue |
| 4 (bottom) | THD | PQ/3 | 0 – 5 % | Green |


<img width="1920" height="991" alt="full_system_scoope" src="https://github.com/user-attachments/assets/779c347c-0b0a-4cd7-acdc-ad5e97dc8fdc" />



> **→ Figure E: `fig_simulink_scope.png`**
> *Four-panel scope output from a 6-second simulation run at G = 1000 W/m², T = 35 °C.*

**Reading Figure E panel by panel:**

**Panel 1  Vdc:**
The gold trace starts at the 400 V IC, overshoots to approximately 600 V within the first 0.1 s, and oscillates at the boost LC resonance frequency f_n = 12.05 Hz before the PI damps it to 400 V by t ≈ 1.5 s. The ~50% overshoot is expected for this lightly damped LC resonance; the observed peak is consistent with the analytical open-loop resonance. After t = 1.5 s the trace is flat at 400.3 V  the 0.3 V (0.075%) residual is a structural load-model offset, not a control error, and is explained in Section 4.5.

**Panel 2  Vrms:**
The trace rises from 0 and reaches 230 V with a smooth exponential approach, settling by t ≈ 0.5 s (= 5 × τ = 5 × 100 ms). The absence of 100 Hz ripple confirms that τ = 100 ms is sufficient. The trace is completely stable at 230.0 V from t ≈ 1 s onward.

**Panel 3  Vout instantaneous:**
The raw LC filter outputs a ±325 V sinusoid at 50 Hz with 10 kHz SPWM switching texture overlaid. The three-level unipolar SPWM character is visible: the signal dwells at 0 V near zero-crossings of the reference, and reaches ±400 V (the DC bus) during the peak half-cycles. The sinusoidal 50 Hz envelope at ±325 V corresponds to Ma·Vdc = 0.8132 × 400 = 325.3 V peak (230 V RMS). No startup transient is visible because the LC filter integrators start at zero and charge up within a few milliseconds.

**Panel 4  THD:**
The trace is held at zero by the gate until t = 2 s, when the Step block switches to 1 and the measurement activates. The trace then settles immediately to ~1.1%, which is stable and well below the 5% top of the axis (the IEEE 1547-2018 limit). The choice of 0–5% for the Y axis makes compliance visible at a glance: a non-compliant system would produce a trace touching the top of the panel.

---

### 4.5 Verified Steady-State Results

All five display values read at t = 6 s (nine settling time-constants from start-up):

| Display | Observed | Predicted | Δ | Explanation |
|---|---|---|---|---|
| disp_Vrms | **230.0 V** | 229.87 V | +0.055% | R_C reactive addition at 50 Hz ✓ |
| disp_Pout | **238.6 W** | 238.61 W | < 0.01% | LP(v·i) at mean ✓ |
| disp_THD  | **1.08 %** | 1.02–1.1% | within band | Dead-time adds ≈ 0.06 pp vs Python ✓ |
| disp_Vdc  | **400.3 V** | 400.0 V | +0.075% | Structural load-model offset ✓ |
| disp_eta  | **0.9835** | 0.9839 | −0.04% | LP ripple in LP(IL²) ✓ |

---

## 5. Python Implementation Architecture

### 5.1 Module Structure

```
src/
├── pv_model.py          PVPanel, PVPanelParams, characterise()
├── boost_converter.py   BoostConverter, BoostParams, design_boost()
├── mppt.py              PerturbObserve, IncrementalConductance, benchmark_mppt()
├── inverter.py          Inverter, InverterParams, SPWMModulator, design_lc_filter()
├── power_quality.py     PowerQualityAnalyser, HarmonicSpectrum, IEEE1547Result
├── simulation.py        run_mppt_benchmark(), run_power_quality(), run_full_system()
└── plots.py             generate_all_figures()

matlab/
└── build_full_system.m  Builds full_system.slx programmatically
```

**Design principle:** Each module can be imported and run independently. `inverter.py __main__` designs the LC filter, constructs InverterParams from those designed values, simulates, and reports using the same parameter set  the design and simulation are always consistent.

#### Key class interfaces

```python
# pv_model.py
panel = PVPanel()
V, I, P       = panel.iv_curve(G=1000, T_cell=35)     # full curve
Vm, Im, Pm    = panel.mpp(G=1000, T_cell=35)           # MPP: 29.48V, 8.13A, 239.7W

# boost_converter.py
bc = BoostConverter()
D             = bc.duty_cycle(V_in=29.48, V_out=400)   # → 0.9263
eta           = bc.efficiency(V_in=29.48, I_in=8.13, V_out=400)  # → 0.9819

# mppt.py
algo = IncrementalConductance(InCondParams(delta_V=0.3))
V_ref_new     = algo.step(V=29.1, I=8.05)              # → updated reference

# inverter.py
filt = design_lc_filter(f_sw=10e3, P_rated=239.7, V_dc=400)
# filt.L_f = 13.91 mH, filt.C_f = 1.82 µF, filt.f_c = 1000 Hz
inv  = Inverter(InverterParams(L_f=filt.L_f, C_f=filt.C_f, ...))
res  = inv.simulate(t_end=0.40, dt=1e-6)               # → V_rms=230.45V, THD=1.02%

# power_quality.py
pq   = PowerQualityAnalyser(f_grid=50, n_harmonics=200)
spec = pq.analyse(res['V_out'], dt=1e-6)               # → HarmonicSpectrum
chk  = pq.ieee1547_check(spec_current, I_rated=I_rms)  # uses current spectrum
```

### 5.2 Data Flow: Irradiance to Grid

```python
# 1 PV at operating condition (T=35°C, G=1000 W/m²)
Vm, Im, Pm = panel.mpp(G=1000, T_cell=35)
# Vm=29.48V, Im=8.13A, Pm=239.7W

# 2 MPPT tracks voltage reference
V_ref = algo.step(V=Vm, I=Im)     # converges to Vm within a few cycles

# 3 Boost raises PV voltage to 400V DC
D = 1 - Vm/400          # = 0.9263  (ideal; PI corrects for losses)
# Boost losses: RL*IL² = 0.05*8.13² = 3.30W → η_boost = 98.19%

# 4 Inverter modulates DC → AC
Ma = 230*sqrt(2)/400     # = 0.8132 exactly  (NOT 325/400 = 0.8125)
# VAB(t) = (Sa - Sb) × 400V  →  {-400, 0, +400}V at 10kHz
# Key: all harmonic energy at 2*f_sw sidebands (H397-H401 = 19850-20050 Hz)
# NOT at low-order harmonics H3-H29 (those are < 0.25%)

# 5 LC filter removes 2*f_sw switching cluster
# Lf=13.91mH, Cf=1.82µF, fc=1000Hz, attenuation=-40dB at f_sw
# |H(50Hz)| ≈ 1.00003  →  V_rms = 230.45V (simulation)
# Note: 0.45V offset from ideal is SPWM quantisation artefact at dt=1µs, not a model error

# 6 Power quality
# 1.02% (Python, no dead-time) / 1.08% (Simulink, 2µs dead-time)
# IEEE 1547-2018: PASS (limit 5%, margin 4.9× Python / 4.6× Simulink)
# η_system = 98.35% (boost + inverter copper losses)
```

---

## 6. Validation and Results

### 6.1 Figure Guide and Plot Placement

| Figure | File | Section | Role |
|---|---|---|---|
| Fig 1  PV I-V & P-V | `fig01_pv_iv_curves.png`#fig1 | §3.1.4 | Validates single-diode model, I-V shape, and MPP locus |
| Fig 2  MPP locus | `fig02_mpp_locus.png`#fig2 | §3.1.5 | V_mpp and P_mpp vs G at two temperatures; marks operating point |
| Fig 3  MPPT step | `fig03_mppt_step.png`#fig3 | §3.3.4 | Quantifies P&O oscillation vs InCond stability |
| Fig 4  MPPT cloudy | `fig04_mppt_cloudy.png`#fig4 | §3.3.4 | Shows InCond advantage under fast irradiance changes |
| Fig 5  Boost char. | `fig05_boost_characteristics.png`#fig5 | §3.2.4 | Validates D(V_in), η(V_in), ΔI_L(V_in) curves |
| Fig 6  Inverter waveforms | `fig06_inverter_waveforms.png`#fig6 | §3.4.4 | Shows V_AB, V_out, I_L, I_out, P(t) time domain; 2ms zoom shows 3-level SPWM |
| Fig 7  Harmonic spectrum | `fig07_harmonic_spectrum.png`#fig7 | §3.5.4 | Continuous spectrum shows 2·f_sw cluster and filter attenuation; bar chart shows residual harmonics |
| Fig 8  Bode plot | `fig08_bode_plot.png`#fig8 | §3.5.5 | Confirms −40 dB/decade and f_c = 1 kHz |
| Fig 9  THD vs C_f | `fig09_thd_vs_capacitance.png`#fig9 | §3.5.6 | Demonstrates C_f optimisation with visible 5% compliance limit |
| Fig 10  IEEE 1547 | `fig10_ieee1547_compliance.png`#fig10 | §3.5.7 | Definitive compliance proof with individual harmonic margins |

### 6.2 System Performance Summary

| Subsystem | Metric | Python | Simulink | Spec | Status |
|---|---|---|---|---|---|
| **PV (STC)** | P_mpp | 250.2 W |  | 250 W ± 1 % | ✓ |
| **PV (operating)** | P_mpp at T=35°C | 239.7 W | 238.6 W |  | ✓ |
| **MPPT P&O** | η_overall (step) | 99.86 % |  | > 99 % | ✓ |
| **MPPT InCond** | η_overall (step) | 99.95 % |  | > 99 % | ✓ |
| **Boost** | V_out |  | 400.3 V | 400 V ± 1 % | ✓ |
| **Boost** | η | 98.2 % | 98.2 % | > 96 % | ✓ |
| **Boost** | D_ss | 0.9263 | 0.9263 | 1 − V_in/V_out | ✓ |
| **Boost** | ΔI_L | 0.683 A (8.4 %) |  | ≤ 15 % | ✓ |
| **Inverter** | V_rms | 230.45 V† | 230.0 V | 230 V | ✓ |
| **Inverter** | M_a | 0.8132 | 0.8132 | < 1.0 | ✓ |
| **Inverter** | P_out | 240.63 W | 238.6 W | ~240 W | ✓ |
| **Filter** | THD | 1.02 % | 1.08 % | < 5 % (IEEE) | ✓ |
| **Filter** | Attenuation at f_sw | −40.0 dB |  | ≥ −40 dB | ✓ |
| **Filter** | f_c | 1000 Hz | 1000 Hz | f_sw/10 | ✓ |
| **Filter** | L_f | 13.91 mH |  | design | ✓ |
| **Filter** | C_f | 1.82 µF |  | design | ✓ |
| **System** | η_chain |  | 98.35 % | > 96 % | ✓ |
| **Compliance** | IEEE 1547-2018 | PASS (4.9×) | PASS (4.6×) | < 5 % THD | ✓✓ |

† 0.45 V (0.20%) above target due to 1 µs SPWM switching-edge quantisation  see §1 note. Within EN 50160 ±1% grid tolerance.

### 6.3 Energy Conversion Chain Efficiency

At the operating condition (G = 1000 W/m², T = 35 °C), the full power chain achieves:

```
239.7 W  ← PV extracts from sunlight (P_pv at T=35°C, G=1000 W/m²)
  │
  ├─ MPPT loss:    −0.14 %  → 239.4 W  (algorithm perturbation, P&O)
  │
  ├─ Boost copper: −3.30 W  → 236.1 W  (RL_boost × IL²;  η_boost = 98.6 %)
  │
  ├─ Inverter ESR: −0.54 W  → 235.6 W  (RL_filter × Iout²; η_inv_filt = 99.8 %)
  │
  └─ Delivered to grid:  235.6 W  (η_chain = 235.6 / 239.7 = 98.29 %)
```

**Combined measured efficiency: η = 98.35 %** (Simulink display at t = 6 s).

> Note on context: This is the efficiency at a single operating point (full sun, T=35°C). Real-world annual energy yield varies significantly due to partial load, higher ambient temperatures, and intermittent irradiance, where typical field measurements achieve 85–92 % annual average for well-maintained systems.

### 6.4 Cross-Environment Consistency

Both Python and Simulink are solved from identical physical parameters and agree within the bounds of their respective approximations:

| Parameter | Python | Simulink | Residual | Cause |
|---|---|---|---|---|
| V_rms | 230.45 V | 230.0 V | 0.20 % | SPWM quantisation at dt=1µs (Python); ode23 + dead-time (Simulink) |
| THD | 1.02 % | 1.08 % | 0.06 pp | Dead-time absent in Python |
| Boost η | 98.2 % | 98.35 % | 0.15 pp | Different loss models |
| M_a | 0.8132 | 0.8132 | 0 | Unified definition 230√2/400 |
| f_c | 1000 Hz | 1000 Hz | 0 | Shared design_lc_filter() |
| L_f | 13.91 mH | 13.91 mH | 0 | Shared design |
| C_f | 1.82 µF | 1.82 µF | 0 | Shared design |
