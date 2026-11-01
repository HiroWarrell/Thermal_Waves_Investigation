# -*- coding: utf-8 -*-
"""
Thermal waves analysis.

Workflow
--------
1. For every file: select the last N_PERIODS periods (steady state), fit a
   sine to each thermistor, and get the amplitude ratio (gamma) and phase lag
   relative to thermistor 1, each with an uncertainty.
2. Files that are repeats of the same material and period (e.g. the three
   Brass rounds) are AVERAGED FIRST: gamma and phase lag are averaged
   thermistor-by-thermistor and the individual fit uncertainties are
   propagated through the mean.
3. D (diffusivity), kappa, reflection are fitted ONCE per material/period
   from the averaged data, and ONE set of plots is produced.
"""
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Interactive mode off: figures are only rendered when plt.show() is called,
# so each plot appears once (otherwise Spyder's inline backend can draw a
# figure on creation AND again on plt.show()).
plt.ioff()


# ============================================================
# SETTINGS
# ============================================================

delta_d = 0.005          # thermistor spacing (m)
L_TOP_OFFSET = 0.003     # distance from last thermistor to the far end (m)
N_PERIODS = 15           # number of final periods used for sine fitting
MIN_PERIODS = 5          # skip a file entirely if it has fewer periods than this
SHOW_PLOTS = True        # False = only save PNGs, never display

output_dir = Path("thermal_analysis_output")
plots_dir = output_dir / "thermal_plots"
output_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path):
    """Load logger data and extract the optional Comments field."""
    data = pd.read_csv(path, header=3)
    if data.shape[1] < 4:
        raise ValueError(
            f"Expected timestamp, voltage, current and at least one thermistor column: {path}"
        )
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=[data.columns[0]])
    timestamp = data.iloc[:, 0].to_numpy(dtype=float)
    voltage = data.iloc[:, 1].to_numpy(dtype=float)
    current = data.iloc[:, 2].to_numpy(dtype=float)
    thermistor_temperatures = data.iloc[:, 3:].to_numpy(dtype=float)
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        file_text = file.read()
    comments_match = re.search(
        r"Comments:\s*(.*)$", file_text, re.MULTILINE | re.IGNORECASE
    )
    comments = comments_match.group(1).strip() if comments_match else ""
    return timestamp, voltage, current, thermistor_temperatures, comments


# folder, filename, tau, D_guess
files_config = [
    # Brass (3 rounds)
    ("Brass", "5s 1.csv", 5.0, 3e-5),
    ("Brass", "10s 1.csv", 10.0, 3e-5),
    ("Brass", "15s 1.csv", 15.0, 3e-5),
    ("Brass", "20s 1.csv", 20.0, 3e-5),
    ("Brass", "25s 1.csv", 25.0, 3e-5),
    ("Brass", "30s 1.csv", 30.0, 3.5e-5),
    ("Brass", "60s 1.csv", 60.0, 5e-5),

    ("Brass", "5s 2.csv", 5.0, 3e-5),
    ("Brass", "10s 2.csv", 10.0, 3e-5),
    ("Brass", "15s 2.csv", 15.0, 3e-5),
    ("Brass", "20s 2.csv", 20.0, 3e-5),
    ("Brass", "25s 2.csv", 25.0, 3e-5),
    ("Brass", "30s 2.csv", 30.0, 3.5e-5),
    ("Brass", "60s 2.csv", 60.0, 5e-5),

    ("Brass", "5s 3.csv", 5.0, 3e-5),
    ("Brass", "10s 3.csv", 10.0, 3e-5),
    ("Brass", "15s 3.csv", 15.0, 3e-5),
    ("Brass", "20s 3.csv", 20.0, 3e-5),
    ("Brass", "25s 3.csv", 25.0, 3e-5),
    ("Brass", "30s 3.csv", 30.0, 3.5e-5),
    ("Brass", "60s 3.csv", 60.0, 5e-5),

    # Aluminium (1 round)
    ("Aluminium", "aluminium 5s.csv", 5.0, 8e-5),
    ("Aluminium", "aluminium 10s.csv", 10.0, 8e-5),
    ("Aluminium", "aluminium 15s.csv", 15.0, 8e-5),
    ("Aluminium", "aluminium 20s.csv", 20.0, 8e-5),
    ("Aluminium", "aluminium 30s.csv", 30.0, 8e-5),
    ("Aluminium", "aluminium 60s.csv", 60.0, 8e-5),

    # Steel (1 round)
    ("Steel", "steel 5s.csv", 5.0, 1.2e-5),
    ("Steel", "steel 10s.csv", 10.0, 1.2e-5),
    ("Steel", "steel 15s.csv", 15.0, 1.2e-5),
    ("Steel", "steel 20s.csv", 20.0, 1.2e-5),
    ("Steel", "steel 30s.csv", 30.0, 1.2e-5),
    ("Steel", "steel 60s.csv", 60.0, 1.2e-5),
]

material_properties = {
    "Brass": {"rho": 8500.0, "cp": 385.0},
    "Aluminium": {"rho": 2700.0, "cp": 900.0},
    "Steel": {"rho": 7850.0, "cp": 470.0},
}

# Results storage
results = []             # one row per material/period (averaged over rounds)
sensor_results = []      # per-thermistor results for each material/period
timepoint_results = []   # raw data, one row per timestamp per file


# ============================================================
# STAGE 1: SINE FITS FOR A SINGLE FILE
# ============================================================

def fit_file(folder, filename, tau, D_guess):
    """Select the last N_PERIODS, fit a sine to every thermistor and return
    amplitude ratio / phase lag (with uncertainties) for this one file."""
    file_path = os.path.join(folder, filename)
    timestamp, voltage, current, thermistor_temperatures, comments = load_dataset(file_path)

    round_match = re.search(r"(\d+)\.csv$", filename)
    round_number = int(round_match.group(1)) if round_match else 1

    # Save the complete raw recording (one row per time point)
    for row_idx, time_value in enumerate(timestamp):
        timepoint_row = {
            "Material": folder,
            "Filename": filename,
            "Round": round_number,
            "Period (s)": tau,
            "Timestamp (s)": time_value,
            "Voltage": voltage[row_idx],
            "Current": current[row_idx],
        }
        for sensor_idx in range(thermistor_temperatures.shape[1]):
            timepoint_row[f"Thermistor_{sensor_idx + 1}"] = (
                thermistor_temperatures[row_idx, sensor_idx]
            )
        timepoint_results.append(timepoint_row)

    # ---- data selection: last N_PERIODS full periods only ----
    omega = 2 * np.pi / tau
    t_end = timestamp[-1]
    available_periods = (t_end - timestamp[0]) / tau

    if available_periods < N_PERIODS:
        print(
            f"WARNING: {file_path} only contains {available_periods:.1f} "
            f"periods (< {N_PERIODS}); the transient may not be excluded."
        )
        if available_periods < MIN_PERIODS:
            raise ValueError(
                f"Only {available_periods:.1f} periods available "
                f"(minimum {MIN_PERIODS}); skipping."
            )
        mask = np.ones_like(timestamp, dtype=bool)
    else:
        mask = timestamp >= t_end - N_PERIODS * tau

    therm_data = thermistor_temperatures[mask]
    t = timestamp[mask] - timestamp[mask][0]

    # ---- sine fits ----
    def sine_model(t, T_0, C, phi):
        return T_0 + C * np.sin(omega * t + phi)

    amplitudes, amp_errs = [], []
    phases, phase_errs = [], []
    mean_temperatures = []

    for i in range(therm_data.shape[1]):
        y_data = therm_data[:, i]
        p0_guess = [
            np.mean(y_data),
            (np.max(y_data) - np.min(y_data)) / 2,
            0.0,
        ]
        popt, pcov = curve_fit(sine_model, t, y_data, p0=p0_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        T_0, C, phi = popt

        if C < 0:                       # make amplitude positive
            C = -C
            phi += np.pi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi)

        amplitudes.append(C)
        amp_errs.append(perr[1])
        phases.append(phi)
        phase_errs.append(perr[2])
        mean_temperatures.append(T_0)

    amplitudes = np.array(amplitudes)
    amp_errs = np.array(amp_errs)
    phases = np.array(phases)
    phase_errs = np.array(phase_errs)
    mean_temperatures = np.array(mean_temperatures)
    distances = np.arange(len(amplitudes)) * delta_d

    # ---- normalised amplitude ----
    if not np.isfinite(amplitudes[0]) or amplitudes[0] == 0:
        raise ValueError("Reference thermistor amplitude is zero or non-finite.")

    gamma = amplitudes / amplitudes[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_errs = np.abs(gamma) * np.sqrt(
            (amp_errs / amplitudes) ** 2 + (amp_errs[0] / amplitudes[0]) ** 2
        )
    gamma_errs[0] = 0.0   # reference thermistor has gamma = 1 exactly

    # ---- phase lag ----
    phase_lag = np.unwrap(phases[0] - phases)
    phase_lag_errs = np.sqrt(phase_errs[0] ** 2 + phase_errs ** 2)

    return {
        "folder": folder,
        "filename": filename,
        "round": round_number,
        "tau": tau,
        "omega": omega,
        "D_guess": D_guess,
        "comments": comments,
        "distances": distances,
        "amplitudes": amplitudes,
        "amp_errs": amp_errs,
        "mean_temperatures": mean_temperatures,
        "gamma": gamma,
        "gamma_errs": gamma_errs,
        "phase_lag": phase_lag,
        "phase_lag_errs": phase_lag_errs,
        "phases": phases,
        "phase_errs": phase_errs,
    }


# ============================================================
# STAGE 2: AVERAGE REPEATS (propagating uncertainties)
# ============================================================

def combine_rounds(fits):
    """Average repeats of the same material/period thermistor-by-thermistor.

    For N rounds with individual uncertainties s_i, the uncertainty on the
    mean is  s_mean = sqrt(sum(s_i^2)) / N.
    The scatter between rounds (SEM) is also stored for comparison.
    """
    n = len(fits)
    n_sensors = min(len(f["distances"]) for f in fits)
    if any(len(f["distances"]) != n_sensors for f in fits):
        print(f"WARNING: rounds have different thermistor counts; "
              f"using the first {n_sensors}.")

    def stack(key):
        return np.vstack([np.asarray(f[key])[:n_sensors] for f in fits])

    def mean_and_propagated(value_key, err_key):
        values, errs = stack(value_key), stack(err_key)
        mean = values.mean(axis=0)
        prop = np.sqrt((errs ** 2).sum(axis=0)) / n
        return mean, prop

    def repeat_sem(values):
        if n < 2:
            return np.full(values.shape[1], np.nan)
        return values.std(axis=0, ddof=1) / np.sqrt(n)

    gamma, gamma_errs = mean_and_propagated("gamma", "gamma_errs")

    # Align each round's unwrapped phase lag to the first round (2*pi branches)
    lags = stack("phase_lag")
    lags = lags + 2 * np.pi * np.round((lags[0] - lags) / (2 * np.pi))
    phase_lag = lags.mean(axis=0)
    phase_lag_errs = np.sqrt((stack("phase_lag_errs") ** 2).sum(axis=0)) / n

    amplitudes, amp_errs = mean_and_propagated("amplitudes", "amp_errs")

    first = fits[0]
    comments = "; ".join(dict.fromkeys(f["comments"] for f in fits if f["comments"]))

    return {
        "folder": first["folder"],
        "tau": first["tau"],
        "omega": first["omega"],
        "D_guess": first["D_guess"],
        "n_rounds": n,
        "rounds": [f["round"] for f in fits],
        "filenames": [f["filename"] for f in fits],
        "comments": comments,
        "distances": first["distances"][:n_sensors],
        "amplitudes": amplitudes,
        "amp_errs": amp_errs,
        "mean_temperatures": stack("mean_temperatures").mean(axis=0),
        "gamma": gamma,
        "gamma_errs": gamma_errs,
        "gamma_repeat_sem": repeat_sem(stack("gamma")),
        "phase_lag": phase_lag,
        "phase_lag_errs": phase_lag_errs,
        "phase_lag_repeat_sem": repeat_sem(lags),
        # Raw phases are relative to each file's own window start, so they
        # are not meaningful to average. Only kept for single-round groups.
        "phases": first["phases"] if n == 1 else np.full(n_sensors, np.nan),
        "phase_errs": first["phase_errs"] if n == 1 else np.full(n_sensors, np.nan),
    }


# ============================================================
# REFLECTED WAVE MODELS
# ============================================================

def make_models(omega, L_top):
    def spatial_profile(x, D):
        alpha = np.sqrt(omega / (2 * D))
        beta = alpha * (-1.0 + 1.0j)
        C_minus_factor = np.exp(2.0 * beta * L_top)
        return np.exp(beta * x) + C_minus_factor * np.exp(-beta * x)

    def gamma_model(x, D):
        return np.abs(spatial_profile(np.atleast_1d(x), D)) / np.abs(spatial_profile(0.0, D))

    def phase_model(x, D):
        X_x = spatial_profile(np.atleast_1d(x), D)
        X_0 = spatial_profile(0.0, D)
        return np.unwrap(np.angle(X_x) - np.angle(X_0))

    return gamma_model, phase_model


# ============================================================
# PLOT HELPER (each figure is saved once and displayed once)
# ============================================================

def finish_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ============================================================
# STAGE 3: FIT D, kappa, reflection, AND PLOT (once per material/period)
# ============================================================

def process_group(folder, tau, fits):
    g = combine_rounds(fits)
    omega = g["omega"]
    distances = g["distances"]
    gamma, gamma_errs = g["gamma"], g["gamma_errs"]
    phase_lag, phase_lag_errs = g["phase_lag"], g["phase_lag_errs"]
    D_guess = g["D_guess"]
    n_rounds = g["n_rounds"]

    L_top = distances[-1] + L_TOP_OFFSET
    gamma_model, phase_model = make_models(omega, L_top)

    # ---- fit thermal diffusivity ----
    popt_g, pcov_g = curve_fit(
        gamma_model, distances, gamma, p0=[D_guess],
        bounds=(1e-7, 1e-3), maxfev=10000
    )
    D_gamma, dD_gamma = popt_g[0], np.sqrt(np.diag(pcov_g))[0]

    popt_p, pcov_p = curve_fit(
        phase_model, distances, phase_lag, p0=[D_guess],
        bounds=(1e-7, 1e-3), maxfev=10000
    )
    D_phase, dD_phase = popt_p[0], np.sqrt(np.diag(pcov_p))[0]

    # ---- thermal conductivity ----
    rho = material_properties[folder]["rho"]
    cp = material_properties[folder]["cp"]
    kappa_gamma, dkappa_gamma = D_gamma * rho * cp, dD_gamma * rho * cp
    kappa_phase, dkappa_phase = D_phase * rho * cp, dD_phase * rho * cp

    # ---- reflection coefficient (with propagated uncertainty) ----
    # R = 100*exp(-2*alpha*L),  alpha = sqrt(omega/(2D))
    # dR/dD = R*L*alpha/D   ->   dR = R*L*alpha*dD/D
    alpha_val = np.sqrt(omega / (2 * D_gamma))
    reflection = np.exp(-2.0 * alpha_val * L_top) * 100.0
    reflection_err = reflection * L_top * alpha_val * dD_gamma / D_gamma

    # ---- fit statistics ----
    gamma_fit = gamma_model(distances, D_gamma)
    phase_fit = phase_model(distances, D_phase)
    residuals_gamma = gamma - gamma_fit
    residuals_phase = phase_lag - phase_fit
    dof = len(distances) - 1
    red_chi2_gamma = np.sum(residuals_gamma ** 2) / dof if dof > 0 else np.nan
    red_chi2_phase = np.sum(residuals_phase ** 2) / dof if dof > 0 else np.nan

    # ---- labels ----
    if n_rounds > 1:
        round_label = "Mean of " + ", ".join(str(r) for r in g["rounds"])
        plot_name = f"{folder}_{tau}s_mean.png"
        title_suffix = f"mean of {n_rounds} rounds"
    else:
        round_label = g["rounds"][0]
        plot_name = f"{folder}_{tau}s_Round{g['rounds'][0]}.png"
        title_suffix = f"Round {g['rounds'][0]}"

    # ---- save summary ----
    results.append({
        "Material": folder,
        "Filename": "; ".join(g["filenames"]),
        "Round": round_label,
        "Number of Rounds Averaged": n_rounds,
        "Period (s)": tau,
        "Angular Frequency (rad/s)": omega,
        "Number of Thermistors": len(distances),
        "Measurement Length (m)": distances[-1],
        "L_top (m)": L_top,
        "D_guess (m^2/s)": D_guess,
        "D_gamma (m^2/s)": D_gamma,
        "D_gamma_uncertainty (m^2/s)": dD_gamma,
        "D_phase (m^2/s)": D_phase,
        "D_phase_uncertainty (m^2/s)": dD_phase,
        "kappa_gamma (W/mK)": kappa_gamma,
        "kappa_gamma_uncertainty (W/mK)": dkappa_gamma,
        "kappa_phase (W/mK)": kappa_phase,
        "kappa_phase_uncertainty (W/mK)": dkappa_phase,
        "Reflection (%)": reflection,
        "Reflection_uncertainty (%)": reflection_err,
        "Reduced Chi2 Gamma": red_chi2_gamma,
        "Reduced Chi2 Phase": red_chi2_phase,
        "Density (kg/m^3)": rho,
        "Specific Heat (J/kgK)": cp,
        "Comments": g["comments"],
    })

    # ---- save per-thermistor results ----
    repeat_sem_g = g.get("gamma_repeat_sem", np.full(len(distances), np.nan))
    repeat_sem_p = g.get("phase_lag_repeat_sem", np.full(len(distances), np.nan))
    for i in range(len(distances)):
        sensor_results.append({
            "Material": folder,
            "Filename": "; ".join(g["filenames"]),
            "Round": round_label,
            "Number of Rounds Averaged": n_rounds,
            "Period (s)": tau,
            "Thermistor": i + 1,
            "Distance (m)": distances[i],
            "Distance (mm)": distances[i] * 1000,
            "Mean Temperature": g["mean_temperatures"][i],
            "Amplitude": g["amplitudes"][i],
            "Amplitude Uncertainty": g["amp_errs"][i],
            "Phase (rad)": g["phases"][i],
            "Phase Uncertainty (rad)": g["phase_errs"][i],
            "Amplitude Ratio": gamma[i],
            "Amplitude Ratio Uncertainty": gamma_errs[i],
            "Amplitude Ratio Repeat SEM": repeat_sem_g[i],
            "Amplitude Ratio Fit": gamma_fit[i],
            "Phase Lag (rad)": phase_lag[i],
            "Phase Lag Uncertainty (rad)": phase_lag_errs[i],
            "Phase Lag Repeat SEM (rad)": repeat_sem_p[i],
            "Phase Lag Fit (rad)": phase_fit[i],
            "Amplitude Residual": residuals_gamma[i],
            "Phase Residual (rad)": residuals_phase[i],
        })

    # ---- plotting (ONE figure per material/period) ----
    x_space = np.linspace(0, distances[-1], 200)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{folder} - Thermal Waves Analysis (Tau = {tau}s, {title_suffix})",
        fontsize=15, fontweight="bold"
    )

    axes[0].errorbar(
        distances * 1000, gamma, yerr=gamma_errs,
        fmt="x", capsize=4, markersize=7, label="Experimental Data"
    )
    axes[0].plot(
        x_space * 1000, gamma_model(x_space, D_gamma), linewidth=2,
        label=f"Fit: D = {D_gamma:.3e} m²/s"
    )
    axes[0].set_xlabel("Distance (mm)", fontsize=12)
    axes[0].set_ylabel("Amplitude Ratio (γ)", fontsize=12)
    axes[0].set_title("Attenuation vs Distance", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        distances * 1000, phase_lag, yerr=phase_lag_errs,
        fmt="x", capsize=4, markersize=7, label="Experimental Data"
    )
    axes[1].plot(
        x_space * 1000, phase_model(x_space, D_phase), linewidth=2,
        label=f"Fit: D = {D_phase:.3e} m²/s"
    )
    axes[1].set_xlabel("Distance (mm)", fontsize=12)
    axes[1].set_ylabel("Phase Lag (Δφ) [rad]", fontsize=12)
    axes[1].set_title("Phase Lag vs Distance", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    finish_figure(fig, plots_dir / plot_name)

    print(
        f"Completed: {folder} | tau = {tau}s | {n_rounds} round(s) | "
        f"D_gamma = {D_gamma:.3e} m²/s"
    )


# ============================================================
# MAIN LOOP
# ============================================================

# Stage 1: sine-fit every file, grouped by (material, period)
groups = {}
for folder, filename, tau, D_guess in files_config:
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        print(f"Skipping missing file: {file_path}")
        continue
    try:
        fit = fit_file(folder, filename, tau, D_guess)
        groups.setdefault((folder, tau), []).append(fit)
    except Exception as e:
        print(f"ERROR processing {file_path}: {e}")

# Stage 2 + 3: average repeats, fit D once, plot once
for (folder, tau), fits in groups.items():
    try:
        process_group(folder, tau, fits)
    except Exception as e:
        print(f"ERROR analysing {folder} tau={tau}s: {e}")


# ============================================================
# SUMMARY DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(results)
sensor_df = pd.DataFrame(sensor_results)
timepoint_df = pd.DataFrame(timepoint_results)

if not summary_df.empty:
    brass_summary_df = summary_df.loc[summary_df["Material"] == "Brass"].copy()
else:
    brass_summary_df = pd.DataFrame()


# ============================================================
# EXPORT CSV FILES (overwrite existing; handle locked files)
# ============================================================

def save_csv_overwrite(df, path, retries=3, wait=2.0):
    """Overwrite a CSV. If it is locked (e.g. open in Excel), retry,
    then fall back to a timestamped copy instead of crashing."""
    path = Path(path)
    for attempt in range(1, retries + 1):
        try:
            df.to_csv(path, index=False, float_format="%.8g")
            print(f"Saved (overwritten): {path}")
            return path
        except PermissionError:
            print(
                f"WARNING: {path.name} is locked (probably open in Excel). "
                f"Attempt {attempt}/{retries}; close it to let me overwrite."
            )
            time.sleep(wait)

    fallback = path.with_name(
        f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    )
    df.to_csv(fallback, index=False, float_format="%.8g")
    print(f"Could not overwrite {path.name}; saved to {fallback.name} instead.")
    return fallback


summary_csv = save_csv_overwrite(summary_df, output_dir / "thermal_summary.csv")
sensor_csv = save_csv_overwrite(sensor_df, output_dir / "thermal_sensor_fit_data.csv")
timepoint_csv = save_csv_overwrite(timepoint_df, output_dir / "thermal_sensor_data.csv")
brass_csv = save_csv_overwrite(brass_summary_df, output_dir / "brass_averaged_summary.csv")


# ============================================================
# OVERARCHING BRASS PLOTS: AVERAGED-ROUND RESULTS VS FREQUENCY
# ============================================================
# One point per period, from the fit to the round-averaged data. Error bars
# are the fit uncertainties (propagated from the averaged data). A straight
# line least-squares fit is overlaid. Frequency is f = 1 / period.

def make_brass_overview_plot(dataframe, value_col, err_col, y_label,
                             title, filename, fit_label="Linear best fit"):
    if dataframe.empty or value_col not in dataframe.columns:
        print(f"Skipping Brass overview plot: no data for {value_col}")
        return

    plot_data = dataframe[["Period (s)", value_col, err_col]].copy()
    plot_data["Frequency (Hz)"] = 1.0 / pd.to_numeric(plot_data["Period (s)"], errors="coerce")
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
    plot_data[err_col] = pd.to_numeric(plot_data[err_col], errors="coerce")
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
    plot_data = plot_data.dropna(subset=["Frequency (Hz)", value_col])
    plot_data = plot_data.sort_values("Frequency (Hz)")

    if plot_data.empty:
        print(f"Skipping Brass overview plot: no finite values for {value_col}")
        return

    x = plot_data["Frequency (Hz)"].to_numpy(dtype=float)
    y = plot_data[value_col].to_numpy(dtype=float)
    yerr = np.maximum(plot_data[err_col].fillna(0.0).to_numpy(dtype=float), 0.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(
        x, y, yerr=yerr, fmt="o", markersize=7, capsize=5,
        elinewidth=1.4, linewidth=1.4,
        label="Brass (rounds averaged) ± propagated uncertainty"
    )

    if len(x) >= 2 and np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 250)
        ax.plot(x_fit, slope * x_fit + intercept, "--", linewidth=2,
                label=f"{fit_label}: y = {slope:.3g}x {intercept:+.3g}")

        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.text(
            0.03, 0.97, f"Linear fit R² = {r_squared:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8}
        )

    ax.set_xlabel("Frequency, f = 1 / period (Hz)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    fig.tight_layout()

    save_path = plots_dir / filename
    finish_figure(fig, save_path)
    print(f"Saved Brass overview plot: {save_path.resolve()}")


if not brass_summary_df.empty:
    overview_plot_specs = [
        ("D_gamma (m^2/s)", "D_gamma_uncertainty (m^2/s)",
         "Thermal diffusivity, Dγ (m²/s)",
         "Brass: Amplitude-derived diffusivity vs frequency",
         "Brass_Dgamma_vs_frequency.png"),
        ("D_phase (m^2/s)", "D_phase_uncertainty (m^2/s)",
         "Thermal diffusivity, Dφ (m²/s)",
         "Brass: Phase-derived diffusivity vs frequency",
         "Brass_Dphase_vs_frequency.png"),
        ("kappa_gamma (W/mK)", "kappa_gamma_uncertainty (W/mK)",
         "Thermal conductivity, κγ (W m⁻¹ K⁻¹)",
         "Brass: Amplitude-derived thermal conductivity vs frequency",
         "Brass_kappa_gamma_vs_frequency.png"),
        ("kappa_phase (W/mK)", "kappa_phase_uncertainty (W/mK)",
         "Thermal conductivity, κφ (W m⁻¹ K⁻¹)",
         "Brass: Phase-derived thermal conductivity vs frequency",
         "Brass_kappa_phase_vs_frequency.png"),
        ("Reflection (%)", "Reflection_uncertainty (%)",
         "Reflection coefficient (%)",
         "Brass: Reflection coefficient vs frequency",
         "Brass_reflection_vs_frequency.png"),
    ]
    for value_col, err_col, y_label, title, filename in overview_plot_specs:
        make_brass_overview_plot(
            brass_summary_df, value_col, err_col, y_label, title, filename
        )
else:
    print("No Brass results available; overarching Brass plots were not generated.")


# ============================================================
# DISPLAY SUMMARY TABLE
# ============================================================

if not summary_df.empty:
    display_df = summary_df.copy()

    for column in [
        "D_gamma (m^2/s)", "D_gamma_uncertainty (m^2/s)",
        "D_phase (m^2/s)", "D_phase_uncertainty (m^2/s)",
        "kappa_gamma (W/mK)", "kappa_gamma_uncertainty (W/mK)",
        "kappa_phase (W/mK)", "kappa_phase_uncertainty (W/mK)",
        "Reduced Chi2 Gamma", "Reduced Chi2 Phase",
    ]:
        display_df[column] = display_df[column].map(lambda x: f"{x:.3e}")

    for column in ["Reflection (%)", "Reflection_uncertainty (%)"]:
        display_df[column] = display_df[column].map(lambda x: f"{x:.2f}%")

    display_df["Period (s)"] = display_df["Period (s)"].map(lambda x: f"{x:.1f}")

    print("\n")
    print("=" * 120)
    print("THERMAL WAVES ANALYSIS - SUMMARY RESULTS")
    print("=" * 120)
    print(display_df.to_string(index=False, justify="center"))
    print("=" * 120)


# ============================================================
# DISPLAY INDIVIDUAL SENSOR DATA
# ============================================================

if not sensor_df.empty:
    print("\n")
    print("=" * 100)
    print("INDIVIDUAL THERMISTOR DATA")
    print("=" * 100)
    print(f"Rows in per-thermistor fit table: {len(sensor_df)}")
    print(sensor_df.head(12).to_string(index=False))
    if len(sensor_df) > 12:
        print("... remaining rows are available in the exported CSV.")
    print("=" * 100)


# ============================================================
# FINAL OUTPUT LOCATIONS
# ============================================================

print("\n")
print("ANALYSIS COMPLETE")
print("-" * 60)
print(f"Summary CSV: {Path(summary_csv).resolve()}")
print(f"Raw sensor time-point CSV: {Path(timepoint_csv).resolve()}")
print(f"Per-thermistor fit CSV: {Path(sensor_csv).resolve()}")
print(f"Brass averaged summary CSV: {Path(brass_csv).resolve()}")
print(f"Plots directory: {plots_dir.resolve()}")
print("-" * 60)