# -*- coding: utf-8 -*-
"""
Thermal waves analysis, v2: standard and finite-length models in ONE run.

For every file: last N_PERIODS periods -> sine fit per thermistor -> amplitude
ratio (gamma) and phase lag relative to thermistor 1, with uncertainties.
Repeat rounds (brass) are averaged thermistor-by-thermistor, and each round
is ALSO fitted on its own, so repeatability can be reported.

Two models are fitted to the same data:
  standard : semi-infinite rod, no end effects
             gamma = exp(-alpha x),  lag = alpha x,   alpha = sqrt(omega/(2D))
  finite   : rod of length L with an insulated (no-flux) far end

Three estimates of D are produced for each model:
  D_gamma  from amplitude ratio only
  D_phase  from phase lag only
  D_joint  one D fitted to amplitude AND phase together (main result)

Fits are weighted (1/sigma). Parameter uncertainties are scaled by the
observed scatter (like curve_fit with absolute_sigma=False), so they do not
rely on the sine-fit uncertainties being correct in absolute terms.

NOTE ON "chi2_nu": it is a proper weighted chi-squared per degree of
freedom, but the sigmas are only approximate (sine-fit covariance assumes
independent samples, which is optimistic), so treat it as a RELATIVE measure
for comparing models on the same data, not an absolute goodness-of-fit test.
"""
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

# Figures only render on plt.show(), so each plot appears once.
plt.ioff()


# ============================================================
# SETTINGS
# ============================================================

delta_d = 0.005          # thermistor spacing (m)
L_TOP_OFFSET = 0.003     # last thermistor -> far end of the rod (m)
L_UNCERTAINTY = 0.001    # +/- uncertainty on that geometry (m), for a systematic on D
N_PERIODS = 15           # number of final periods used for sine fitting
MIN_PERIODS = 5          # skip a file if it has fewer periods than this
MIN_SNR = 5.0            # drop far thermistors whose amplitude / its uncertainty is below this
                         # (0 disables). Prevents noise-only sensors corrupting the phase unwrap.
SHOW_PLOTS = True        # False = only save PNGs
D_BOUNDS = (1e-7, 1e-3)  # allowed range for D (m^2/s)

MODELS = ("standard", "finite")
MODEL_LABELS = {
    "standard": "Standard (semi-infinite)",
    "finite": "Finite length (no-flux end)",
}

output_dir = Path("thermal_analysis_output_v2")
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


# folder, filename, tau, D_guess   (files that don't exist are skipped)
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
    ("Aluminium", "aluminium 25s.csv", 25.0, 8e-5),
    ("Aluminium", "aluminium 30s.csv", 30.0, 8e-5),
    ("Aluminium", "aluminium 40s.csv", 40.0, 8e-5),   # extra, not a matched condition
    ("Aluminium", "aluminium 60s.csv", 60.0, 8e-5),

    # Steel (1 round)
    ("Steel", "steel 5s.csv", 5.0, 1.2e-5),
    ("Steel", "steel 10s.csv", 10.0, 1.2e-5),
    ("Steel", "steel 15s.csv", 15.0, 1.2e-5),
    ("Steel", "steel 20s.csv", 20.0, 1.2e-5),
    ("Steel", "steel 25s.csv", 25.0, 1.2e-5),
    ("Steel", "steel 30s.csv", 30.0, 1.2e-5),
    ("Steel", "steel 60s.csv", 60.0, 1.2e-5),
]

material_properties = {
    "Brass": {"rho": 8500.0, "cp": 385.0},
    "Aluminium": {"rho": 2700.0, "cp": 900.0},
    "Steel": {"rho": 7850.0, "cp": 470.0},
}

# Results storage
results = []             # one row per material / period / model
sensor_results = []      # one row per material / period / model / thermistor
round_results = []       # one row per material / period / model / round (fits of each repeat)
timepoint_results = []   # raw data, one row per timestamp per file


# ============================================================
# MODELS
# ============================================================
# Convention: the harmonic field is Re[ X(x) exp(-i*omega*t) ], so the
# diffusion equation gives X'' = (-i*omega/D) X, i.e. exp(+/- alpha(1-i)x).
# The decaying (forward) solution is exp(beta x) with beta = alpha(-1+i),
# whose phase INCREASES with x -> arg X(x) - arg X(0) is the (positive) phase
# LAG. This matches the lag measured from sin(omega t + phi) fits as
# phi_0 - phi_i. (Using exp(+i*omega*t) instead gives the complex conjugate,
# i.e. the same lag with the opposite sign of the raw phase; the lag itself
# is convention-independent.) check_model_conventions() below verifies this.

def finite_profile(x, D, omega, L):
    """Forward wave + boundary-induced wave satisfying dX/dx = 0 at x = L."""
    alpha = np.sqrt(omega / (2 * D))
    beta = alpha * (-1.0 + 1.0j)
    return np.exp(beta * x) + np.exp(2.0 * beta * L) * np.exp(-beta * x)


def make_models(model, omega, L):
    if model == "standard":
        def gamma_model(x, D):
            return np.exp(-np.sqrt(omega / (2 * D)) * np.atleast_1d(x))

        def phase_model(x, D):
            return np.sqrt(omega / (2 * D)) * np.atleast_1d(x)
    elif model == "finite":
        def gamma_model(x, D):
            X = finite_profile(np.atleast_1d(x), D, omega, L)
            return np.abs(X) / np.abs(finite_profile(0.0, D, omega, L))

        def phase_model(x, D):
            X = finite_profile(np.atleast_1d(x), D, omega, L)
            return np.unwrap(np.angle(X) - np.angle(finite_profile(0.0, D, omega, L)))
    else:
        raise ValueError(model)
    return gamma_model, phase_model


def check_model_conventions():
    """The finite model must reduce to the standard model (positive lag) when
    the rod is much longer than the penetration depth."""
    D, omega, L = 3e-5, 2 * np.pi / 2.0, 0.5
    x = np.linspace(0.0, 0.02, 9)
    g_s, p_s = make_models("standard", omega, L)
    g_f, p_f = make_models("finite", omega, L)
    assert np.allclose(g_s(x, D), g_f(x, D), atol=1e-9), "gamma limit check failed"
    assert np.allclose(p_s(x, D), p_f(x, D), atol=1e-9), "phase limit check failed"
    assert p_f(x, D)[-1] > 0, "phase lag should be positive"
    print("Model self-check passed: finite model -> standard model in the long-rod limit.")


check_model_conventions()


# ============================================================
# STAGE 1: SINE FITS FOR A SINGLE FILE
# ============================================================

def fit_file(folder, filename, tau, D_guess):
    file_path = os.path.join(folder, filename)
    timestamp, voltage, current, thermistor_temperatures, comments = load_dataset(file_path)

    round_match = re.search(r"(\d+)\.csv$", filename)
    round_number = int(round_match.group(1)) if round_match else 1

    for row_idx, time_value in enumerate(timestamp):
        timepoint_row = {
            "Material": folder, "Filename": filename, "Round": round_number,
            "Period (s)": tau, "Timestamp (s)": time_value,
            "Voltage": voltage[row_idx], "Current": current[row_idx],
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

    amplitudes, amp_errs, phases, phase_errs, mean_temperatures = [], [], [], [], []
    for i in range(therm_data.shape[1]):
        y_data = therm_data[:, i]
        p0_guess = [np.mean(y_data), (np.max(y_data) - np.min(y_data)) / 2, 0.0]
        popt, pcov = curve_fit(sine_model, t, y_data, p0=p0_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        T_0, C, phi = popt
        if C < 0:
            C = -C
            phi += np.pi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
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
    n_total = len(amplitudes)
    distances = np.arange(n_total) * delta_d

    if not np.isfinite(amplitudes[0]) or amplitudes[0] == 0:
        raise ValueError("Reference thermistor amplitude is zero or non-finite.")

    # ---- which thermistors have a usable signal? ----
    # Thermistors beyond the first one with amplitude/uncertainty < MIN_SNR are
    # excluded from the model fits (their phases are noise and break unwrapping).
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = amplitudes / amp_errs
    bad = np.where(~(snr >= MIN_SNR))[0]
    n_usable = int(bad[0]) if bad.size else n_total
    if n_usable < n_total:
        print(f"NOTE: {file_path}: using first {n_usable} of {n_total} thermistors "
              f"(signal-to-noise < {MIN_SNR} beyond that).")
    if n_usable < 4:
        raise ValueError(f"Only {n_usable} usable thermistors; need at least 4.")

    # ---- normalised amplitude and phase lag ----
    gamma = amplitudes / amplitudes[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_errs = np.abs(gamma) * np.sqrt(
            (amp_errs / amplitudes) ** 2 + (amp_errs[0] / amplitudes[0]) ** 2
        )
    gamma_errs[0] = 0.0

    phase_lag = np.unwrap(phases[0] - phases)
    phase_lag_errs = np.sqrt(phase_errs[0] ** 2 + phase_errs ** 2)

    return {
        "folder": folder, "filename": filename, "round": round_number,
        "tau": tau, "omega": omega, "D_guess": D_guess, "comments": comments,
        "n_total": n_total, "n_usable": n_usable, "snr": snr,
        "distances": distances, "amplitudes": amplitudes, "amp_errs": amp_errs,
        "mean_temperatures": mean_temperatures,
        "gamma": gamma, "gamma_errs": gamma_errs,
        "phase_lag": phase_lag, "phase_lag_errs": phase_lag_errs,
        "phases": phases, "phase_errs": phase_errs,
    }


# ============================================================
# STAGE 2: COMBINE REPEAT ROUNDS
# ============================================================

def total_sigma(prop, sem, n_use, n_rounds):
    """sigma_total = sqrt(sigma_fit_propagated^2 + SEM_repeat^2).

    With only a few repeats the per-thermistor scatter is itself very
    uncertain (N=3 -> 2 degrees of freedom), so the repeat term is floored at
    half the median across thermistors. This stops one thermistor whose
    repeats happen to agree from dominating the weighted fit."""
    if n_rounds < 2:
        return prop.copy()
    floor = 0.5 * np.nanmedian(sem[1:n_use])
    sigma = np.sqrt(prop ** 2 + np.maximum(sem, floor) ** 2)
    sigma[0] = 0.0
    return sigma


def combine_rounds(fits):
    n = len(fits)
    n_total = min(f["n_total"] for f in fits)
    n_use = min(f["n_usable"] for f in fits)
    if any(f["n_total"] != n_total for f in fits):
        print(f"WARNING: rounds have different thermistor counts; using the first {n_total}.")

    def stack(key):
        return np.vstack([np.asarray(f[key])[:n_total] for f in fits])

    def propagated(err_key):
        return np.sqrt((stack(err_key) ** 2).sum(axis=0)) / n

    def repeat_sem(values):
        if n < 2:
            return np.full(values.shape[1], np.nan)
        return values.std(axis=0, ddof=1) / np.sqrt(n)

    gammas = stack("gamma")
    gamma = gammas.mean(axis=0)
    gamma_prop = propagated("gamma_errs")
    gamma_sem = repeat_sem(gammas)

    # align each round's unwrapped lag to round 1 (2*pi branches), then average
    lags = stack("phase_lag")
    lags = lags + 2 * np.pi * np.round((lags[0] - lags) / (2 * np.pi))
    lag = lags.mean(axis=0)
    lag_prop = propagated("phase_lag_errs")
    lag_sem = repeat_sem(lags)

    first = fits[0]
    return {
        "folder": first["folder"], "tau": first["tau"], "omega": first["omega"],
        "D_guess": first["D_guess"], "n_rounds": n,
        "rounds": [f["round"] for f in fits],
        "filenames": [f["filename"] for f in fits],
        "comments": "; ".join(dict.fromkeys(f["comments"] for f in fits if f["comments"])),
        "n_total": n_total, "n_use": n_use,
        "distances": first["distances"][:n_total],
        "amplitudes": stack("amplitudes").mean(axis=0), "amp_errs": propagated("amp_errs"),
        "mean_temperatures": stack("mean_temperatures").mean(axis=0),
        "gamma": gamma, "gamma_prop": gamma_prop, "gamma_sem": gamma_sem,
        "gamma_sigma": total_sigma(gamma_prop, gamma_sem, n_use, n),
        "lag": lag, "lag_prop": lag_prop, "lag_sem": lag_sem,
        "lag_sigma": total_sigma(lag_prop, lag_sem, n_use, n),
        "round_lags": lags,
        "fits": fits,
    }


# ============================================================
# STAGE 3: WEIGHTED FITS (gamma only, phase only, joint)
# ============================================================

def run_wls(resid_fn, D0, n_data):
    """Weighted least squares for one parameter D.

    resid_fn(D) returns (model - data)/sigma. The fit is done in units of
    D0 so the numerical derivatives are well scaled. The parameter variance
    is scaled by chi2_nu (absolute_sigma=False behaviour)."""
    sol = least_squares(
        lambda s: resid_fn(s[0] * D0), x0=[1.0],
        bounds=(D_BOUNDS[0] / D0, D_BOUNDS[1] / D0),
    )
    D = float(sol.x[0] * D0)
    chi2 = float(np.sum(sol.fun ** 2))
    dof = n_data - 1
    chi2_nu = chi2 / dof if dof > 0 else np.nan
    jtj = float((sol.jac.T @ sol.jac)[0, 0])
    dD = D0 * np.sqrt(chi2_nu / jtj) if (jtj > 0 and np.isfinite(chi2_nu)) else np.nan
    return D, dD, chi2_nu


def fit_all(model, omega, L, x, gamma, gamma_sig, lag, lag_sig, D0):
    """x and data exclude the reference thermistor (gamma = 1, lag = 0 by definition)."""
    gamma_model, phase_model = make_models(model, omega, L)
    gamma_sig = np.maximum(gamma_sig, 1e-12)
    lag_sig = np.maximum(lag_sig, 1e-12)

    def r_gamma(D):
        return (gamma_model(x, D) - gamma) / gamma_sig

    def r_phase(D):
        return (phase_model(x, D) - lag) / lag_sig

    def r_joint(D):
        return np.concatenate([r_gamma(D), r_phase(D)])

    Dg, dDg, cg = run_wls(r_gamma, D0, len(x))
    Dp, dDp, cp_ = run_wls(r_phase, D0, len(x))
    Dj, dDj, cj = run_wls(r_joint, D0, 2 * len(x))
    return {
        "D_gamma": Dg, "dD_gamma": dDg, "chi2_gamma": cg,
        "D_phase": Dp, "dD_phase": dDp, "chi2_phase": cp_,
        "D_joint": Dj, "dD_joint": dDj, "chi2_joint": cj,
    }


def mean_sd_sem(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    if v.size < 2:
        return m, np.nan, np.nan
    sd = float(v.std(ddof=1))
    return m, sd, sd / np.sqrt(v.size)


# ============================================================
# PLOT HELPER
# ============================================================

def finish_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ============================================================
# PER MATERIAL / PERIOD ANALYSIS
# ============================================================

MODEL_STYLE = {
    "standard": {"ls": "--", "color": "tab:orange", "marker": "s"},
    "finite": {"ls": "-", "color": "tab:blue", "marker": "o"},
}


def process_group(folder, tau, fits):
    g = combine_rounds(fits)
    omega = g["omega"]
    n_use, n_total, n_rounds = g["n_use"], g["n_total"], g["n_rounds"]
    D0 = g["D_guess"]
    L = (n_total - 1) * delta_d + L_TOP_OFFSET

    distances = g["distances"]
    idx = slice(1, n_use)                       # fitted thermistors (reference excluded)
    x = distances[idx]
    gamma, gamma_sig = g["gamma"][idx], g["gamma_sigma"][idx]
    lag, lag_sig = g["lag"][idx], g["lag_sigma"][idx]

    rho = material_properties[folder]["rho"]
    cp = material_properties[folder]["cp"]

    if n_rounds > 1:
        round_label = "Mean of " + ", ".join(str(r) for r in g["rounds"])
        plot_name = f"{folder}_{tau}s_mean.png"
        title_suffix = f"mean of {n_rounds} rounds"
    else:
        round_label = g["rounds"][0]
        plot_name = f"{folder}_{tau}s_Round{g['rounds'][0]}.png"
        title_suffix = f"Round {g['rounds'][0]}"

    fitted = {}
    for model in MODELS:
        fit = fit_all(model, omega, L, x, gamma, gamma_sig, lag, lag_sig, D0)
        fitted[model] = fit
        Dj = fit["D_joint"]

        # systematic from the uncertain rod length (finite model only)
        if model == "finite":
            Dj_shift = [
                fit_all(model, omega, L + dL, x, gamma, gamma_sig, lag, lag_sig, D0)["D_joint"]
                for dL in (-L_UNCERTAINTY, L_UNCERTAINTY)
            ]
            D_L_sys = max(abs(d - Dj) for d in Dj_shift)
        else:
            D_L_sys = np.nan

        # per-round fits (same model, fit-only sigmas) -> repeatability
        per_round = []
        if n_rounds > 1:
            for k, f in enumerate(g["fits"]):
                rf = fit_all(
                    model, omega, L, x,
                    f["gamma"][idx], f["gamma_errs"][idx],
                    g["round_lags"][k][idx], f["phase_lag_errs"][idx], D0,
                )
                per_round.append(rf)
                round_results.append({
                    "Material": folder, "Model": model, "Period (s)": tau,
                    "Round": f["round"], "Filename": f["filename"],
                    "D_gamma (m^2/s)": rf["D_gamma"], "D_gamma_uncertainty (m^2/s)": rf["dD_gamma"],
                    "D_phase (m^2/s)": rf["D_phase"], "D_phase_uncertainty (m^2/s)": rf["dD_phase"],
                    "D_joint (m^2/s)": rf["D_joint"], "D_joint_uncertainty (m^2/s)": rf["dD_joint"],
                    "chi2_nu gamma": rf["chi2_gamma"], "chi2_nu phase": rf["chi2_phase"],
                    "chi2_nu joint": rf["chi2_joint"],
                })

        stats = {}
        for key, name in [("D_gamma", "D_gamma"), ("D_phase", "D_phase"), ("D_joint", "D_joint")]:
            m, sd, sem = mean_sd_sem([r[key] for r in per_round])
            stats[f"{name}_round_mean (m^2/s)"] = m
            stats[f"{name}_round_SD (m^2/s)"] = sd
            stats[f"{name}_round_SEM (m^2/s)"] = sem

        # wave descriptors from the joint D
        alpha = np.sqrt(omega / (2 * Dj))
        x_last = distances[n_use - 1]
        if model == "finite":
            # |boundary-induced component| / |forward component| at x: exp(-2*alpha*(L - x))
            b0 = np.exp(-2 * alpha * L) * 100.0
            b_last = np.exp(-2 * alpha * (L - x_last)) * 100.0
            db0 = b0 * L * alpha * fit["dD_joint"] / Dj
            db_last = b_last * (L - x_last) * alpha * fit["dD_joint"] / Dj
        else:
            b0 = db0 = b_last = db_last = np.nan

        results.append({
            "Material": folder, "Model": model,
            "Filename": "; ".join(g["filenames"]), "Round": round_label,
            "Number of Rounds Averaged": n_rounds,
            "Period (s)": tau, "Angular Frequency (rad/s)": omega,
            "Thermistors Available": n_total, "Thermistors Used": n_use,
            "Measurement Length (m)": x_last, "Rod Length L (m)": L,
            "Penetration Depth 1/alpha (mm)": 1000.0 / alpha, "alpha*L": alpha * L,
            "D_guess (m^2/s)": D0,
            "D_gamma (m^2/s)": fit["D_gamma"], "D_gamma_uncertainty (m^2/s)": fit["dD_gamma"],
            "D_phase (m^2/s)": fit["D_phase"], "D_phase_uncertainty (m^2/s)": fit["dD_phase"],
            "D_joint (m^2/s)": Dj, "D_joint_uncertainty (m^2/s)": fit["dD_joint"],
            "D_joint_L_systematic (m^2/s)": D_L_sys,
            "kappa_gamma (W/mK)": fit["D_gamma"] * rho * cp,
            "kappa_gamma_uncertainty (W/mK)": fit["dD_gamma"] * rho * cp,
            "kappa_phase (W/mK)": fit["D_phase"] * rho * cp,
            "kappa_phase_uncertainty (W/mK)": fit["dD_phase"] * rho * cp,
            "kappa_joint (W/mK)": Dj * rho * cp,
            "kappa_joint_uncertainty (W/mK)": fit["dD_joint"] * rho * cp,
            "kappa_joint_L_systematic (W/mK)": D_L_sys * rho * cp,
            "chi2_nu gamma": fit["chi2_gamma"], "chi2_nu phase": fit["chi2_phase"],
            "chi2_nu joint": fit["chi2_joint"],
            "Boundary term at x=0 (%)": b0, "Boundary term at x=0 uncertainty (%)": db0,
            "Boundary term at last thermistor (%)": b_last,
            "Boundary term at last thermistor uncertainty (%)": db_last,
            **stats,
            "Density (kg/m^3)": rho, "Specific Heat (J/kgK)": cp,
            "Comments": g["comments"],
        })

        # per-thermistor table (fit curves use this model's joint D)
        gm, pm = make_models(model, omega, L)
        gamma_fit_all = gm(distances, Dj)
        lag_fit_all = pm(distances, Dj)
        for i in range(n_total):
            sensor_results.append({
                "Material": folder, "Model": model, "Round": round_label,
                "Number of Rounds Averaged": n_rounds, "Period (s)": tau,
                "Thermistor": i + 1, "Used in fit": bool(1 <= i < n_use),
                "Distance (mm)": distances[i] * 1000,
                "Mean Temperature": g["mean_temperatures"][i],
                "Amplitude": g["amplitudes"][i], "Amplitude Uncertainty": g["amp_errs"][i],
                "Amplitude Ratio": g["gamma"][i],
                "Amplitude Ratio Sigma (fit propagated)": g["gamma_prop"][i],
                "Amplitude Ratio Repeat SEM": g["gamma_sem"][i],
                "Amplitude Ratio Sigma (used in fit)": g["gamma_sigma"][i],
                "Amplitude Ratio Model (joint D)": gamma_fit_all[i],
                "Amplitude Ratio Residual": g["gamma"][i] - gamma_fit_all[i],
                "Phase Lag (rad)": g["lag"][i],
                "Phase Lag Sigma (fit propagated)": g["lag_prop"][i],
                "Phase Lag Repeat SEM": g["lag_sem"][i],
                "Phase Lag Sigma (used in fit)": g["lag_sigma"][i],
                "Phase Lag Model (joint D)": lag_fit_all[i],
                "Phase Lag Residual (rad)": g["lag"][i] - lag_fit_all[i],
            })

    # ---- ONE figure per material/period: both models, fits + residuals ----
    x_space = np.linspace(0, distances[-1], 300)
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.08, wspace=0.22)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_rg = fig.add_subplot(gs[1, 0], sharex=ax_g)
    ax_rp = fig.add_subplot(gs[1, 1], sharex=ax_p)
    fig.suptitle(f"{folder} - Thermal Waves (Tau = {tau}s, {title_suffix})",
                 fontsize=15, fontweight="bold")

    used = np.arange(n_total) < n_use
    for ax, y, yerr in [(ax_g, g["gamma"], g["gamma_sigma"]), (ax_p, g["lag"], g["lag_sigma"])]:
        ax.errorbar(distances[used] * 1000, y[used], yerr=yerr[used], fmt="k.", capsize=3,
                    markersize=8, label="Data")
        if (~used).any():
            ax.plot(distances[~used] * 1000, y[~used], "o", mfc="none", mec="grey",
                    label="Excluded (low SNR)")

    for model in MODELS:
        st = MODEL_STYLE[model]
        Dj = fitted[model]["D_joint"]
        gm, pm = make_models(model, omega, L)
        lab = f"{MODEL_LABELS[model]}: D = {Dj:.3e} m²/s"
        ax_g.plot(x_space * 1000, gm(x_space, Dj), st["ls"], color=st["color"], lw=2, label=lab)
        ax_p.plot(x_space * 1000, pm(x_space, Dj), st["ls"], color=st["color"], lw=2, label=lab)
        ax_rg.plot(distances[used] * 1000, (g["gamma"] - gm(distances, Dj))[used],
                   st["marker"], color=st["color"], mfc="none")
        ax_rp.plot(distances[used] * 1000, (g["lag"] - pm(distances, Dj))[used],
                   st["marker"], color=st["color"], mfc="none")

    for ax in (ax_rg, ax_rp):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Distance (mm)", fontsize=12)
        ax.grid(True, alpha=0.3)
    ax_rg.set_ylabel("Residual (data - model)")
    ax_rp.set_ylabel("Residual [rad]")
    ax_g.set_ylabel("Amplitude Ratio (γ)", fontsize=12)
    ax_p.set_ylabel("Phase Lag (Δφ) [rad]", fontsize=12)
    ax_g.set_title("Attenuation vs Distance", fontsize=13)
    ax_p.set_title("Phase Lag vs Distance", fontsize=13)
    for ax in (ax_g, ax_p):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)
    finish_figure(fig, plots_dir / plot_name)

    fj = fitted["finite"]["D_joint"]
    sj = fitted["standard"]["D_joint"]
    print(f"Completed: {folder} | tau = {tau}s | {n_rounds} round(s) | "
          f"D_joint finite = {fj:.3e}, standard = {sj:.3e} m²/s")


# ============================================================
# MAIN LOOP
# ============================================================

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

for (folder, tau), fits in groups.items():
    try:
        process_group(folder, tau, fits)
    except Exception as e:
        print(f"ERROR analysing {folder} tau={tau}s: {e}")


# ============================================================
# DATAFRAMES AND CSV EXPORT
# ============================================================

summary_df = pd.DataFrame(results)
sensor_df = pd.DataFrame(sensor_results)
round_df = pd.DataFrame(round_results)
timepoint_df = pd.DataFrame(timepoint_results)


def save_csv_overwrite(df, path, retries=3, wait=2.0):
    """Overwrite a CSV; if locked (e.g. open in Excel) retry, then save a timestamped copy."""
    path = Path(path)
    for attempt in range(1, retries + 1):
        try:
            df.to_csv(path, index=False, float_format="%.8g")
            print(f"Saved (overwritten): {path}")
            return path
        except PermissionError:
            print(f"WARNING: {path.name} is locked (probably open in Excel). "
                  f"Attempt {attempt}/{retries}; close it to let me overwrite.")
            time.sleep(wait)
    fallback = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    df.to_csv(fallback, index=False, float_format="%.8g")
    print(f"Could not overwrite {path.name}; saved to {fallback.name} instead.")
    return fallback


summary_csv = save_csv_overwrite(summary_df, output_dir / "thermal_summary.csv")
sensor_csv = save_csv_overwrite(sensor_df, output_dir / "thermal_sensor_fit_data.csv")
round_csv = save_csv_overwrite(round_df, output_dir / "thermal_per_round_fits.csv")
timepoint_csv = save_csv_overwrite(timepoint_df, output_dir / "thermal_sensor_data.csv")


# ============================================================
# MATERIAL SUMMARY PLOTS: D vs period for each model, and chi2_nu
# ============================================================

def plot_material_summary(df, material):
    d = df[df["Material"] == material]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    series = [
        ("D_gamma (m^2/s)", "D_gamma_uncertainty (m^2/s)", "o", "Dγ (amplitude)", 0.96),
        ("D_phase (m^2/s)", "D_phase_uncertainty (m^2/s)", "s", "Dφ (phase)", 1.00),
        ("D_joint (m^2/s)", "D_joint_uncertainty (m^2/s)", "D", "D joint", 1.04),
    ]
    all_vals = []
    for ax, model in zip(axes[:2], MODELS):
        m = d[d["Model"] == model].sort_values("Period (s)")
        if m.empty:
            continue
        for vcol, ecol, mk, lab, shift in series:
            ax.errorbar(m["Period (s)"] * shift, m[vcol], yerr=m[ecol], fmt=mk, capsize=3,
                        label=lab)
            all_vals.extend(m[vcol].dropna().tolist())
        if m["D_joint_round_SEM (m^2/s)"].notna().any():
            ax.errorbar(m["Period (s)"] * 1.08, m["D_joint_round_mean (m^2/s)"],
                        yerr=m["D_joint_round_SEM (m^2/s)"], fmt="^", color="k", capsize=3,
                        label="Mean of per-round D joint ± SEM")
        ax.set_xscale("log")
        ax.set_xlabel("Period (s)", fontsize=12)
        ax.set_title(MODEL_LABELS[model], fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Thermal diffusivity D (m²/s)", fontsize=12)
    if all_vals:   # share a sensible y-range, ignoring wild failed fits
        lo, hi = np.percentile(all_vals, [5, 95])
        pad = 0.5 * (hi - lo) + 0.1 * abs(hi)
        for ax in axes[:2]:
            ax.set_ylim(max(lo - pad, 0), hi + pad)
    for model in MODELS:
        m = d[d["Model"] == model].sort_values("Period (s)")
        if not m.empty:
            axes[2].plot(m["Period (s)"], m["chi2_nu joint"], MODEL_STYLE[model]["marker"] + "-",
                         color=MODEL_STYLE[model]["color"], label=MODEL_LABELS[model])
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Period (s)", fontsize=12)
    axes[2].set_ylabel("chi2_nu (joint fit)", fontsize=12)
    axes[2].set_title("Model comparison (relative)", fontsize=13)
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{material}: diffusivity vs period", fontsize=15, fontweight="bold")
    fig.tight_layout()
    finish_figure(fig, plots_dir / f"{material}_D_vs_period.png")
    print(f"Saved summary plot for {material}")


if not summary_df.empty:
    for material in summary_df["Material"].unique():
        plot_material_summary(summary_df, material)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

if not summary_df.empty:
    cols = [
        "Material", "Model", "Period (s)", "Number of Rounds Averaged", "Thermistors Used",
        "D_gamma (m^2/s)", "D_phase (m^2/s)", "D_joint (m^2/s)", "D_joint_uncertainty (m^2/s)",
        "chi2_nu joint", "kappa_joint (W/mK)",
    ]
    show = summary_df[cols].copy()
    for c in ["D_gamma (m^2/s)", "D_phase (m^2/s)", "D_joint (m^2/s)",
              "D_joint_uncertainty (m^2/s)", "chi2_nu joint"]:
        show[c] = show[c].map(lambda v: f"{v:.3e}")
    show["kappa_joint (W/mK)"] = show["kappa_joint (W/mK)"].map(lambda v: f"{v:.1f}")
    print("\n" + "=" * 130)
    print("THERMAL WAVES ANALYSIS v2 - SUMMARY (full tables in the CSV files)")
    print("=" * 130)
    print(show.to_string(index=False))
    print("=" * 130)

print("\nANALYSIS COMPLETE")
print("-" * 60)
print(f"Summary CSV: {Path(summary_csv).resolve()}")
print(f"Per-round fits CSV: {Path(round_csv).resolve()}")
print(f"Per-thermistor CSV: {Path(sensor_csv).resolve()}")
print(f"Raw time-point CSV: {Path(timepoint_csv).resolve()}")
print(f"Plots directory: {plots_dir.resolve()}")
print("-" * 60)
