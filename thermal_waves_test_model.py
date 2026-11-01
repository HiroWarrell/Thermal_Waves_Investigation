# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:21:09 2026

@author: haw25
"""
import pandas as pd
import re
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def load_dataset(path):    
    data = pd.read_csv(path, header=3)    
    timestamp = data.iloc[:, 0].to_numpy()    
    voltage = data.iloc[:, 1].to_numpy()    
    current = data.iloc[:, 2].to_numpy()    
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()    
    comments = re.search(r"Comments: (.*)$", open(path).read(), re.MULTILINE)[1]    
    return timestamp, voltage, current, thermistor_temperatures, comments

# Organized directory structure mapping: folder, filename, tau, D_guess
files_config = [
    # Brass (3 rounds across periods 5, 10, 15, 20, 25, 30, 60)
    # Note: Adjust filenames or loop logic here to match your exact naming format
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

delta_d = 0.005
results = []

rho_brass = 8500.0  # Density in kg/m^3
c_v_brass = 385.0   # Specific heat capacity in J/(kg*K)

for folder, filename, tau, D_guess in files_config:
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        print(f"Skipping missing file: {file_path}")
        continue
        
    timestamp, voltage, current, thermistor_temperatures, comments = load_dataset(file_path)
    
    # Use only the last 15 time periods of the dataset
    # 1 period corresponds to 'tau' seconds. We estimate indices based on sample rate or rows if needed, 
    # but here we slice the tail end of the thermistor temperature array by rows corresponding to last 15 periods.
    # Assuming standard sampling, let's determine rows per period or directly slice the last portion.
    # Alternatively, if you mean last 15 oscillation cycles: cycles * (tau / dt). 
    # If you meant the last 15 rows/entries, adjust accordingly. Here we slice last N rows for 15 periods:
    # Let's approximate rows per period using time step. If time step is uniform, dt = timestamp[1] - timestamp[0]
    dt = timestamp[1] - timestamp[0] if len(timestamp) > 1 else 0.1
    rows_per_period = int(tau / dt) if dt > 0 else 100
    target_rows = max(15 * rows_per_period, 15) # fallback safeguard
    
    if len(thermistor_temperatures) > target_rows:
        therm_data = thermistor_temperatures[-target_rows:]
        t = timestamp[-target_rows:] - timestamp[-target_rows]
    else:
        therm_data = thermistor_temperatures
        t = timestamp - timestamp[0]
    
    omega = 2 * np.pi / tau

    def sine_model(t, T_0, C, phi):
        return T_0 + C * np.sin(omega * t + phi)

    amplitudes = []
    amp_errs = []
    phases = []
    phase_errs = []
    distances = []
    
    for i in range(therm_data.shape[1]):
        y_data = therm_data[:, i]
        p0_guess = [np.mean(y_data), (np.max(y_data) - np.min(y_data)) / 2, 0.0]
        popt, pcov = curve_fit(sine_model, t, y_data, p0=p0_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov)) 
        
        T_0, C, phi = popt
        C_err = perr[1]
        phi_err = perr[2]
        
        if C < 0:
            C = -C
            phi += np.pi
            
        amplitudes.append(C)
        amp_errs.append(C_err)
        phases.append(phi)
        phase_errs.append(phi_err)
        distances.append(i * delta_d)

    amplitudes = np.array(amplitudes)
    amp_errs = np.array(amp_errs)
    phases = np.array(phases)
    phase_errs = np.array(phase_errs)
    distances = np.array(distances)
    
    gamma = amplitudes / amplitudes[0]
    gamma_errs = gamma * np.sqrt((amp_errs / amplitudes)**2 + (amp_errs[0] / amplitudes[0])**2)
    
    phase_lag = phases[0] - phases
    phase_lag = np.unwrap(phase_lag)
    phase_lag_errs = np.sqrt(phase_errs[0]**2 + phase_errs**2)
    
    L_top = distances[-1] + 0.003
    
    def spatial_profile(x, D):
        alpha = np.sqrt(omega / (2 * D))
        beta = alpha * (-1.0 + 1.0j) 
        C_minus_factor = np.exp(2.0 * beta * L_top) 
        return np.exp(beta * x) + C_minus_factor * np.exp(-beta * x)

    def gamma_model(x, D):
        X_x = spatial_profile(x, D)
        X_0 = spatial_profile(0.0, D)
        return np.abs(X_x) / np.abs(X_0)

    def phase_model(x, D):
        x_arr = np.atleast_1d(x)
        phases_val = []
        X_0 = spatial_profile(0.0, D)
        for xi in x_arr:
            X_x = spatial_profile(xi, D)
            phases_val.append(np.angle(X_x) - np.angle(X_0))
        return np.unwrap(phases_val)

    popt_D_gamma, pcov_D_gamma = curve_fit(gamma_model, distances, gamma, p0=D_guess, bounds=(1e-7, 1e-3), maxfev=10000)
    D_gamma = popt_D_gamma[0]
    dD_gamma = np.sqrt(np.diag(pcov_D_gamma))[0]

    popt_D_phase, pcov_D_phase = curve_fit(phase_model, distances, phase_lag, p0=[D_guess], bounds=(1e-7, 1e-3), maxfev=10000)
    D_phase = popt_D_phase[0]
    dD_phase = np.sqrt(np.diag(pcov_D_phase))[0]

    kappa_gamma = D_gamma * rho_brass * c_v_brass

    alpha_val = np.sqrt(omega / (2 * D_gamma))
    reflection_coefficient = np.exp(-2.0 * alpha_val * L_top) * 100.0

    residuals_gamma = gamma - gamma_model(distances, D_gamma)
    ssr_gamma = np.sum(residuals_gamma**2)
    dof_gamma = len(distances) - 1
    red_chi2_gamma = ssr_gamma / dof_gamma if dof_gamma > 0 else np.nan

    results.append({
        "folder": folder, "file": filename, "tau": tau, "reflection_pct": reflection_coefficient,
        "D_gamma": D_gamma, "dD_gamma": dD_gamma, "kappa": kappa_gamma,
        "red_chi2_gamma": red_chi2_gamma
    })

    x_space = np.linspace(0, distances[-1], 100)
    gamma_curve = gamma_model(x_space, D_gamma)
    phase_curve = phase_model(x_space, D_phase)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"{folder} - Thermal Waves Analysis (Tau = {tau}s)", fontsize=14)
    
    plt.subplot(1, 2, 1)
    plt.errorbar(distances * 1000, gamma, yerr=gamma_errs, fmt='x', capsize=3, label='Data')
    plt.plot(x_space * 1000, gamma_curve, label=f'Fit D={D_gamma:.2e}')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Amplitude Ratio ($\gamma$)')
    plt.title('Attenuation vs Distance (Reflected Model)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.errorbar(distances * 1000, phase_lag, yerr=phase_lag_errs, fmt='x', capsize=3, label='Data')
    plt.plot(x_space * 1000, phase_curve, label=f'Fit D={D_phase:.2e}')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Phase Lag ($\Delta \phi$) [rad]')
    plt.title('Phase Lag vs Distance (Reflected Model)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

print("\n--- Advanced Summary Across All Folders & Files ---")
for res in results:
    print(f"[{res['folder']}] Tau: {res['tau']}s | Reflection: {res['reflection_pct']:.1f}% | D: {res['D_gamma']:.2e} ± {res['dD_gamma']:.2e} | Chi2: {res['red_chi2_gamma']:.2e}")