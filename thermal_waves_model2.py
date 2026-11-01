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

def load_dataset(path):    
    data = pd.read_csv(path, header=3)    
    timestamp = data.iloc[:, 0].to_numpy()    
    voltage = data.iloc[:, 1].to_numpy()    
    current = data.iloc[:, 2].to_numpy()    
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()    
    comments = re.search(r"Comments: (.*)$", open(path).read(), re.MULTILINE)[1]    
    return timestamp, voltage, current, thermistor_temperatures, comments

files = {"A7_tau10_1.csv": (10.0, 25300, 3e-5),"15_seconds.csv": (15.0, 1830, 3e-5), "20_seconds.csv": (20.0, 1830, 3e-5), "30_seconds.csv": (30.0, 2900, 3.5e-5), "60_seconds.csv": (60.0, 3900, 5e-5)}

delta_d = 0.005
results = []

for file_path, (tau, start_idx, D_guess) in files.items():
    timestamp, voltage, current, thermistor_temperatures, comments = load_dataset(file_path)
    t = timestamp[start_idx:] - timestamp[start_idx]
    therm_data = thermistor_temperatures[start_idx:]
    
    omega = 2 * np.pi / tau

    def sine_model(t, T_0, C, phi):
        return T_0 + C * np.sin(omega * t + phi)

    amplitudes = []
    phases = []
    distances = []
    
    for i in range(therm_data.shape[1]):
        y_data = therm_data[:, i]
        p0_guess = [np.mean(y_data), (np.max(y_data) - np.min(y_data)) / 2, 0.0]
        popt, _ = curve_fit(sine_model, t, y_data, p0=p0_guess, maxfev=10000)
        T_0, C, phi = popt
        
        if C < 0:
            C = -C
            phi += np.pi
            
        amplitudes.append(C)
        phases.append(phi)
        distances.append(i * delta_d)

    amplitudes = np.array(amplitudes)
    phases = np.array(phases)
    distances = np.array(distances)
    
    gamma = amplitudes / amplitudes[0]
    phase_lag = phases[0] - phases
    phase_lag = np.unwrap(phase_lag)
    
    # Define exact top boundary L (last thermistor position + 3mm top extension)
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

    popt_D_gamma, _ = curve_fit(gamma_model, distances, gamma, p0=D_guess, bounds=(1e-7, 1e-3), maxfev=10000)
    D_gamma = popt_D_gamma[0]

    popt_D_phase, _ = curve_fit(phase_model, distances, phase_lag, p0=[D_guess], bounds=(1e-7, 1e-3), maxfev=10000)
    D_phase = popt_D_phase[0]

    # Goodness-of-fit calculations (SSR & Reduced Chi-Squared)
    residuals_gamma = gamma - gamma_model(distances, D_gamma)
    ssr_gamma = np.sum(residuals_gamma**2)
    dof_gamma = len(distances) - 1
    red_chi2_gamma = ssr_gamma / dof_gamma if dof_gamma > 0 else np.nan

    residuals_phase = phase_lag - phase_model(distances, D_phase)
    ssr_phase = np.sum(residuals_phase**2)
    dof_phase = len(distances) - 1
    red_chi2_phase = ssr_phase / dof_phase if dof_phase > 0 else np.nan

    #print(f"--- Tau: {tau}s ---")
    #print(f"D (Attenuation): {D_gamma:.4e} | SSR: {ssr_gamma:.4e} | Red. Chi2: {red_chi2_gamma:.4e}")
    #print(f"D (Phase Lag):   {D_phase:.4e} | SSR: {ssr_phase:.4e} | Red. Chi2: {red_chi2_phase:.4e}")
    
    results.append({
        "file": file_path, "tau": tau, 
        "D_gamma": D_gamma, "ssr_gamma": ssr_gamma, "red_chi2_gamma": red_chi2_gamma,
        "D_phase": D_phase, "ssr_phase": ssr_phase, "red_chi2_phase": red_chi2_phase
    })

    x_space = np.linspace(0, distances[-1], 100)
    
    gamma_curve = gamma_model(x_space, D_gamma)
    phase_curve = phase_model(x_space, D_phase)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Thermal Waves Analysis with Reflections (Time Period = {tau}s)", fontsize=14)
    
    plt.subplot(1, 2, 1)
    plt.plot(distances * 1000, gamma, 'x', label='Data')
    plt.plot(x_space * 1000, gamma_curve, label=f'D={D_gamma:.2e}')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Amplitude Ratio ($\gamma$)')
    plt.title('Attenuation vs Distance (Reflected Model)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(distances * 1000, phase_lag, 'x', label='Data')
    plt.plot(x_space * 1000, phase_curve, label=f'D={D_phase:.2e}')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Phase Lag ($\Delta \phi$) [rad]')
    plt.title('Phase Lag vs Distance (Reflected Model)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

print("\n--- Summary Across All Files ---")
for res in results:
    print(f"Tau: {res['tau']}s | D_gamma: {res['D_gamma']:.2e} (Chi2: {res['red_chi2_gamma']:.2e}) | D_phase: {res['D_phase']:.2e} (Chi2: {res['red_chi2_phase']:.2e})")