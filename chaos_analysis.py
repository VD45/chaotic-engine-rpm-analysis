# VD Version 1.3 (August 2024)
# For article "Chaotic Behavior in the Rotational Speed of Internal Combustion Engines" - Lomakin and co.

import numpy as np
import pandas as pd
from scipy.signal import correlate, welch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import nolds
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import psutil

# Set up logging for debugging and tracking
logging.basicConfig(filename='chaos_analysis.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(message)s')

# Define sine function for fitting
def sine_function(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

# Custom function to compute histogram bin edges
def custom_histogram_bin_edges(data, bins=50):
    data_min, data_max = data.min(), data.max()
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    return bin_edges

# Function to compute Average Mutual Information (AMI)
def compute_ami(data, max_lag, n_bins=50):
    data_discrete = np.digitize(data, bins=custom_histogram_bin_edges(data, bins=n_bins))
    ami_values = Parallel(n_jobs=-1)(delayed(mutual_info_score)(data_discrete[:-lag], data_discrete[lag:]) for lag in range(1, max_lag))
    return ami_values

# Function to determine optimal embedding dimension using Sample Entropy (SampEn)
def determine_embedding_dimension(data, max_dim):
    num_jobs = max(1, psutil.cpu_count(logical=False) // 2)
    sampen_values = Parallel(n_jobs=num_jobs)(delayed(nolds.sampen)(data, emb_dim=dim) for dim in range(1, max_dim + 1))
    return sampen_values

# Function to compute Lyapunov Exponent
def compute_lyap_r(data_chunk, emb_dim, tau, min_tsep):
    return nolds.lyap_r(data_chunk, emb_dim=emb_dim, tau=tau, min_tsep=min_tsep, debug_plot=False)

# Function to reconstruct the phase space
def phase_space_reconstruction(data, lag, dimension):
    n = len(data)
    reconstructed = np.array([data[i: i + dimension * lag: lag] for i in range(n - (dimension - 1) * lag)])
    return reconstructed

def save_results(filename, data):
    np.savetxt(filename, data)

# Main function to manage the analysis
def main():
    try:
        # Load the data from files
        datasets = {
            'a)': pd.read_csv('550.txt', sep='\s+', header=None).values.flatten(),
            'b)': pd.read_csv('1000.txt', sep='\s+', header=None).values.flatten(),
            'c)': pd.read_csv('2000.txt', sep='\s+', header=None).values.flatten(),
            'd)': pd.read_csv('3100.txt', sep='\s+', header=None).values.flatten(),
            'e)': pd.read_csv('high.txt', sep='\s+', header=None).values.flatten(),
            'f)': pd.read_csv('low.txt', sep='\s+', header=None).values.flatten(),
        }
        logging.info("Data loaded successfully.")

        for label, rpm in datasets.items():
            # Fit sine function to the RPM data
            combined_data = rpm
            x_data = np.linspace(0, 2 * np.pi, len(combined_data))
            initial_guess = [100, 0.01, 0, 550]
            popt, _ = curve_fit(sine_function, x_data, combined_data, p0=initial_guess, maxfev=10000)
            a, b, c, d = popt
            logging.info(f"Fitted parameters for {label}: a={a}, b={b}, c={c}, d={d}")

            # Plot the fitted sine function against the original RPM data
            plt.figure(figsize=(8, 6), dpi=300)
            plt.plot(rpm, color='grey', linewidth=0.5)
            y_fit = sine_function(x_data, a, b, c, d)
            plt.plot(y_fit, color='orange', linestyle='-', linewidth=2, label=f'{a:.2f} * sin({b:.5f} * x + {c:.2f}) + {d:.2f}')
            plt.title(f'Original RPM Data with Fitted Sine Function ({label})')
            plt.xlabel('Sample Index')
            plt.ylabel('RPM')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')
            plt.tight_layout()
            plt.savefig(f'original_rpm_with_sine_fit_{label}.png')
            plt.close()

            # Normalize each data series by subtracting the fitted sine function
            sine_fit = sine_function(np.linspace(0, 2 * np.pi, len(rpm)), a, b, c, d)
            rpm_normalized = rpm - sine_fit
            logging.info(f"Data normalized for {label}.")

            # Save normalized data
            pd.DataFrame(rpm_normalized).to_csv(f'{label}_normalized.txt', sep=' ', header=False, index=False)

            # Compute AMI and determine optimal time delay (tau)
            max_lag = 50
            ami_values = compute_ami(rpm_normalized, max_lag)
            tau = np.argmax(ami_values) + 1
            logging.info(f"Optimal time delay (tau) for {label}: {tau}")

            # Phase Space Reconstruction
            dimension = 3
            phase_space = phase_space_reconstruction(rpm_normalized, tau, dimension)
            plt.figure(figsize=(8, 6), dpi=300)
            plt.plot(phase_space[:, 0], phase_space[:, 1], 'o', markersize=1, color='black')
            plt.title(f'Phase Space Reconstruction ({label})')
            plt.xlabel('x(t)')
            plt.ylabel(f'x(t + {tau})')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')
            plt.tight_layout()
            plt.savefig(f'phase_space_reconstruction_{label}.png')
            plt.close()

            # Compute Sample Entropy and determine embedding dimension (m)
            max_dim = 10
            sampen_values = determine_embedding_dimension(rpm_normalized, max_dim)
            m = np.argmin(sampen_values) + 1
            logging.info(f"Optimal embedding dimension (m) for {label}: {m}")

            # Compute and plot Lyapunov Exponent for data chunks
            data_chunks = np.array_split(rpm_normalized, 5)
            lyap_exps = Parallel(n_jobs=num_jobs)(delayed(compute_lyap_r)(chunk, emb_dim=m, tau=tau, min_tsep=500) for chunk in tqdm(data_chunks, desc=f"Computing Lyapunov Exponents for {label}"))
            plt.figure(figsize=(8, 6), dpi=300)
            plt.plot(lyap_exps, 'o-', color='black')
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f'Largest Lyapunov Exponent for Data Chunks ({label})')
            plt.xlabel('Data Chunk')
            plt.ylabel('Largest Lyapunov Exponent')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')
            plt.tight_layout()
            plt.savefig(f'lyapunov_exponent_chunks_{label}.png')
            plt.close()

            # Save results for each measure
            save_results(f'lyapunov_exponent_{label}.txt', lyap_exps)

            # Correlation Dimension Calculation
            corr_dims = [nolds.corr_dim(chunk, emb_dim=m) for chunk in data_chunks]
            plt.figure(figsize=(8, 6), dpi=300)
            plt.plot(corr_dims, 'o-', color='black')
            plt.title(f'Correlation Dimension for Data Chunks ({label})')
            plt.xlabel('Data Chunk')
            plt.ylabel('Correlation Dimension (D2)')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')
            plt.tight_layout()
            plt.savefig(f'correlation_dimension_chunks_{label}.png')
            plt.close()

            # Save correlation dimensions
            save_results(f'correlation_dimension_{label}.txt', corr_dims)

            # Power Spectral Density (PSD)
            f, Pxx_den = welch(rpm_normalized, fs=1.0, nperseg=512)
            plt.figure(figsize=(8, 6), dpi=300)
            plt.semilogy(f, Pxx_den, color='black')
            plt.title(f'Power Spectral Density (PSD) ({label})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')
            plt.tight_layout()
            plt.savefig(f'psd_{label}.png')
            plt.close()

            # Save PSD results
            save_results(f'psd_{label}.txt', np.column_stack((f, Pxx_den)))

    except Exception as e:
        logging.error("An error occurred", exc_info=True)

if __name__ == '__main__':
    main()

