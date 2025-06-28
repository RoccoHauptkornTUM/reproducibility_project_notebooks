import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

# Output folder
output_dir = "sc_output"
os.makedirs(output_dir, exist_ok=True)

# === Step 1: Monte Carlo Pi estimation ===

def estimate_pi(n_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    inside = (x**2 + y**2) <= 1
    n_inside = np.sum(inside)

    return 4 * n_inside / n_samples

pi_value = estimate_pi(10000, seed=42)
print(f"Estimated pi with 10,000 samples: {pi_value}")

# === Step 2: Convergence analysis ===

sample_sizes = [10**i for i in range(2,6)]
results = []

for n in sample_sizes:
    start_time = time.time()
    pi_est = estimate_pi(n, seed=42)
    elapsed = time.time() - start_time
    error = abs(pi_est - np.pi)

    results.append({
        "samples": n,
        "pi_estimate": pi_est,
        "error": error,
        "time_sec": elapsed
    })

    # Convert to Dataframe for convenience
    df_results = pd.DataFrame(results)

    print("\nSimulation results:")
    print(df_results)

# === Step 3: Plot convergence ===

# Plot: pi estimate vs sample size
plt.figure(figsize=(8, 5))
plt.plot(df_results["samples"], df_results["pi_estimate"], marker="o", label="Estimated Pi")
plt.axhline(np.pi, color="red", linestyle="--", label="True Pi")
plt.xscale("log")
plt.xlabel("Number of samples (log scale)")
plt.ylabel("Pi estimate")
plt.title("Monte Carlo Estimation of Pi vs. Sample Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pi_convergence.png"))
plt.close()

# Plot: error vs. sample size (log-log)
plt.figure(figsize=(8, 5))
plt.plot(df_results["samples"], df_results["error"], marker="o", color="orange")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of samples (log scale)")
plt.ylabel("Absolute error (log scale)")
plt.title("Error vs. Sample Size (Log-Log)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_convergence.png"))
plt.close()

# === Step 4: Repeat simulation for statistical error analysis ===

n_samples = 10000   # fixed sample size per run
n_trials = 1000      # number of reapeated simulations

estimates = []
for i in range(n_trials):
    pi_est = estimate_pi(n_samples, seed=i)
    estimates.append(pi_est)

estimates = np.array(estimates)
mean_pi = np.mean(estimates)
std_pi = np.std(estimates)
min_pi = np.min(estimates)
max_pi = np.max(estimates)

print(f"\nAfter {n_trials} trials:")
print(f"Mean Pi: {mean_pi:.6f}")
print(f"Std Dev: {std_pi:.6f}")
print(f"Min Pi: {min_pi:.6f}, Max Pi: {max_pi:.6f}")

# === Step 5: Plot histogram ===

plt.figure(figsize=(8, 5))
plt.hist(estimates, bins=15, color="skyblue", edgecolor="black")
plt.axvline(np.pi, color="red", linestyle="--", label="True Pi")
plt.axvline(mean_pi, color="green", linestyle="--", label="Mean estimate")
plt.xlabel("Estimated Pi")
plt.ylabel("Frequency")
plt.title(f"Histogram of Pi Estimates ({n_trials} trials, {n_samples} samples)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pi_estimate_histogram.png"))
plt.close()

# === Step 6: Save results to .txt ===

summary_path = os.path.join(output_dir, "simulation_summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Monte Carlo Pi Estimation Summary ===\n\n")
    
    f.write("Convergence Results:\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")

    f.write(f"Repeated Trials (n={n_trials}, samples={n_samples}):\n")
    f.write(f"Mean Pi: {mean_pi:.6f}\n")
    f.write(f"Std Dev: {std_pi:.6f}\n")
    f.write(f"Min Pi: {min_pi:.6f}, Max Pi: {max_pi:.6f}\n")