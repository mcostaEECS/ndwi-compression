import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configure plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titleweight': 'bold',
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

# Hypothesis parameters (Gaussian distributions)
mu0, sigma0 = 0, 1   # H0: No anomaly (ρ = 0)
mu1, sigma1 = 3, 1   # H1: Anomaly present (ρ ≠ 0)
gamma = 1.96          # Detection threshold (q=0.95)

# Create distribution curves
x = np.linspace(-4, 7, 1000)
h0_curve = norm.pdf(x, mu0, sigma0)
h1_curve = norm.pdf(x, mu1, sigma1)

# Create figure
fig, ax = plt.subplots()

# Plot PDF curves
ax.plot(x, h0_curve, 'b-', linewidth=2.5, label=r'$\mathcal{H}_0: \rho=0$ (No anomaly)')
ax.plot(x, h1_curve, 'r-', linewidth=2.5, label=r'$\mathcal{H}_1: \rho\neq0$ (Anomaly)')

# Add threshold line
ax.axvline(gamma, color='k', linestyle='--', linewidth=2, 
           label=f'Threshold $\gamma={gamma}$')

# Shade error regions
alpha_region = x[x >= gamma]
ax.fill_between(alpha_region, 0, norm.pdf(alpha_region, mu0, sigma0), 
                color='blue', alpha=0.3, label=r'Type I error ($\alpha$)')

beta_region = x[x <= gamma]
ax.fill_between(beta_region, 0, norm.pdf(beta_region, mu1, sigma1), 
                color='red', alpha=0.3, label=r'Type II error ($\beta$)')

# Add detection indicators
ax.annotate('Detection Region', xy=(gamma+0.2, 0.1), xytext=(3.5, 0.2),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=12)
ax.annotate('No Detection Region', xy=(gamma-0.2, 0.1), xytext=(-2, 0.2),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=12)

# Add parameters and labels
ax.text(mu0, norm.pdf(mu0, mu0, sigma0)+0.01, r'$\mu_0=0$', 
        ha='center', fontsize=12)
ax.text(mu1, norm.pdf(mu1, mu1, sigma1)+0.01, r'$\mu_1=3$', 
        ha='center', fontsize=12)
ax.text(gamma, -0.02, r'$\gamma$', ha='center', fontsize=14)

# Configure plot appearance
ax.set_title('Hypothesis Testing in Statistical Domain', pad=20)
ax.set_xlabel('Test Statistic $\hat{\\rho}$', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)
ax.legend(loc='upper right', framealpha=0.95)
ax.set_ylim(0, 0.45)
ax.set_xlim(-4, 7)

plt.tight_layout()
#plt.savefig('hypothesis_testing_statistical_domain.png', dpi=300)
plt.show()