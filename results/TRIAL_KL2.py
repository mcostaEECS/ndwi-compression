import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configure plot settings
plt.rcParams.update({
    'font.size': 14,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.figsize': (10, 7),
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

# Create Gaussian distributions
x = np.linspace(-4, 7, 1000)
mu_p, sigma_p = 0, 1  # Distribution P (H0)
mu_q, sigma_q = 3, 1  # Distribution Q (H1)
p = norm.pdf(x, mu_p, sigma_p)
q = norm.pdf(x, mu_q, sigma_q)

# Calculate KL divergence integrand
kl_integrand = p * np.log(p / np.where(q > 1e-10, q, 1e-10))  # Avoid log(0)
kl_total = np.trapz(kl_integrand, x)  # Total KL divergence

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

# Top: Distributions P and Q
ax1.plot(x, p, 'b-', lw=3, label=r'$P(x) \sim \mathcal{N}(0,1)$ ($\mathcal{H}_0$)')
ax1.plot(x, q, 'r-', lw=3, label=r'$Q(x) \sim \mathcal{N}(3,1)$ ($\mathcal{H}_1$)')
ax1.set_ylabel('Probability Density', fontsize=14)
ax1.legend(loc='upper right', fontsize=12)
ax1.set_title('Probability Distributions', pad=15)
ax1.set_ylim(0, 0.45)

# Highlight KL divergence region
x_fill = x[(x >= -1) & (x <= 4)]
ax1.fill_between(x_fill, 
                norm.pdf(x_fill, mu_p, sigma_p),
                norm.pdf(x_fill, mu_q, sigma_q),
                where=(norm.pdf(x_fill, mu_p, sigma_p) > norm.pdf(x_fill, mu_q, sigma_q)),
                color='blue', alpha=0.2, hatch='//')

# Bottom: KL Divergence D_KL(P||Q)
ax2.plot(x, kl_integrand, 'm-', lw=3, label=r'$P(x) \log \frac{P(x)}{Q(x)}$')
ax2.fill_between(x, kl_integrand, color='purple', alpha=0.2)
ax2.set_xlabel('Test Statistic $\hat{\\rho}$', fontsize=14)
ax2.set_ylabel('KL Integrand', fontsize=14)
ax2.set_title(f'Kullback-Leibler Divergence: $D_{{KL}}(P||Q) = {kl_total:.2f}$', pad=15)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_ylim(-0.5, 2.0)

# Add annotations
ax1.text(0.5, 0.25, r'$D_{KL}(P||Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx$', 
         fontsize=16, bbox=dict(facecolor='white', alpha=0.9))
ax1.annotate('Information gain when\nupdating from Q to P', 
             xy=(1.0, 0.15), xytext=(-1.5, 0.3),
             arrowprops=dict(arrowstyle='->', lw=2, color='black'),
             fontsize=12, ha='center')

# Add mean markers
ax1.plot([mu_p, mu_p], [0, norm.pdf(mu_p, mu_p, sigma_p)], 'b--', lw=1.5)
ax1.plot([mu_q, mu_q], [0, norm.pdf(mu_q, mu_q, sigma_q)], 'r--', lw=1.5)
ax1.text(mu_p, -0.05, r'$\mu_P$', ha='center', fontsize=14, color='b')
ax1.text(mu_q, -0.05, r'$\mu_Q$', ha='center', fontsize=14, color='r')

plt.tight_layout()
plt.savefig('kl_divergence_p_to_q.png', dpi=300, bbox_inches='tight')
plt.show()