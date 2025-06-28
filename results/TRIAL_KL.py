import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from matplotlib.patches import ConnectionPatch

# Configure plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titleweight': 'bold',
    'figure.figsize': (12, 6),
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

# Create distributions (Gaussian)
x = np.linspace(-4, 7, 1000)
mu0, sigma0 = 0, 1  # H0 distribution
mu1, sigma1 = 3, 1  # H1 distribution
p = norm.pdf(x, mu0, sigma0)
q = norm.pdf(x, mu1, sigma1)

# Calculate KL divergences
kl_pq = np.sum(p * np.log(p / q)) * (x[1]-x[0])  # D_KL(P||Q)
kl_qp = np.sum(q * np.log(q / p)) * (x[1]-x[0])  # D_KL(Q||P)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ======================
# Left: Statistical Domain
# ======================
ax1.plot(x, p, 'b-', lw=2.5, label=r'$P(x) = \mathcal{H}_0$')
ax1.plot(x, q, 'r-', lw=2.5, label=r'$Q(x) = \mathcal{H}_1$')
ax1.fill_between(x, p, q, where=(q>p), color='red', alpha=0.15)
ax1.fill_between(x, p, q, where=(p>q), color='blue', alpha=0.15)

# Add KL divergence arrows
ax1.annotate('', xy=(1.5, 0.15), xytext=(0.5, 0.15),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax1.text(1.0, 0.18, r'$D_{KL}(P \| Q)$', color='purple', 
         ha='center', fontsize=14, weight='bold')

ax1.annotate('', xy=(2.5, 0.15), xytext=(3.5, 0.15),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax1.text(3.0, 0.18, r'$D_{KL}(Q \| P)$', color='green', 
         ha='center', fontsize=14, weight='bold')

ax1.set_title('Statistical Domain: Hypothesis Distributions')
ax1.set_xlabel('Test Statistic $\hat{\\rho}$')
ax1.set_ylabel('Probability Density')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 0.45)
ax1.text(0.05, 0.95, f'$D_{{KL}}(P||Q) = {kl_pq:.2f}$\n$D_{{KL}}(Q||P) = {kl_qp:.2f}$',
         transform=ax1.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))

# ======================
# Right: Information-Theoretic Domain
# ======================
# Create KL divergence surface
X, Y = np.meshgrid(np.linspace(-0.5, 5, 50), np.linspace(0, 0.5, 50))
Z = np.zeros_like(X)

# Calculate KL divergence at each point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        px = norm.pdf(X[i,j], mu0, sigma0)
        qx = norm.pdf(X[i,j], mu1, sigma1)
        if px > 1e-6 and qx > 1e-6:
            Z[i,j] = px * np.log(px / qx)

# Plot KL divergence surface
contour = ax2.contourf(X, Y, Z, 20, cmap='viridis')
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label('$P(x) \log \\frac{P(x)}{Q(x)}$', rotation=270, labelpad=20)

# Add distributions on top
ax2.plot(x, p, 'w-', lw=2, alpha=0.7, label=r'$P(x)$')
ax2.plot(x, q, 'c-', lw=2, alpha=0.7, label=r'$Q(x)$')

# Add divergence markers
ax2.plot([mu0, mu0], [0, norm.pdf(mu0, mu0, sigma0)], 'b--', lw=1.5)
ax2.plot([mu1, mu1], [0, norm.pdf(mu1, mu1, sigma1)], 'r--', lw=1.5)
ax2.text(mu0, 0.45, r'$\mu_P$', color='blue', ha='center', fontsize=14)
ax2.text(mu1, 0.45, r'$\mu_Q$', color='red', ha='center', fontsize=14)

ax2.set_title('Information-Theoretic Domain: KL Divergence')
ax2.set_xlabel('Test Statistic $\hat{\\rho}$')
ax2.set_ylabel('Probability Density')
ax2.legend(loc='upper right')

# Connect the concepts between domains
con = ConnectionPatch(
    xyA=(3.0, 0.18), coordsA=ax1.transData,
    xyB=(mu1, norm.pdf(mu1, mu1, sigma1)), coordsB=ax2.transData,
    arrowstyle="->", shrinkA=5, shrinkB=5, 
    mutation_scale=20, fc="grey", alpha=0.7
)
fig.add_artist(con)

plt.tight_layout()
#plt.savefig('information_theoretic_domain.png', dpi=300, bbox_inches='tight')
plt.show()