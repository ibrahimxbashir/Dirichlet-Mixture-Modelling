import numpy as np
from scipy.special import digamma, gamma, loggamma, polygamma
from sklearn.cluster import KMeans

# Auxiliary Functions
def initial_params(data, k, c=60):
    """Initialize parameters using k-means clustering"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_ * c

    # Calculate pi (cluster proportions)
    unique, counts = np.unique(labels, return_counts=True)
    pi = counts / len(data)

    return {'alpha': centers, 'pi': pi}


def dirichlet_pdf(x, alpha):
    """Compute Dirichlet PDF manually"""
    # Ensure alpha is a numpy array
    alpha = np.array(alpha)

    # Calculate log PDF
    log_pdf = (loggamma(np.sum(alpha)) -
               np.sum(loggamma(alpha)) +
               np.sum((alpha - 1) * np.log(x), axis=1))

    return np.exp(log_pdf)


def log_likelihood(x, pi, alpha):
    """Compute the log-likelihood"""
    n_components = len(pi)
    n_samples = len(x)
    pi_fj = np.zeros((n_components, n_samples))

    for j in range(n_components):
        pi_fj[j, :] = pi[j] * dirichlet_pdf(x, alpha[j])

    return np.sum(np.log(np.sum(pi_fj, axis=0)))


def compute_gamma(x, pi, alpha):
    """Compute gamma_{ij} (responsibilities)"""
    n_samples = len(x)
    n_components = len(pi)
    pij_fj = np.zeros((n_samples, n_components))

    for j in range(n_components):
        pij_fj[:, j] = pi[j] * dirichlet_pdf(x, alpha[j])

    return pij_fj / np.sum(pij_fj, axis=1)[:, np.newaxis]


def inv_digamma(y, tol=1e-16, max_iter=100):
    """Inverse digamma function using Newton-Raphson method"""

    # Initial estimates
    x = np.where(y >= -2.22, np.exp(y) + 0.5, -1.0 / (y - digamma(1.0)))

    for _ in range(max_iter):
        x_new = x - (digamma(x) - y) / polygamma(1, x)  # Newton-Raphson update
        if np.all(np.abs(x_new - x) < tol):  # Check convergence
            break
        x = x_new

    return x


def log_likelihood_Q(x, pi, alpha, gammaij):
    """Compute Q function for EM algorithm"""
    n_samples = len(x)
    n_components = len(pi)
    fj = np.zeros((n_samples, n_components))

    for j in range(n_components):
        fj[:, j] = dirichlet_pdf(x, alpha[j])

    term1 = np.sum(np.sum(gammaij * np.log(pi)))
    term2 = np.sum(np.sum(gammaij * np.log(fj)))
    return term1 + term2


# The Clustering Algorithm
def clustering(data, k=3, epsilon=1e-4, max_iter=1000):
    """Main clustering function using Dirichlet mixture model"""
    data = np.array(data)

    # Initialize parameters
    params = initial_params(data, k)
    alphas_hats = initial_alpha = params['alpha']
    pi_hats = initial_pi = params['pi']

    # Evaluate initial log-likelihood
    loglik_i = log_likelihood(data, initial_pi, initial_alpha)

    logdiff = epsilon + 1
    iters = 0
    cluster = np.zeros(len(data), dtype=int)

    while (logdiff > epsilon) and (max_iter > iters):
        # E-step: Evaluate the gammas
        gammas = compute_gamma(data, pi_hats, alphas_hats)

        # M-step: Update parameters
        pi_hats = np.sum(gammas, axis=0) / len(data)

        # Assign cluster values
        cluster = np.argmax(gammas, axis=1)

        # Update alpha for each component
        for i in range(k):
            cluster_indices = np.where(cluster == i)[0]

            if len(cluster_indices) == 0:  # If no points in cluster
                alphas_hats[i] = initial_alpha[i]
            else:
                cluster_data = data[cluster_indices]
                psi_a = digamma(np.sum(alphas_hats[i])) + np.mean(np.log(cluster_data), axis=0)
                alphas_hats[i] = inv_digamma(psi_a)

        # Compute new log-likelihood and check convergence
        loglik_j = log_likelihood(data, pi_hats, alphas_hats)
        logdiff = abs(loglik_i - loglik_j)
        loglik_i = loglik_j
        iters += 1

    return {
        'alphas': alphas_hats,
        'weights': pi_hats,
        'clusters': cluster
    }


'''

# Example usage from .rmd walkthrough:

# Generate sample data
rng = np.random.default_rng(42)
sample_dirichlet = np.vstack([
    dirichlet.rvs([30, 20, 10], size=500, random_state=rng),
    dirichlet.rvs([10, 20, 30], size=100, random_state=rng),
    dirichlet.rvs([15, 15, 15], size=300, random_state=rng)
])

# Run clustering
result = clustering(sample_dirichlet, k=3)
result['alphas']
np.bincount(result['clusters'])

'''

