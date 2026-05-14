# Dirichlet-Mixture-Modelling
Dirichlet Mixture Modelling in R and Python from Scratch. 

An implementation of the algorithm presented in the research paper: *"Clustering compositional data using Dirichlet mixture model"* by Samyajoy Pal and Christian Heumann (2022).  

Download the html file for a full walk-through of the math alongside the R code. The same functions and approach are used in the `.py` and `.rmd` files.

### Dirichlet Overview
The Dirichlet distribution is a multivariate probability distribution defined on the simplex in a $p$-dimensional space. It takes as input a vector of positive real numbers (alpha parameters) and produces a probability distribution over $p$-dimensional vectors where each component is a non-negative real number that sums to $1$. In the context of the Dirichlet distribution, the alpha values represent the parameters that determine the shape of the distribution. More specifically, the relative values of the alpha parameters determine where the probability mass is concentrated on the simplex, while the total concentration $\alpha_0 = \sum_{m=1}^{p}\alpha_m$ controls the spread of the distribution.

By adjusting the values of the alpha parameters, you can control the shape and spread of the distribution. For example, when all alpha values are equal to $1$, the distribution is uniform (equal probability mass across all components). When some alpha values are much larger than others, the distribution becomes more concentrated around the corresponding components. When all alpha values are less than $1$, the distribution concentrates more strongly around the corners of the simplex. The idea behind the meaning of the parameters of the distribution will be elaborated on in the real-life example section at the end.


### Math Overview for Modelling Algorithm

I implemented the "Hard DMM 1" clustering algorithm for a mixture of Dirichlet distributions with provision for empty clusters (Algorithm 1) from the referenced research paper *"Clustering compositional data using Dirichlet mixture model"* by Samyajoy Pal and Christian Heumann (2022). Here is an overview of the relevant functions used in the algorithm.

Let $X_1, X_2, \dots, X_N$ denote a random sample of size $N$, where each observation $x_i$ is a $p$-dimensional compositional vector satisfying:

$$
x_i = (x_{i1}, x_{i2}, \dots, x_{ip}), \qquad x_{im} > 0, \qquad \sum_{m=1}^{p} x_{im} = 1
$$

The density of a mixture model with $K$ components for an observation $x_i$ is given by the mixture density:

$$
p(x_i) = \sum_{j=1}^{K} \pi_j f_j(x_i \mid \alpha_j) \quad \text{(1)}
$$

where $\pi$ is a vector of length $K$ of weights (i.e. $\pi = (\pi_1, \pi_2, \dots, \pi_K)$) that are used as the mixture proportions of each component such that $\sum_{j=1}^{K} \pi_j = 1$ and $0 \le \pi_j \le 1$.

The Dirichlet density component for cluster $j$ is given by:

$$
f_j(x_i \mid \alpha_j)
=
\frac{
\Gamma\left(\sum_{m=1}^{p}\alpha_{jm}\right)
}{
\prod_{m=1}^{p}\Gamma(\alpha_{jm})
}
\prod_{m=1}^{p}
x_{im}^{\alpha_{jm}-1}
\quad \text{(2)}
$$

where $\alpha_j = (\alpha_{j1}, \alpha_{j2}, \dots, \alpha_{jp})$ is the parameter vector for mixture component $j$.

Accordingly, the log-likelihood of the model for a sample of size $N$ is given by:

$$
\log p(x_1, x_2, \dots, x_N \mid \alpha, \pi)
=
\sum_{i=1}^{N}
\log
\left(
\sum_{j=1}^{K}
\pi_j f_j(x_i \mid \alpha_j)
\right)
\quad \text{(3)}
$$

The equation that represents the probability of a data point $i$ belonging to cluster $j$ is given by:

$$
\gamma_{ij}
=
\frac{
\pi_j f_j(x_i \mid \alpha_j)
}{
\sum_{l=1}^{K}
\pi_l f_l(x_i \mid \alpha_l)
}
\quad \text{(4)}
$$

The approach in the paper implements a Hard EM algorithm. Accordingly, they optimize the expected complete-data log-likelihood function:

$$
Q(\alpha, \pi \mid \alpha^{t-1}, \pi^{t-1})
=
E\left[
\sum_{i=1}^{N}
\log(p(x_i,z_i \mid \alpha,\pi))
\mid
x,\alpha^{t-1},\pi^{t-1}
\right]
$$

which can be written as:

$$
Q(\alpha, \pi \mid \alpha^{t-1}, \pi^{t-1})
=
\sum_{i=1}^{N}\sum_{j=1}^{K}
\gamma_{ij}\log\pi_j
+
\sum_{i=1}^{N}\sum_{j=1}^{K}
\gamma_{ij}\log f_j(x_i \mid \alpha_j)
\quad \text{(5)}
$$

where $z_i \in \{1,2,\dots,K\}$ is the latent cluster assignment variable.

This function is optimized with respect to $\alpha$ and $\pi$, having us update: $\pi_j^{\text{new}} = \frac{N_j}{N}$ where $N_j = \sum_{i=1}^{N}\gamma_{ij}$.

The Hard EM assignment step is then given by:

$$
z_i = \underset{j}{\mathrm{argmax}} \ \gamma_{ij}
$$

where each data point $x_i$ is assigned to the cluster with the highest membership probability.

The research paper references another paper (*Estimating a Dirichlet distribution*, Thomas P. Minka, 2000) to find a suitable estimate for $\alpha_j^{MLE}$. It uses a fixed-point iteration to estimate the Dirichlet parameters:

$$
\Psi(\alpha_{jm}^{\text{new}})
=
\Psi\left(
\sum_{m=1}^{p}
\alpha_{jm}^{\text{old}}
\right)
+
\frac{1}{N_j}
\sum_{i:z_i=j}
\log(x_{im})
\quad \text{(6)}
$$

where $\Psi$ is the digamma function, $p$ is the dimension of the distribution, and the second term takes the mean of the log of the data points assigned to cluster $j$ corresponding to $\alpha_j$.

This requires another Newton-Raphson algorithm to invert $\Psi$ (i.e. to solve $\Psi^{-1}(y)=x$ for the equation $\Psi(x)=y$), but the paper (P. Minka, 2000; Appendix C) provides a reasonable estimate. Five Newton iterations are generally sufficient to reach very high precision for the initialization used in my function.

The research paper also provides an initialization for the $\alpha$ and $\pi$ parameters. For the Dirichlet Mixture Model, they initialize $\alpha$ with the centroids of KMeans multiplied by a scalar $c$ (they use $c=60$, as did I), and for $\pi$, it can be initialized either by sampling from a Dirichlet$(1,1,\dots,1)$ distribution or by using the empirical ratios of the number of cluster members from the KMeans algorithm divided by the total number of observations. The paper uses the KMeans method, so I did too.

With that said, the auxiliary functions (found in the `.rmd` section, and in the beginning of the `.py` file) compute all of these steps. The function `initial_params` provides the initialization parameters for $\alpha$ and $\pi$; `log.lik.gij` computes the log-likelihood equation found in $(3)$; `gammaij` computes the $\gamma_{ij}$ from $(4)$; and `inv.digamma` uses the Newton-Raphson iterative algorithm to estimate $\Psi^{-1}(x)$ within some specified precision (I use `1e-15` for the tolerance).

### Dirichlet Mixture Modelling Algorithm

With that said, the algorithm can be expressed as follows:

$$
\begin{array}{l}
\text{Initialize the model parameters } \alpha, \pi, \text{ and the log likelihood using equation } (3). \\
\\
\textbf{While} \, \text{log difference} \ge \epsilon: \\
\\
\quad 1. \ \text{Evaluate } \gamma_{ij} \text{ from equation } (4) \text{ using parameter values } \alpha \text{ and } \pi, \text{ and the data.} \\
\\
\quad 2. \ \pi_j^{\text{new}} = \frac{N_j}{N}
\quad \text{where }
N_j = \sum_{i=1}^{N}\gamma_{ij}. \\
\\
\quad 3. \ \textbf{for } i = 1, \dots, N: \\
\\
\quad \quad \quad z_i = \underset{j}{\mathrm{argmax}} \, \gamma_{ij}. \\
\\
\quad \quad \quad \text{Assign data point } x_i \text{ to cluster } z_i. \\
\\
\quad 4. \ \textbf{for } j = 1, \dots, K: \\
\\
\quad \quad \quad \textbf{if cluster } j \text{ is empty:} \\
\\
\quad \quad \quad \quad \text{Use initial values of } \alpha_j \text{ as update.} \\
\\
\quad \quad \quad \textbf{else:} \\
\\
\quad \quad \quad \quad \alpha_j^{\text{new}} = \alpha_j^{\text{MLE}} \\
\\
\quad 5. \ \text{Re-evaluate the log-likelihood using updated parameters.}
\end{array}
$$

If you would like to see how I have used this approach in a practical setting with real data, you can read the end of my post on Medium:

https://ibrahimxbashir.medium.com/dirichlet-mixture-modelling-in-r-from-scratch-e29ee1a10c8b
