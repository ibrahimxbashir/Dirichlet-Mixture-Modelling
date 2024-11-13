# Dirichlet-Mixture-Modelling
Dirichlet Mixture Modelling in R and Python from Scratch. 

An implementation of the Algorithm presented in the research paper: "Clustering compositional data using Dirichlet mixture model" by Samyajoy Pal and Christian Heumann (2022)
Download the html file for a full walk-through of the math and the code. The same functions and approach is used in the .py and .rmd files. 

### Dirichlet Overview
The Dirichlet distribution is a multivariate probability distribution defined on the simplex in an k-dimensional space. It takes as input a vector of positive real numbers (alpha parameters) and produces a probability distribution over k-dimensional vectors where each component is a non-negative real number that sums to 1. And in the context of the Dirichlet distribution, the alpha values represent the parameters that determine the shape of the distribution. More specifically, the alpha parameters determine the concentration of probability mass around the corners of the simplex. A high alpha value for a specific component means that the distribution is more concentrated around that corner, resulting in higher probabilities for vectors with larger values in that component.

By adjusting the values of the alpha parameters, you can control the shape and spread of the distribution. For example, when all alpha values are equal, the distribution is uniform (equal probability mass across all components). When some alpha values are much larger than others, the distribution becomes more peaked or concentrated in those corners. The idea behind the meaning of the parameters of the distribution will be elaborated on in the real-life example section at the end.


### Math Overview for Modelling Algorithm

I Implemented the "Hard DMM 1" clustering algorithm for a mixture of Dirichlet distributions with provision for empty clusters (Algorithm 1) from the referenced research paper "Clustering compositional data using Dirichlet mixture model" by Samyajoy Pal and Christian Heumann (2022). Here is an overview of the relevant functions used in the algorithm:

The density of a mixture model with $k$ components for an observation $x_i$ is given by the mixture density: 

$$p(x_i) = \sum_{j=1}^{k} \pi_j f_j(x_i | \alpha_j) \tag{1}$$ 

where $\pi$ is a vector of length $k$ of weights (i.e. $\pi = (\pi_1, \pi_2,...,\pi_k)$) that are used as the mixture proportions of each component $k$ such that $\sum_{i=1}^{k} \pi_i =1$ and $0 \le \pi_i \le 1$. The $f_j(x_i | \alpha_j)$ is the density component of the mixture $j$ with the corresponding parameters of that mixture $\alpha_j$ (with the length of the dimension of the Dirichlet model, and since we have $k$ Dirichlet models, we have that $j=1,2,...,k$); so $\alpha = (\alpha_1, \alpha_2, ..., \alpha_k)$ (again, each of the length of the dimension of the Dirichlet model) represents the vector of all the parameters of the model. 

Accordingly, the log-likelihood of the model of size $N$ is given by: 
$$\log p(x_1,x_2,...,x_N|\alpha,\pi) = \sum_{i=1}^{N}\log \left(\sum_{j=1}^{k}\pi_j f_j(x_i | \alpha_j)\right) \tag{2}$$
The equation that represents the probability of a data point $i$ for cluster $j$ is given as:
$$\gamma_{ij} = \frac{\pi_j f_j(x_i | \alpha_j)}{\sum_{l=1}^{k} \pi_l f_l(x_i | \alpha_l)} \tag{3}$$

The approach in the paper implements an EM algorithm, and accordingly they aim at optimizing the function:
$$ Q(\alpha, alpha^{t-1}) = E[\sum_{i=1}^{N}\log(p(x_i,z_i|\alpha))|x,\alpha^{t-1}] = \sum_{i=1}^{N}\sum_{j=1}^{k} \gamma_{ij}\log\pi_j + \sum_{i=1}^{N}\sum_{j=1}^{k}\gamma_{ij}\log f_j(x_i|\alpha_j)$$
And this function is optimized with respect to $\alpha$ and $\pi$, having us update $\pi_j$ with $\frac{N_j}{N}$ where $N_j = \sum_{i=1}^{N} \gamma_{ij}$, the cluster assignment being cluster = $\underset{j}{\mathrm{argmax}} \gamma_{ij}$, and $\alpha$ with the MLE of $\alpha$ with the clustered data. The research paper references another paper (Estimating a Dirichlet distribution, Thomas P. Minka, 2000) to find a suitable estimate for $\alpha_j^{MLE}$; it uses a fixed/one point iteration using the Newton-Raphson algorithm to provide the MLE of the Dirichlet parameters:

$$\Psi(\alpha_{jm}^{new}) = \Psi(\sum_{m=1}^{p}\alpha_{jm}^{old}) + \frac{1}{N_j}\sum_{i=1}^{N_j} \log (x_{im}) \tag{4}$$
