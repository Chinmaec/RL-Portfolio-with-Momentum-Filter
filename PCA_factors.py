# PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from numpy.linalg import eigh

def PCA(returns, variance = 0.95, plot = False, verbose = False): 
    """
    1. Standardise the data 
    2. Compute the covariance matrix  
    3. Eigenvalue decomposition:  Σ = V Λ Vᵀ
    4. Sort the eigenvalues in descending order and pick top k eigenvectors       

    """
    # standardize(columnwise) 
    R = returns.to_numpy()
    R_std = (R - R.mean(axis=0)) / (R.std(axis=0) + (1e-8)) #incase of zero variance 

    # covariance maxtrix 
    cov = np.cov(R_std.T)

    # eigenvalues, eigenvectors 
    eigenvalues, eigenvectors = eigh(cov)

    # descending order 
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order].real
    eigenvectors = eigenvectors[:,order].real

    # explained variance ratio 
    explained = eigenvalues / eigenvalues.sum()
    cumulative = explained.cumsum()

    # how many components explain 95% of variance?
    n_components = np.searchsorted(cumulative, variance) + 1

    # factor scores 
    factors = R_std @ eigenvectors[:,:n_components]
    factors_df = pd.DataFrame(factors, index= returns.index, 
                              columns=[f"PC{i+1}" for i in range(n_components)])

    # scree plot 
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        k = len(explained)

        ax.bar(range(1, k + 1), explained[:k] * 100, alpha=0.7, label="Individual")
        ax.plot(range(1, k + 1), cumulative[:k] * 100, "o-", color="red", label="Cumulative")
        ax.axhline(80, color="gray", ls="--", lw=1, label="80% threshold")
        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title("PCA - Scree Plot")
        ax.legend()
        plt.tight_layout()
        plt.show()  

    if verbose: 
        print(f"{n_components} components explain {(variance*100)}% of total variance")
        print(f"Factor matrix shape: {factors_df.shape}")

    return eigenvalues, eigenvectors, factors_df
