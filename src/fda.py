import numpy as np
import scipy.linalg
from plssmooth import plss

def PCA(data, n_components=2):
    """Implementation of multivariate PCA. 
    
    Args:
        data (ndarray): Two dimensional array with replications in first axis, assumed mean centered.
        
        n_components (ndarray): Number of principal components to keep.
        
    Returns:
        evals (ndarray): Eigen values
        
        evecs (ndarray): Eigen vectors
        
    """
    m, n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = scipy.linalg.eigh(R, check_finite=False)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    return evals, evecs[:, :n_components]


def fpca(Y, B, P, log_lambda, nderiv, J, n_components):
    """Implementation of regularised functional principal component analysis.
    
    [1]J. O. Ramsay and B. W. Silverman, Functional data analysis. New York (N.Y.): Springer Science+Business Media, 2010.

t
    Args:
        Y (np.ndarray): Two dimensional observation array with subjects in first dimension and observations along
        the second.

        B (np.ndarray): Basis matrix evaluated at observation points. First dimension corresponds to observation
        points. Second dimension is the number of basis functions in the expansion.

        P (np.ndarray): Penalty matrix to use for regularisation term.

        log_lambda (float, Optional): Regularisation parameter to use in log form. Will be overriden if `method`
        parameter is not `fixed`. Defualts to -12.0.
        
        nderiv (int): Derivative corresponding to highest order of derivative forming penalty matrix P.
        
        J (ndarray): Innerproduct of the basis system with itself.

        
    Returns:

        mean_coefs (np.ndarray): Coefficients for estimated mean function in basis B.
        
        eigen_coefs (np.ndarray): Coefficients for estimated eigenfunctions in basis B.
        
        scores (np.ndarray): Scores for eigen decomposition in basis B.
        
        eigen_vals (np.ndarray): Eigenvalues for all eigenfunctions. 
        
    """
    C = plss(Y, B, P, log_lambda, nderiv, method="Cholesky")
    Cbar = np.mean(C, axis=-1)
    C -= Cbar[:, np.newaxis]
    if log_lambda <= -12:
        L = scipy.linalg.cho_factor(J , lower=True, check_finite=False)
    else:
        lam = 10**log_lambda
        L = scipy.linalg.cho_factor(J + lam * P)
    D = scipy.linalg.cho_solve(L, np.matmul(J, C), check_finite=False)
    w, V = PCA(D.T, n_components)
    zeta = scipy.linalg.cho_solve(L, V)
    norms = np.linalg.norm(zeta, axis = 0, keepdims=True)
    #zeta /= norms
    scores = np.matmul(C.T, np.matmul(J, V))
    return Cbar, zeta, scores, w

def mafr(zeta, scores, J):
    """Obtain the MAFR rotation from an fpca decomp with additional penalty P.
    
    [1]G. Hooker and S. Roberts, ‘Maximal autocorrelation functions in functional data analysis’, Stat Comput, vol. 26, no. 5, pp. 945–950, Sep. 2016, doi: 10.1007/s11222-015-9582-5.

    
    Args:
        zeta (ndarray): Eigenfunctions from an fpca decomposition. 
        
        scores (ndarray): Scores from an fpca decomposition.
        
        J (ndarray): Penalty matrix of Basis in eigenfunction expansion for MAFR rotation.
        
    Returns:
        mafr_zeta (ndarray): MAFR rotated eigenfunctions.
        
        mafr_scores (ndarray): MAFR rodataed scores
        
        evecs (ndarray): MAFR rotation matrix. 
    """
    P = np.matmul(np.matmul(zeta.T, J), zeta)
    evals, evecs = scipy.linalg.eigh(P, check_finite=False)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    mafr_zeta = np.matmul(zeta, evecs)
    mafr_scores = np.matmul(scores, evecs)
    return mafr_zeta, mafr_scores, evecs
