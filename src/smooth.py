import numpy as np
import scipy.linalg
import scipy.optimize


def plss(Y, B, P, log_lambda, nderiv,  method="QR"):
    """Implementation of penalised least squares for fixed hyperparameter

    Args:
        Y (np.ndarray): Two dimensional observation array with subjects in first dimension and observatinos along
        the second.

        B (np.ndarray): Basis matrix evaluated at observation points. First dimension corresponds to observation
        points. Second dimension is the number of basis functions in the expansion.

        P (np.ndarray): Penalty matrix to use for regularisation term.

        log_lambda (float, Optional): Regularisation parameter to use in log form. Will be overriden if `method`
        parameter is not `fixed`. Defualts to -12.0.
        
        nderiv (int): Derivative corresponding to highest order of derivative forming penalty matrix P.
        
        method (String, {"QR", "Cholesky"}): Which method to use in calculating normal equations.
        
        
    Returns:

        C (np.ndarray): Matrix of estimated coefficients based on the penalised spline smooth fit.
        
    Raises:
        ValueError: If method of decomposition not recognised.


    """
    if log_lambda <= -12.0:
        return lss(Y, B, method)
        
    n, k = B.shape
    lam = 10**log_lambda
    neiglow = k - nderiv
    if method == "QR":
        w, V = scipy.linalg.eigh(P, lower=False, subset_by_index=[k-neiglow,k-1], driver='evr', check_finite=False)
        Pen = np.matmul(V, np.diag(np.sqrt(w)))
        augmentedX = np.vstack((B, np.sqrt(lam)*Pen.T))
        augmentedY = np.concatenate((Y, np.zeros(Pen.shape[-1]))) if len(Y.shape)==1 else np.vstack((Y, np.zeros((Pen.shape[-1], Y.shape[-1]))))
        q,r = scipy.linalg.qr(augmentedX, mode='economic')
        qty = q.T @ augmentedY
        C = scipy.linalg.solve_triangular(r, qty)
    elif method == "Cholesky":
        lam = 10**log_lambda
        BtB = np.matmul(B.T, B)
        tmp = scipy.linalg.cho_factor(BtB + lam*P , lower=True, check_finite=False)
        C = scipy.linalg.cho_solve(tmp, np.matmul(B.T, Y), check_finite=False)
    else:
        raise ValueError("Method of decomposition not recognised.")
     
    return C


def lss(Y, B, method="QR"):
    """Implementation of least squares smooth.

    Args:
        Y (np.ndarray): Two dimensional observation array with subjects in first dimension and observatinos along
        the second.

        B (np.ndarray): Basis matrix evaluated at observation points. First dimension corresponds to observation
        points. Second dimension is the number of basis functions in the expansion.
        
        method (String, {"QR", "Cholesky"}): Which method to use in calculating normal equations.

    Returns:

        C (np.ndarray): Matrix of estimated coefficients based on the spline smooth fit.
        
    Raises:
        ValueError: If method of decomposition not recognised.

    """
    if method == "QR":   
        q,r = scipy.linalg.qr(B, mode='economic')
        qty = q.T @ Y
        C = scipy.linalg.solve_triangular(r, qty)
    elif method =="Cholesky":
        BtB = np.matmul(B.T, B)
        tmp = scipy.linalg.cho_factor(BtB, lower=True, check_finite=False)
        C = scipy.linalg.cho_solve(tmp, np.matmul(B.T, Y), check_finite=False)
    else:
        raise ValueError("Method of decomposition not recognised.")
        
    return C
