import numpy as np
from utils import GP, Matern
from fda import fpca, mafr

def forecast_score(score, x_ob, x_new):
    """ Forecast a single score series using a GP regression.
    
    Args:
        score (ndarray): Score series to regress upon.
        x_ob (ndarray): Obersvation points corresponding to score.
        x_new (ndarray): New point to forecast the score series at.
    
    returns:
        mu_for (ndarray): posterior mean of the fitted gaussian process.
    """
    gp = GP(Matern(nu=2.5, rho=1.0, sigma=1.0))
    gp.fit(x_ob, score.T, bounds=[(-6,2), (-6,2), (-6, 2)], n_init=100)
    mu_for, V_for = gp.posterior(x_new, mean_only=True)
    return mu_for

def forecast(Y_noisy, Y_true, t, B, P, log_lambda, nderiv, J, N_train, steps, ncomp, use_mafr=False):
    """Forecast methodology to a single simulated data set with noise. 

    Methodology based on the FTSA methodology, [1].

    [1]H. Shang Lin, ‘ftsa: An R Package for Analyzing Functional Time Series’, The R Journal, vol. 5, no. 1, p. 64, 2013, doi: 10.32614/RJ-2013-006.

    
    Args:
        Y_noisy (ndarray): Noisy simulated data set to forecast. 
        Y_true (ndarray): Actual unobserved data for error calculation.
        t (ndarray): time steps of each observed surface. 
        B (ndarray): Basis system matrix for the data used in regularised FPCA. 
        P (ndarray): Penalty matrix for the data used in regularised MAFR.
        log_lambda (float): Log regularisation parameter.
        nderiv (int): Highest order deriviative for penalty P. 
        J (ndarray): Inner product of the basis system used in regularised FPCA. 
        N_train (int): Number of time points to use as training data exclusively. 
        steps (List of ints): Step sizes for forecast ahead. 
        ncomp (int): Number of componentes to use in FPCA decomposition.
        use_mafr (boolean): Indicator whether to use mafr rotation. 
    
    Returns:
        results (ndarray): The average mean square error of the `steps` step ahead forecast. 
    """
    iterations = len(t)-N_train-np.max(steps)
    results = np.zeros((iterations, len(steps)))
    for i in np.arange(iterations):
        Y_train = Y_noisy[:(N_train+i)]
        x_ob = t[:(N_train+i)]
        x_new = t[(N_train+i):(N_train+i+np.max(steps))][[s-1 for s in steps]]
        Cbar, zeta, scores, w = fpca(Y_train.T, B, P, log_lambda, nderiv, J, ncomp)
        if use_mafr:
            zeta, scores, U = mafr(zeta, scores, P)
        mean_scores_for = np.array([forecast_score(score, x_ob, x_new) for score in scores.T])
        actual = Y_true[(N_train+i):(N_train+i+np.max(steps))][[s-1 for s in steps]].T
        recon = np.matmul(np.matmul(B, zeta), mean_scores_for) + np.matmul(B, Cbar)[:, np.newaxis]
        results[i] = np.mean((actual-recon)**2, axis=0)
    return np.mean(results, axis=0)
