import numpy as np
import scipy.linalg
import scipy.optimize


def plss(Y, B, P, log_lambda):
    """Implementation of penalised least squares for fixed hyperparameter

    Args:
        Y (np.ndarray): Two dimensional observation array with subjects in first dimension and observatinos along
        the second.

        B (np.ndarray): Basis matrix evaluated at observation points. First dimension corresponds to observation
        points. Second dimension is the number of basis functions in the expansion.

        P (np.ndarray): Penalty matrix to use for regularisation term.

        log_lambda (float, Optional): Regularisation parameter to use in log form. Will be overriden if `method`
        parameter is not `fixed`. Defualts to -12.0.
    Returns:

        C (np.ndarray): Matrix of estimated coefficients based on the penalised spline smooth fit.

        log_lambda (np.ndarray): The regularisation parameter value used in the smoothing methodology in log form.

    """
    BtB = np.matmul(B.T, B)
    chol_temp = scipy.linalg.cho_factor(BtB + 10 ** log_lambda * P, lower=True, check_finite=False)
    C = scipy.linalg.cho_solve(chol_temp, np.matmul(Y, B).T, check_finite=False)
    return C, log_lambda


def plss_rgcv(Y, B, P, gamma=0.4, bounds=[-8, 8], N=20):
    """Implementation of penalised least squares with rgcv method of choosing hyperparameter.

    Args:
        Y (np.ndarray): Two dimensional observation array with subjects in first dimension and observations along
        the second.

        B (np.ndarray): Basis matrix evaluated at observation points. First dimension corresponds to observation
        points. Second dimension is the number of basis functions in the expansion.

        P (np.ndarray): Penalty matrix to use for regularisation term.

        gamma (float, Optional): Robustness parameter for rgcv. Defaults to 0.4

        bounds (list): List specifying the bounds to use for the log lambda.

        N (int): Number of initial tries of log lambda.
    Returns:

        C (np.ndarray): Matrix of estimated coefficients based on the penalised spline smooth fit.

        log_lambda (np.ndarray): The regularisation parameter value used in the smoothing methodology in log form.

    """
    BtB = np.matmul(B.T, B)
    R = np.matmul(Y, B).T
    I = np.eye(BtB.shape[0])

    def rgcv(ll):
        lams = 10 ** ll
        lP = lams * P
        temp = BtB + lP
        chol_temp = scipy.linalg.cho_factor(temp, lower=True, check_finite=False)
        G = scipy.linalg.cho_solve(chol_temp, I, check_finite=False)
        hat = np.matmul(np.matmul(B, G), B.T)
        n_measure = hat.shape[0]
        mu = np.trace(np.matmul(hat, hat)) / hat.shape[0]
        diag_hat = np.diag(hat)
        Y_hat = np.matmul(B, np.matmul(G, R)).T
        denom = (n_measure - np.sum(diag_hat)) ** 2
        numerator = n_measure * np.sum((Y_hat - Y) ** 2)
        V = numerator / denom
        return (gamma + (1 - gamma) * mu) * V

    initial = np.array([np.random.uniform(bounds[0], bounds[1], N)]).T
    initial_score = [rgcv(init) for init in initial]
    res = scipy.optimize.minimize(rgcv, x0=initial[np.argmin(initial_score)], method='L-BFGS-B', bounds=[bounds])
    log_lambda = res.x
    return plss(Y, B, P, log_lambda)


class PLSS:
    """Calculate the coefficients from a penalised least squares smooth.

    Attributes:
        method ({"fixed", "gcv", "rgcv"}, Optional): Method in choosing regularisation hyperparameter.
    
    """

    def __init__(self, method="fixed", options={}):
        """Init PLSS class

        Args:
            method ({"fixed", "gcv", "rgcv"}, Optional): Method in choosing regularisation hyperparameter.
            * "fixed" is default choice.
            * "gcv" uses generalised cross validation to choose regularisation hyperparameter.
            * "rgcv" uses a robust generalised cross validation procedure.

            options (dict, Optional): Options to pass to method of choosing hyper parameter.
            * bounds (List, Optional): Bounds on the log scale regularisation parameter. Defaults to [-8,8]
            * N (Int, Optional): Integer of number of initial guesses of log lambda to pass to minimiser. Defaults to 20

        """
        self.method = method
        self.options = options

    @property
    def method(self):
        """Getter for method attribute.

        """
        return self.__method

    @method.setter
    def method(self, method):
        """

        Args:
            method ({"fixed", "gcv", "rgcv"}): Method in choosing regularisation hyperparameter.

        """
        if method in ["fixed", "gcv", "rgcv"]:
            self.__method = method
        else:
            raise NotImplementedError("Method not implemented.")

    @property
    def options(self):
        """Getter for options attribute

        """
        return self.__options

    @options.setter
    def options(self, options):
        """

        Args:
            options (Dict): Dictionary of possible options to pass to hyperparameter selection methods.
            * bounds (List): List of lower and upper bounds in log-scale for the regularisation hyper parameter.
            * N (int): Number of initial guesses of the regularisation parameter to try to feed into minimisation.

        """
        self.__options = {}
        if options is None:
            self.__options['bounds'] = [-8, 8]
            self.__options['N'] = 20
        elif isinstance(options, dict):
            self.__options['bounds'] = options.get('bounds', [-8, 8])
            self.__options['N'] = options.get('N', 20)
        else:
            raise ValueError("Options must be a dictionary or None.")

    def fit(self, Y, B, P, log_lambda=0.0):
        """Calculate the coefficients from a penalised least squares smooth of the data.

         We implement a penalised least squared procedure with basis matrix B and penalty matrix P governed by the
         regularisation parameter. Such a method chooses coefficients :math:`\mathbf{c}` that minimize:

         .. math:
            \\sum_{i=1}^n w_i ( y_i - B(t_i)^\top \\mathbf{c} )^2 + \\mathbf{c}^\top P \\mathbf{c}

        Args:
            Y (np.ndarray): Two dimensional observation array with subjects in first dimension and observatinos along
            the second.

            B (np.ndarray): Basis matrix evaluated at observation points. First dimension corresponds to observation
            points. Second dimension is the number of basis functions in the expansion.

            P (np.ndarray): Penalty matrix to use for regularisation term.

            log_lambda (float, Optional): Regularisation parameter to use in log form. Will be overriden if `method`
            parameter is not `fixed`. Defualts to -12.0.

        Returns:

            C (np.ndarray): Matrix of estimated coefficients based on the penalised spline smooth fit.

            reg_param (np.ndarray): The regularisation parameter value used in the smoothing methodology.

        """
        if self.method == "fixed":
            return plss(Y, B, P, log_lambda)
        elif self.method == "gcv":
            return plss_rgcv(Y, B, P, 1.0, **self.options)
        elif self.method == "rgcv":
            return plss_rgcv(Y, B, P, 0.4, **self.options)
        else:
            raise ValueError("Method not")
