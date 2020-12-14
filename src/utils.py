from abc import ABC, abstractmethod
import numpy as np
import scipy.stats
from scipy.special import kv, gamma
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.optimize import minimize
import math
import cv2


class Kernel(ABC):
    """Abstract kernel

    Attributes:
        parameters (dict): Dictionary of kernel parameters.
    """

    def __init__(self, parameters):
        """Inits Kernel.
        Args:
            parameters (dict): Dictionary of parameters of the kernel.
        """
        self.parameters = parameters

    def set_param(self, parameter_name, value):
        """Set a parameter value.

        Args:
            parameter_name (string): Dictionary key for parameter.
            value (numeric): Value of parameter.
        """
        if parameter_name in self.parameters:
            self.parameters[parameter_name] = value
        else:
            raise ValueError('parameter_name must be in the key.')

    def get_param(self, parameter_name):
        """Returns a parameter by name.

        Args:
            parameter_name (string): Dictionary key for parameter.

        Returns:
            param_value (numeric): Parameter value.
        """
        return self.parameters[parameter_name]

    @abstractmethod
    def __call__(self, x1, x2=None):
        """Calculate kernel.

        Args:
            x1 (array_like): Input for kernel, x1.
            x2 (array_like): Input for kernel, x2.

        Returns:
            K (array_like): x1 by x2 covariance kernel.
        """
        pass


class Identity(Kernel):
    def __init__(self, sigma):
        """Inits identity kernel"""
        params = {}
        params['sigma'] = sigma
        super().__init__(params)

    def __call__(self, x1, x2=None):
        """Calculates kernel
        Args:
            x1 (array_like): Input for kernel, x1.
            x2 (array_like): Input for kernel, x2.

        Returns:
            k (array_like): kernel evaluated at x1 and x2.
        """
        var = self.get_param('sigma')**2
        x1 = np.atleast_2d(x1).T if len(x1.shape) == 1 else x1
        if x2 is None:
            d = squareform(pdist(x1))
        else:
            x2 = np.atleast_2d(x2).T if len(x2.shape) == 1 else x2
            d = cdist(x1, x2)
        if isinstance(d, np.ndarray):
            return var * (d == 0)
        else:
            return var if d == 0 else 0
        
    def jac(self, x1, x2=None):
        """Calculates jacobian of identity kernel.
        
        """
        sigma = self.get_param('sigma')
        x1 = np.atleast_2d(x1).T if len(x1.shape) == 1 else x1
        if x2 is None:
            d = squareform(pdist(x1))
        else:
            x2 = np.atleast_2d(x2).T if len(x2.shape) == 1 else x2
            d = cdist(x1, x2)
        if isinstance(d, np.ndarray):
            return (2*sigma * (d == 0))[np.newaxis,:,:]
        else:
            return np.array([2*sigma]) if d == 0 else np.array([0])


class Matern(Kernel):
    def __init__(self, nu, sigma, rho):
        """Inits Matern kernel with shape nu and lengthscale rho.

        Args:
            nu (numeric): Shape parameter of kernel.
            rho (List of  numeric): Length scale parameter per dimension of input.
        """
        params = {'nu': nu}
        params['sigma'] = sigma
        rho = rho if isinstance(rho, list) else [rho]
        for i, r in enumerate(rho):
            params['rho_' + str(i)] = r
        super().__init__(params)

    def __call__(self, x1, x2=None):
        """Calculate kernel.
        Args:
            x1 (array_like): Input for kernel, x1.
            x2 (array_like): Input for kernel, x2.

        Returns:
            k (array_like): kernel evaluated at d.
        """
        nu = self.get_param('nu')
        sigma = self.get_param('sigma')
        rho = np.array([self.get_param(r) for r in self.parameters.keys() if r.startswith('rho_')])
        x1 = np.atleast_2d(x1).T if len(x1.shape) == 1 else x1
        if x2 is None:
            dists = squareform(pdist(np.matmul(x1, np.diag(1.0 / rho))))
        else:
            x2 = np.atleast_2d(x2).T if len(x2.shape) == 1 else x2
            dists = cdist(np.matmul(x1, np.diag(1.0 / rho)), np.matmul(x2, np.diag(1.0 / rho)))
        if nu == 0.5:
            k = np.exp(-dists)
        elif nu == 1.5:
            k = dists * math.sqrt(3)
            k = (1. + k) * np.exp(-k)
        elif nu == 2.5:
            k = dists * math.sqrt(5)
            k = (1. + k + k ** 2 / 3.0) * np.exp(-k)
        elif nu == np.inf:
            k = np.exp(-dists ** 2 / 2.0)
        else:  # general case; expensive to evaluate
            k = dists
            k[k == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * nu) * k)
            k.fill((2 ** (1. - nu)) / gamma(nu))
            k *= tmp ** nu
            k *= kv(nu, tmp)
        return sigma**2 * k

    def jac(self, x1, x2=None):
        """Calculates kernel Jacobian with respect to params.
            Args:
            x1 (array_like): Input for kernel, x1.
            x2 (array_like): Input for kernel, x2.
        Returns:
            grad (array_like): Jacobian of kernel evaluated at x1, x2 w.r.t the hyperparameters.
        """
        nu = self.get_param('nu')
        sigma = self.get_param('sigma')
        noise_jac = (2*sigma*self(x1, x2)/sigma**2)[np.newaxis,:,:]
        x1 = np.atleast_2d(x1).T if len(x1.shape) == 1 else x1
        rho = np.array([self.get_param(r) for r in self.parameters.keys() if r.startswith('rho_')])
        if x2 is None:
            dists = squareform(pdist(np.matmul(x1, np.diag(1.0 / rho))))
            grad_dists = np.array([squareform(pdist(np.atleast_2d(xi).T)) for xi in x1.T]) ** 2
        else:
            x2 = np.atleast_2d(x2).T if len(x2.shape) == 1 else x2
            dists = cdist(np.matmul(x1, np.diag(1.0 / rho)), np.matmul(x2, np.diag(1.0 / rho)))
            grad_dists = np.array([cdist(np.atleast_2d(xi).T, np.atleast_2d(yi).T) for xi, yi in zip(x1.T, x2.T)]) ** 2
            
        grad_dists *= (1.0 / rho ** 3)[:, np.newaxis, np.newaxis]
        grad_dists = -1.0 * np.divide(grad_dists, dists, out=np.zeros_like(grad_dists), where=dists != 0)
        C = self(x1) if x2 is None else self(x1, x2)
        if nu == 2.5:
            grad = (np.sqrt(5) * grad_dists + (10 * dists / 3) * grad_dists) * np.exp(-np.sqrt(5) * dists) - np.sqrt(
                5) * grad_dists * C
        elif nu == 1.5:
            grad = np.sqrt(3) * grad_dists * np.exp(-np.sqrt(3) * dists) - np.sqrt(3) * grad_dists * C
        elif nu == 0.5:
            grad = - grad_dists * C
        elif nu == np.inf:
            grad = dists * grad_dists * C
        else:
            raise NotImplementedError('Gradient not implemented for arbitrary shape parameter')
            
        return np.vstack((noise_jac, (sigma**2) * grad))

class GP:
    """Implement basic gaussian process
    Args:
        kernel: Kernel function to use.
    """
    
    def __init__(self, kernel, noise_sigma=1e-4):
        """Inits GP class with kernel
        
        Args:
            kernel (Kernel): Kernel to use for the Gaussian process.
        
        """
        self.kernel = kernel
        self.noise_kernel = Identity(noise_sigma)
        
    @property
    def kernel(self):
        """Getter for the kernel attribute.

        """
        return self.__kernel

    @kernel.setter
    def kernel(self, kernel):
        """Setter for kernel attribute.

        Args:
             kernel (Kernel):  The kernel function for the gaussian process.
        Raises:
             ValueError: If kernel not of instance Kernel

        """
        if not isinstance(kernel ,Kernel):
            raise ValueError()
        self.__kernel = kernel
        
    def draw(self, x, nsamples=1):
        """Draw a sample from gaussian process with kernel
        
        Args:
            x (ndarray): The observation points to draw.
            nsamples (int): The number of samples to draw. 
            
        Returns:
            ndarray: Samples from the Gaussian process. 
        """
        K = self.kernel(x) 
        np.fill_diagonal(K, np.diag(K) + np.diag(self.noise_kernel(x)))
        L = scipy.linalg.cholesky(K, lower=True, check_finite=False)
        z = np.random.normal(0, 1, (nsamples, L.shape[0]))
        return np.einsum('ij, kj -> ki', L, z)
    
    def lml(self, x, y):
        """Calculate the log marginal liklihood.
        
        Args:
            x (ndarray): Observation training points.
            y (ndarray): Observed training values.
        """
        n = len(x)
        self.train_x = x
        self.train_K = self.kernel(x)
        np.fill_diagonal(self.train_K, np.diag(self.train_K) + np.diag(self.noise_kernel(x)))
        self.L = scipy.linalg.cholesky(self.train_K, lower=True, check_finite=False)
        beta = scipy.linalg.solve_triangular(self.L, y,lower=True, overwrite_b=False, check_finite=False)
        self.alpha = scipy.linalg.solve_triangular(self.L.T, beta, lower=False, overwrite_b=True, check_finite=False)
        
        ## Log Marginal Liklihood.
        log_det_K = 2*np.sum(np.log(np.diag(self.L)))
        val = -0.5*(np.matmul(y.T, self.alpha) + log_det_K + n*np.log(2*np.pi))
        return val
    
    def fit(self, x, y, bounds, n_init):
        """Fit GP hyperparameters
        Args:
            x (ndarray): Training points. 
            y (ndarray): observation data.
            bounds (List of tuples): Bounds for each hyperparameter.
            n_init (int): Number of initialisations for minimisation.
            
        Returns: 
            res (MinimizeResult): Results of L-BFGS-B minimization.
        """
        def nlml(params):
            self.noise_kernel.set_param('sigma', np.exp(params[0]))
            self.kernel.set_param('sigma', np.exp(params[1]))
            self.kernel.set_param('rho_0', np.exp(params[2]))
            return -np.squeeze(self.lml(x, y))
        initial = np.array([np.random.uniform(*b, n_init) for b in bounds])
        nlmls = [nlml(x0) for x0 in initial.T]
        x0 = initial[:, np.argmin(nlmls)]
        res = scipy.optimize.minimize(nlml, x0=x0, bounds=bounds, method="L-BFGS-B")
        return res
    
    def posterior(self, x_new, mean_only=False):
        Ks = self.kernel(x_new, self.train_x)
        fs = np.matmul(Ks, self.alpha)
        if mean_only:
            return fs, None
        v = scipy.linalg.solve_triangular(self.L, Ks.T, lower=True)
        Kss = self.kernel(x_new, x_new)
        V = Kss - np.matmul(v.T, v)
        return fs, V
        
def whiteNoise(var, ds_size):
    return np.random.normal(0, var, ds_size)

def structNoise(var, ds_size, l=0.2, scale_percent=20):
    kernel = Matern(nu=1.5, sigma=np.sqrt(var), rho=[l,l])
    n = ds_size[0]
    width = int(ds_size[2] * scale_percent / 100)
    height = int(ds_size[1] * scale_percent / 100)
    s =  np.array(np.meshgrid(np.linspace(0,1,height), np.linspace(0,1,width))).reshape(2,-1).T 
    gp = GP(kernel)
    sn = gp.draw(s, n).reshape(n, width, height)
    output = np.array([cv2.resize(sni, (ds_size[2], ds_size[1]), interpolation = cv2.INTER_CUBIC) for sni in sn]).reshape(n, -1)
    return output