from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import romb
from scipy.interpolate import splev


class Basis(ABC):
    """
    Representation of univariate basis system.

    Attributes:
        domain (tuple): The domain over which the basis system covers.

        K (int): The number of basis functions to use in the basis system.

    """

    def __init__(self, domain, K):
        """Inits Basis class.

        Args:
            domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

            K (int): Number of basis functions to use in the basis system.

        """
        self.domain = domain
        self.K = K

    @property
    def domain(self):
        """Getter for the domain attribute.

        """
        return self.__domain

    @domain.setter
    def domain(self, domain):
        """Setter for domain attribute.

        Args:
            domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

        Raises:
             ValueError: If domain not length 2.

        """
        if len(domain) != 2:
            raise ValueError("domain must be of length 2.")
        self.__domain = domain

    @property
    def K(self):
        """Getter for the K attribute."""
        return self.__K

    @K.setter
    def K(self, K):
        """Setter for the K attribute.

        Args:
            K (int): Number of basis functions to use in the basis system.

        Raises:
             ValueError if K is not an integer and cannot be converted to one.

        """
        self.__K = int(K)

    @abstractmethod
    def _evaluate_basis(self, x, q):
        """Evaluate the qth derivative of all basis functions at locations x.

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int): The order of the derivative to take of the basis functions.

        Returns:
            (np.ndarray): A :math:`n \\times K` matrix with :math:`k^\\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        """
        raise NotImplementedError()

    def __call__(self, x, q=0):
        """Evaluate the qth derivative of all basis functions at locations x.

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int, Optional): The order of the derivative to take of the basis functions. Defaults to 0.

        Returns:
            (np.ndarray): A :math:`n \\times K` matrix with :math:`k^\\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        Raises:
            ValueError: If not all locations `x` lie in the domain of the basis system.

        """
        if not (np.all(np.less_equal(self.domain[0], x)) and np.all(np.less_equal(x, self.domain[1]))):
            raise ValueError("Arguments must all be within the domain of the basis system")
        basis_mat = self._evaluate_basis(x, q)
        return basis_mat

    @abstractmethod
    def penalty(self, q, k=12):
        """Calculate the qth order penalty matrix for the basis system.

         The form of the penalty matrix is given by:

        .. math::
            P = [p_{kl}]

        .. math::
            p_{kl} = \int B_k^{(q)}(t) B_l^{(q)}(t)dt

        where :math:`B_k` is the :math:`k^\\text{th}` basis function.

        Args:
            q (int): The order of the derivative of the penalty matrix.

            k (int, Optional): Number of samples for romberg integration calculated by :math:`2^k + 1`. Defaults to 12.

        Returns:
            (np.ndarray): A :math:`K \\times K` matrix with elements given by :math:`p_{kl}`.

        """
        raise NotImplementedError()


class Monomial(Basis):
    """Representation of the univariate monomial basis system.

     Basis system is specified as the collection :math:`\{B_k\}_{k=1}^K` where:

    .. math::
        B_k(t) = t^{k-1}

    Attributes:
        domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

        K (int): Number of basis functions to use in the basis system.

    """

    def __init__(self, domain, K):
        """Inits the Monomial class to represent the monomial basis system across the domain.
        Args:
            domain (tuple): The domain of the basis system specified by the lower and upper bound of the system.

            K (int): Number of basis functions to use in the basis system.

        """
        super().__init__(domain, K)

    def _evaluate_basis(self, x, q):
        """Evaluate the qth derivative of all basis functions at locations x for the Monomial basis system.

        The qth derivative of basis function :math:`B_k(t)` is given by:
        .. math:
            \frac{d^{(q)}B_k(t)}{dt} = \prod_{i=1}^{q}(k-i) t^{k-q}

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int): The order of the derivative to take of the basis functions.

        Returns:
            (np.ndarray): A :math:`n \\times K` matrix with :math:`k^\\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        """
        deg = self.K
        monomial_vecs = np.vander(x, N=deg, increasing=True)
        if q != 0:
            fac = [np.prod(range(f + 1 - q, f + 1)) if f > 0 else 0 for f in range(0, deg)]
            monomial_vecs = fac * ((np.c_[np.ones((len(x), q)), monomial_vecs])[:, :-q])
        return monomial_vecs

    def penalty(self, q, k=12):
        """Calculate the qth order penalty matrix for the basis system.

         The form of the penalty matrix is given by:

        .. math::
            P = [p_{kl}]

        .. math::
            p_{kl} = \int B_k^{(q)}(t) B_l^{(q)}(t)dt

        where :math:`B_k` is the :math:`k^\\text{th}` basis function.

        Args:
            q (int): The order of the derivative of the penalty matrix.

            k (int, Optional): Number of samples for romberg integration calculated by :math:`2^k + 1`. Defaults to 12.

        Returns:
            (np.ndarray): A :math:`K \\times K` matrix with elements given by :math:`p_{kl}`.

        """
        inner_product = np.zeros((self.K, self.K))
        for i in np.arange(q, self.K):
            ifac = 1
            for k in np.arange(1, q + 1):
                ifac *= i - k + 1
            for j in np.arange(i, self.K):
                jfac = 1
                for k in np.arange(1, q + 1):
                    jfac *= j - k + 1
                ipow = i + j - 2 * q + 1
                inner_product[i, j] = (self.domain[1] ** ipow - self.domain[0] ** ipow) * ifac * jfac / ipow
                inner_product[j, i] = inner_product[i, j]
        return inner_product


class Exponential(Basis):
    """Representation of the univariate exponential basis system.

    Basis system is specified as the collection :math:`\{B_k\}_{k=1}^K` where:

    .. math::
        B_k(t) = e^{\\theta_k t}

    where :math:`\\theta` is a :math:`K` length vector of rates.

    Attributes:
        domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

        K (int): Number of basis functions to use in the basis system.

        theta (tuple): Rate parameters for each basis function.

    """

    def __init__(self, domain, K, theta=None):
        """Inits the univariate exponential basis system.

        Args:
            domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

            K (int): Number of basis functions to use in the basis system.

            theta (tuple, optional): Rate parameters for each basis function. Defaults to None.
        """
        super().__init__(domain, K)
        if theta is not None:
            if not (len(set(theta)) == len(theta) and len(theta) == self.K):
                raise ValueError('theta must have unique values of length K')
            self.theta = theta
        else:
            self.theta = tuple(np.arange(self.K))

    @property
    def theta(self):
        """Getter for the theta attribute

        """
        return self.__theta

    @theta.setter
    def theta(self, theta):
        """Setter for the theta attribute.

        Args:
            theta (tuple): Rate vector for the basis system. Must be of length K and have all unique elements.

        Raises:
             ValueError if theta is not of length K and contains only unique elements.

        """
        if not (len(set(theta)) == len(theta) and len(theta) == self.K):
            raise ValueError('theta must have unique values of length K')
        self.__theta = theta

    def _evaluate_basis(self, x, q):
        """Evaluate the qth derivative of all basis functions at locations x for the Exponential basis system.

        The qth derivative of basis function :math:`B_k(t)` is given by:
        .. math:
            \frac{d^{(q)}B_k(t)}{dt} = \\theta^q B_k(t)

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int): The order of the derivative to take of the basis functions.

        Returns:
            (np.ndarray): A :math:`n \\times K` matrix with :math:`k^\\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        """
        expon_vecs = np.exp(np.outer(x, self.theta))
        return np.power(self.theta, q) * expon_vecs if q != 0 else expon_vecs

    def penalty(self, q, k=12):
        """Calculate the qth order penalty matrix for the basis system.

        The form of the penalty matrix is given by:

        .. math::
            P = [p_{kl}]

        .. math::
            p_{kl} = \int \\theta_k^q \\theta_l^q B_k(t) B_l(t)dt

        where :math:`B_k` is the :math:`k^\\text{th}` basis function.

        Args:
            q (int): The order of the derivative of the penalty matrix.

            k (int, Optional): Number of samples for romberg integration calculated by :math:`2^k + 1`. Defaults to 12.

        Returns:
            (np.ndarray): A :math:`K \\times K` matrix with elements given by :math:`p_{kl}`.

        """
        rs = np.add.outer(self.theta, self.theta)
        P = np.divide(np.outer(self.theta, self.theta) ** q, rs, np.zeros((self.K, self.K)), where=rs != 0) * (
                np.exp(rs * self.domain[1]) - np.exp(rs * self.domain[0]))
        if q == 0:
            P[rs == 0] = 0
        return P


class Fourier(Basis):
    """Representation of the univariate fourier basis system.

    Basis system is specified as the collection :math:`\{B_k\}_{k=1}^K` where:

    .. math::
        B_0(t) =  \\sqrt{|T|}^{-1}

    .. math::
        B_{2r-1}(t) = \\sqrt{ 0.5|T|}^{-1} \\sin ( \omega \pi r t )

    .. math::
        B_{2r}(t) = \\sqrt{ 0.5|T|}^{-1} \\cos ( \omega \pi r t )

    where :math:`\omega` is :math:`2\pi` divided by the period of the basis.

    Attributes:
        domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

        K (int): Number of basis functions to use in the basis system.

        period: (float): The period of the periodic fourier functions.

    """

    def __init__(self, domain, K, period=None):
        """Inits the fourier basis system.

        Args:
             domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

            K (int): Number of basis functions to use in the basis system. Must be odd and greater than one.

            period: (float, Optional): The period of the periodic fourier functions. Defaults to the length of the
                domain. Must be positive.

        Raises:
            ValueError: If K is not odd or greater than one.

        """

        if not (K % 2 != 0 and K > 1):
            raise ValueError('K must be odd and greater than one.')
        super().__init__(domain, K)
        self.period = period if period is not None else float(np.diff(self.domain))

    @property
    def period(self):
        """Getter for the period attribute

        """

        return self.__period

    @period.setter
    def period(self, period):
        """Setter for the period attribute.

        Args:
            period (float): The period of the periodic fourier functions. Defaults to the length of the domain.

        Raises:
            ValueError: If period is not positive.

        """

        if period <= 0:
            raise ValueError('Period must be positive')
        self.__period = period

    def _evaluate_basis(self, x, q):
        """"Evaluate the qth derivative of all basis functions at locations x for the Fourier basis system.

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int): The order of the derivative to take of the basis functions.

        Returns:
            (np.ndarray): A :math:`n \\times K` matrix with :math:`k^\\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        """

        omega = 2 * np.pi / self.period
        r = np.arange((self.K + 1) // 2)
        c = np.exp(1j * np.outer(x, r * omega))
        c *= np.outer(np.ones(len(x)), (-1) ** (q // 2) * (np.arange((self.K + 1) // 2) * omega) ** q)
        if q % 2 == 0:
            B = np.concatenate([c.T[0].real[:, np.newaxis] / np.sqrt(2)] + [
                np.concatenate([c[:, i].imag[:, np.newaxis], c[:, i].real[:, np.newaxis]], axis=-1) for i in
                np.arange(1, c.shape[-1])], axis=-1)
        else:
            B = np.concatenate([c.T[0].real[:, np.newaxis] / np.sqrt(2)] + [
                np.concatenate([c[:, i].real[:, np.newaxis], -1.0 * c[:, i].imag[:, np.newaxis]], axis=-1) for i in
                np.arange(1, c.shape[-1])], axis=-1)
        return B / np.sqrt(self.period / 2)

    def penalty(self, q, k=12):
        """Calculate the qth order penalty matrix for the basis system.

        The form of the penalty matrix is given by:

        .. math::
            P = [p_{kl}]

        .. math::
            p_{kl} = \int B_k^{(q)}(t) B_l^{(q)}(t)dt

        where :math:`B_k` is the :math:`k^\\text{th}` basis function. In the case where the period of the basis system
        is equal to the length of the domain we can analytically calculate the above, otherwise we use an romberg
        numerical approximation to the integral which is controlled by parameter ``k``.

        Args:
            q (int): The order of the derivative of the penalty matrix.

            k (int, Optional): Number of samples for romberg integration calculated by :math:`2^k + 1`. Defaults to 12.

        Returns:
            (np.ndarray): A :math:`K \\times K` matrix with elements given by :math:`p_{kl}`.

        """
        if not np.isclose(self.period, np.diff(self.domain)):
            x = np.linspace(*self.domain, 2 ** k + 1)
            dx = np.divide(np.diff(self.domain), 2 ** k)
            phi_mat = self(x, q)
            cross_phi_mat = np.einsum('ij, ik -> ijk', phi_mat, phi_mat)
            integrals = []
            for i in np.arange(self.K):
                integrals.append([romb(y=cross_phi_mat[:, i, j], dx=dx, axis=0) for j in np.arange(self.K)])
            inner_product = np.squeeze(np.stack(integrals))
        else:
            omega = 2 * np.pi / self.period
            inner_prod_diag = np.zeros(self.K)
            if q == 0:
                inner_prod_diag[0] = self.period / 2.0
            index_o = np.arange(1, self.K - 1, 2)
            index_e = index_o + 1
            fac = (self.period / 2.0) * (index_e * omega / 2.0) ** (2 * q)
            inner_prod_diag[index_e] = fac
            inner_prod_diag[index_o] = fac
            inner_product = np.diag(inner_prod_diag * 2 / self.period)
        return inner_product


class Bspline(Basis):
    """Representation of the univariate fourier basis system.

    Basis system is specified as the collection :math:`\{B_k,\}_{k=1}^K` where :math:`B_k` is the order :math:`m`
    B-spline function with knot vector :math:`\\tau`.

    Attributes:
        domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

        K (int): Number of basis functions to use in the basis system.

        order (int): The order of the Bspline basis functions.

        knots (tuple): The full knot vector for the Bspline basis functions.

    """

    def __init__(self, domain, K, order, knots=None):
        """Init the Bspline basis system.

        Args:
            domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

            K (int): Number of basis functions to use in the basis system.

            order (int): The order of the Bspline basis functions.

            knots (tuple, Optional): The full knot vector for the Bspline basis functions. Defaults to linearly spaced
                knots over the domain.

        """
        super().__init__(domain, K)
        self.order = order
        self.knots = knots

    @property
    def order(self):
        """Getter for the order property.

        """
        return self.__order

    @order.setter
    def order(self, order):
        """Setter for the order attribute.

        Args:
            order (int): The order of the B-spline basis functions.

        """
        self.__order = int(order)

    @property
    def knots(self):
        """Getter for knots property.

        """
        return self.__knots

    @knots.setter
    def knots(self, knots):
        """Setter for knots property

        Raises:
            ValueError: If knot vector not of correct length.
        """
        if knots is not None and len(knots) != self.K + 1 - self.order:
            raise ValueError("Knot vector isn't correct length")
        self.__knots = knots if knots is not None else self.default_knots()

    def default_knots(self):
        """Calculate default knot placement for B-spline basis system.

        Returns:
            (np.ndarray): The full knot vector of the B-spline basis system.

        """
        L = self.K - self.order
        tau = np.linspace(*self.domain, L + 2)
        return np.pad(tau, (self.order - 1, self.order - 1), 'edge')

    def _bspline_basis_function(self, i, x, q):
        """Compute the ith basis function for the bspline representation evaluated at positions x.

        Args:
            i (int): The basis function to evaluate.

            x (array_like): The points to evaluate the basis function at.

            q (int): The derivative of the basis function to evaluate.

        Returns:
             basis_eval (array_like): The derivative of the ith basis function evaluated at points x.

        """
        c = np.zeros(self.K)
        c[i] += 1
        return splev(x, (self.knots, c, self.order - 1), der=q)

    def _evaluate_basis(self, x, q):
        """"Evaluate the qth derivative of all basis functions at locations x for the B-spline basis system.

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int): The order of the derivative to take of the basis functions.

        Returns:
            (np.ndarray): A :math:`n \\times K` matrix with :math:`k^\\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        Raises:
            ValueError: If the order of derivative is greater than or equal to the order of the B-spline basis system.

        """
        if q >= self.order:
            raise ValueError("The order of derivative must be less than the order of the B-spline system.")
        return np.array([self._bspline_basis_function(i, x, q) for i in np.arange(self.K)]).T

    def penalty(self, q, k=12):
        """Calculate the qth order penalty matrix for the basis system.

        The form of the penalty matrix is given by:

        .. math::
            P = [p_{kl}]

        .. math::
            p_{kl} = \int B_k^{(q)}(t) B_l^{(q)}(t)dt

        where :math:`B_k` is the :math:`k^\\text{th}` basis function. In the case where the period of the basis system
        is equal to the length of the domain we can analytically calculate the above, otherwise we use an romberg
        numerical approximation to the integral which is controlled by parameter ``k``.

        Args:
            q (int): The order of the derivative of the penalty matrix.

            k (int, Optional): Number of samples for romberg integration calculated by :math:`2^k + 1`. Defaults to 12.

        Returns:
            (np.ndarray): A :math:`K \\times K` matrix with elements given by :math:`p_{kl}`.

        Raises:
            ValueError: If order of derivative for the penalty is greater than or equal to the order of the B-spline
                basis functions.

        """
        x = np.linspace(*self.domain, 2 ** k + 1)
        dx = np.divide(np.diff(self.domain), 2 ** k)
        phi_mat = self(x, q)
        cross_phi_mat = np.einsum('ij, ik -> ijk', phi_mat, phi_mat)

        integrals = []
        for i in np.arange(self.K):
            integrals.append([
                romb(y=cross_phi_mat[:, i, j], dx=dx, axis=0) for j in np.arange(self.K)])
        return np.squeeze(np.stack(integrals))
