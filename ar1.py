"""Functions related to AR(1) processes

This file contains sampling, fitting and log-likelyhood functions
for both regularly and irregularly sampled AR(1) processes.

Note
----
The irregularly sampled AR(1) process is parameterised differently from the
regularly sampled AR(1) process, i.e. with the decorrelation time \tau and
the total standard deviation of the process instead of the lag-one
autocorrelation and the standard deviation of the innovation process.
They can be converted into each other with

    \tau = 1 / \ln(phi)

and

    \sigma_e^2 = \sigma^2 * (1 - \phi**2)


References
----------
.. Wilks, Daniel S. 2005. Statiscial Methods in the Atmospheric Sciences.
   International Geophysics. 2nd ed. Vol. 100. Academic Press.

.. Mudelsee, Manfred. 2010. Climate Time Series Analysis.
   Edited by Lawrence A Mysak and Kevin Hamilton. 1st ed. Vol. 42.
   Atmospheric and Oceanographic Sciences Library.
   Dordrecht: Springer Netherlands.
   doi:10.1007/978-90-481-9482-7.

Tobias Erhardt 2018
"""
from scipy.optimize import minimize
import numpy as np


def normal_like(x, mu=0.0, sigma=1.0):
    """Unnormalized univariate normal log-likelyhood function

        lnp({x}|mu, sigma)

    Parameters
    ----------
    x
    mu : float
        Mean of the distribution
    sigma : float
        Standard deviation of the distribution

    Returns
    -------
    lnp : float
        Unnormalized log-likelyhood of {x}|mu, sigma

    Note
    ----
    This function is the unnormalized equivalet of
    scipy.stats.norm.logpdf and is mainly used because
    the scipy implementation is significantly slower
    """
    k = 2 * np.pi * sigma ** 2
    d = (x - mu) / sigma
    lnp = -0.5 * np.nansum(np.log(k) + d ** 2)
    return lnp


def ar1_like(y, phi, sigma_e=1.0):
    """AR(1) log-likelihood for evenly sampled series x

    Parameters
    ----------
    x : array
        Series of observations/noise
    phi : float
        autocorrelation factor of AR(1) process (0 <=k < 1)
    sigma_e : float
        standard deviation of AR(1) innovations

    Returns
    -------
    lnp : array
        Log-likelyhood of the observations given the parameters of the
        AR(1) process \ln p({x}|k, sigma)
    """
    xim1 = y[:-1]
    xi   = y[1:]
    lnp = normal_like(xi, phi * xim1, sigma=sigma_e)
    sigma_lim = np.sqrt(sigma_e**2 / (1 - phi**2))
    lnp += normal_like(y[0], mu=0.0, sigma=sigma_lim)
    return lnp


def ar1_t_like(t, y, tau, sigma=1.0):
    """Log-likelihood of unevenly sampled AR(1) process

    Parameters
    ----------
    t : np.ndarray
        Observation times (len(t) == len(y))
    y : np.ndarray
        Observations / noise observed at time t
    tau : float
        Autocorrelation time
    sigma : float
        Standard deviation of y


    Returns
    -------
    lnp : np.ndarray
        Log-likelyhood of the observations given the parameters of the
        AR(1) process

    References
    ----------
    .. Mudelsee, Manfred. 2010. Climate Time Series Analysis.
       Edited by Lawrence A Mysak and Kevin Hamilton. 1st ed. Vol. 42.
       Atmospheric and Oceanographic Sciences Library.
       Dordrecht: Springer Netherlands.
       doi:10.1007/978-90-481-9482-7.
    """
    phi = np.exp(-np.abs(np.diff(t)) / tau)
    sigma_e = np.sqrt(sigma ** 2 * (1 - phi ** 2))
    yim1 = y[:-1]
    yi = y[1:]
    lnp = normal_like(yi, phi * yim1, sigma=sigma_e)
    lnp += normal_like(y[0], mu=0.0, sigma=sigma)
    return lnp


def sample_ar1(n, phi, sigma_e=1.0, size=1):
    """Generate samples from an AR(1) process

    Parameters
    ----------
    n : int
        Length of the series that is drawn from the AR(1) process
    phi : float
        Lag one autocorrelation. The decorrelation time is given by
        tau = - 1 / ln(phi)
    sigma_e : float
        Standard deviation of the AR(1) innovations
    size : int
        Number of realizations to generate

    Returns
    -------
    x : np.ndarray
        Realizations from the AR(1) process. x.shape = (size, n)
    """
    x = sigma_e * np.random.randn(n, size)
    x[0] = x[0] * np.sqrt(1 / (1 - phi**2))
    for i in range(1, n):
        x[i] = x[i] + x[i - 1] * phi
    return x.T.squeeze()


def sample_ar1_t(t, tau, sigma=1.0, size=1):
    """Generate samples from an unevenly sampled AR(1) process

    Parameters
    ----------
    t : np.ndarray
        time axis along which the AR(1) process is samples
    tau : float
        Decorrelation time of the AR(1) process. The autocorrelation
        coefficient is given as phi = - 1 / ln(tau) for a sample
        frequency of 1.
    sigma : float
        Standard deviation of the AR(1) sample. x.std() = sigma
    size : int
        Number of realizations to generate

    Returns
    -------
    x : np.ndarray
        Realizations from the AR(1) process. x.shape = (size, len(t))
    """
    dt = np.diff(t)
    x = sigma * np.random.randn(len(t), size)
    for i in range(1, len(t)):
        a = np.exp(-dt[i - 1] / tau)
        s = np.sqrt((1 - a ** 2))
        x[i] = x[i] * s + x[i - 1] * a
    return x.T.squeeze()


def fit_ar1(y):
    """Fit an AR(1) process to an evenly sampled time series
    using the lag-1 autocorrelation

    Parameters
    ----------
    y : np.ndarray
        Evenly sampled time series to fit AR(1) to

    Returns
    -------
    phi : float
        Lag-1 autocorrelation coefficient. To calculate the decorrelation
        time use -dt/np.log(phi), where dt is the sampling interval of y
    sigma_e : float
        Standard deviation of the innovations

    Note
    ----
    Note that this function fits the AR(1) process after removing the average
    value of y and returns the AR(1) parameter relative to this mean.
    """
    ym = y - np.mean(y)
    yi   = ym[1:]
    yim1 = ym[:-1]
    var_yim1, cov_yy, _, var_yi = np.cov(yim1, yi).flat
    phi = cov_yy / np.sqrt(var_yim1 * var_yi)
    sigma_e = np.sqrt(np.var(y) * (1 - phi**2))
    return phi, sigma_e


def fit_ar1_t(t, y):
    """Fit AR(1) process to irregularly sampled data
    using numerical log-likelyhood maximization

    Parameters
    ----------
    t : np.ndarray
        Time reference of time series y
    y : np.ndarray
        Time series data to fit the AR(1) process to

    Returns
    -------
    tau : float
        Decorrelation time of the AR(1) process. To obtain the
        lag-1 autocorrelation parameter use phi = np.exp(-1/tau)
    sigma : float
        Standard deviation of y

    Note
    ----
    Note that this function fits the AR(1) process
    after removing the average value of y and returns
    the AR(1) parameter relative to this mean.
    """
    lntau0 = np.log(np.mean(np.diff(t)))
    sigma = np.std(y)
    yr = y - np.mean(y)
    nlnp = lambda lntau, sigma: -1.0 * ar1_t_like(t, yr, np.exp(lntau), sigma)
    res = minimize(nlnp, lntau0, args=(sigma,), method='Nelder-Mead')
    tau = np.exp(res.x.squeeze())
    return tau, sigma


def calc_ar_neff(phi, n=1):
    """Calculate number of effective, i.e. independent samples
    for a given lag-one autocorrelation

    Parameters
    ----------
    phi : float
        Lag-one autocorrelation parameter
    n :
        Number of datapoints in the time series

    Returns
    -------
    neff : float
        Number of independent observations


    Reference
    ---------
    .. Wilks, D.S., 2006, Statistical methods in the atmospheric sciences,
       Elsevier, 2nd Edition, Chapter 5, p. 144
    """
    neff = n * (1 - phi) / (1 + phi)
    return neff


def calc_ar1_dof_pearsonr(phi1, phi2=1.0, n=1):
    """Calculate degrees of freedom for correlation between
    two autocorrelated time series with autocorrelation coefficients
    phi1 and phi2

    Parameter
    ----------
    phi1, phi2 : float
        Lag-one autocorrelations of the two timeseries
    n : int
        Length of the time series

    Return
    -------
    dof_eff : float
        Degrees of freedom to be used for the null-hypothesis significance
        test of the pearson correlation coefficient

    References
    ----------
      ..Hu, J. et al., Earth Planet. Sci. Lett., 2017,
        Correlation-based interpretations of paleocliamte data -
        where statistics meet past climates.
    """
    dof_eff = n * (1 - phi1 * phi2) / (1 + phi1 * phi2)
    return dof_eff
