import numpy as np

def trueZ_to_specZ(true_z, year):
    """
    Convert true redshifts to spectroscopic redshift realizations.

    This function is intended to generate mock spectroscopic redshift
    measurements by perturbing true redshifts with a Gaussian error model
    whose width decreases with survey duration. The scatter is assumed to
    scale as (1 + z) and improves with total observing time following a
    sqrt(time) relation.

    Parameters
    ----------
    true_z : array_like
        Array of true (cosmological) redshifts.
    year : float
        Effective survey duration in years used to scale the redshift
        uncertainty.

    Returns
    -------
    ndarray
        Array of spectroscopic redshift realizations.

    Notes
    -----

    """
    z_adjust = np.random.normal(
        loc=true_z, scale=0.0004 * (1 + true_z), size=len(true_z)
    )
    return z_adjust


def trueZ_to_photoZ(true_z, year, modeled=False):
    """
    Convert true redshifts to photometric redshift realizations.

    This function generates mock photometric redshifts by applying Gaussian
    scatter to the true redshift distribution. The scatter scales with
    (1 + z) and improves with survey duration as sqrt(10 / year). Two noise
    regimes are supported: a nominal photometric error model and an idealized
    "modeled" case with reduced scatter.

    Parameters
    ----------
    true_z : array_like
        Array of true (cosmological) redshifts.
    year : float
        Effective survey duration in years.
    modeled : bool, optional
        If True, use a reduced photometric error model (prefactor = 0.01).
        If False, use a nominal LSST-like photometric scatter (prefactor = 0.04).

    Returns
    -------
    ndarray
        Array of photometric redshift realizations.
    """
    if modeled:
        prefactor = 0.01
    else:
        prefactor = 0.04
    time_term = np.sqrt(10 / year)
    z_adjust = np.random.normal(
        loc=true_z, scale=prefactor * time_term * (1 + true_z), size=len(true_z)
    )
    return z_adjust
