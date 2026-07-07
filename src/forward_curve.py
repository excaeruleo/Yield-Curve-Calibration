"""Module for a forward curve class.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from typing import Callable, Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator

# interpolator type hint
Interpolator = Callable[
    [Iterable[float], Iterable[float]],
    Callable[[Iterable[float]], np.ndarray]
]


def par_yield_maturities(names: Iterable[str]) -> list[float]:
    """Convert Treasury par yield column names into year fractions.

    Months are converted into actual years by dividing by 12.

    Parameters
    ----------
    names : Iterable[str]
        Maturity strings, e.g. "1 Mo", "2 Yr"
    """
    years = []
    # pre-emptively split "1 Mo" into number and unit
    for num, unit in (name.split() for name in names):
        if unit == "Mo":
            years.append(float(num) / 12)
        # assume years if not month
        else:
            years.append(float(num))
    return years


class ForwardCurve:
    """Forward curve class for computing continuously compounded forward rates.

    Parameters
    ----------
    dfs : Iterable[float]
        Discount factors, i.e. zero coupon bond prices from bootstrapping
    ts : Iterable[float]
        Discount factor maturities in fractional years
    interp : Interpolator, default=scipy.interpolate.PchipInterpolator
        Interpolator for the negative log-bond prices. This can be any
        appropriate callable, e.g. the Akima1DInterpolator, etc.
    """

    def __init__(
        self,
        dfs: Iterable[float],
        ts: Iterable[float],
        interp: Interpolator = PchipInterpolator
    ):
        self._dfs = np.array(dfs)
        self._ts = np.array(ts)
        self._interp = interp
        # curve consists of negative log-bond prices
        self._curve = interp(self._ts, -np.log(self._dfs))

    def discount_factors(self) -> np.ndarray:
        """Return the discount factors used to construct the forward curve."""
        return self._dfs

    def maturities(self) -> np.ndarray:
        """Return the discount factories maturities in years."""
        return self._ts

    def interpolator(self) -> type:
        """Return the interpolator type used."""
        return self._interp

    def __call__(
        self,
        ts: Iterable[float],
        t: float | None = None
    ) -> np.ndarray:
        """Return the [term] forward rates at the given maturities.

        If no term is given the instaneous forward rates are returned.

        Parameters
        ----------
        ts : Iterable[float]:
            Time points to request t-forward rates for
        t : float, default=None
            Forward rate term, e.g. 1 year, etc. If not specified then the term
            is taken to be an instant, yielding instantaneous forward rates
        """
        # instantaneous forward rates
        # note: technically, should also check that t is nonnegative
        if not t:
            return self._curve.derivative()(ts)
        # term forward rates
        else:
            return (self._curve(ts + t) - self._curve(ts)) / t
