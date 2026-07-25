"""Calibrates the Vasicek model against bootstrapped Treasury discount factors.

Instantaneous forward curves are constructed as part of the calibration process
and the least-squares bounded optimization is done using SciPy's implementation
of a constrained trust-region algorithm for robustness. The final
"""

from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# script directory + hack for import from script directory
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from discountFactorParYield import calcDisFact  # bootstrapping routine


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


class VasicekLogBond:
    """Vasicek log-bond price for the given maturity.

    This computes the Vasicek log-bond prices given maturities in fractional
    years, current short rate, and choices of the a, b, and sigma parameters.

    Parameters
    ----------
    r : float
        Current short rate or instantaneous forward rate
    a : float
        Vasicek mean-reversion scale
    b : float
        Vasicek long-term mean level
    sigma : float
        Vasicek diffusion constant
    """

    def __init__(self, r: float, a: float, b: float, sigma: float):
        self.r = r
        self.a = a
        self.b = b
        self.sigma = sigma

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Return the Vasicek log-bond price[s] given maturities.

        Parameters
        ----------
        t : np.ndarray
            Maturities in fractional years
        """
        # Vasicek A(t) result vector
        va = (1 - np.exp(-self.a * t)) / self.a
        # Vasicek C(t) result vector
        vc = (
            self.b * (va - t) +
            0.5 * np.pow(self.sigma / self.a, 2) *
            (t + 0.5 * (1 - np.exp(-2 * self.a * t)) / self.a - 2 * va)
        )
        # return Vasicek log-bond price
        return -self.r * va + vc


class VasicekObjective:
    """Vasicek objective function callable.

    This maintains the log-bond prices and maturities needed to evaluate the
    objective during bounded optimization. The objective is suitable for a
    nonlinear least squares routine that can compute finite differences.

    Parameters
    ----------
    t : Iterable[float]
        Maturities in fractional years
    z : Iterable[float]
        Discount factors for each maturity
    r : float
        Current short rate or instantaneous forward rate
    """

    def __init__(self, t: Iterable[float], z: Iterable[float], r: float):
        self.t = np.array(t)
        self.lz = np.log(z)
        self.r = r

    def __call__(self, x: np.ndarray) -> float:
        """Invoke the objective function given the parameters.

        The input vector should only have 3 components for a, b, and sigma. All
        parameters should be positive which requires bounded optimization.

        Parameters
        ----------
        x : np.ndarray
            Vasicek parameters [a, b, sigma] to optimize over
        """
        """
        # Vasicek A(t) result vector
        va = (1 - np.exp(-x[0] * self.t)) / x[0]
        # Vasicek C(t) result vector
        vc = (
            x[1] * (va - self.t) +
            0.5 * np.pow(x[2] / x[0], 2) *
            (self.t + 0.5 * (1 - np.exp(-2 * x[0] * self.t)) / x[0] - 2 * va)
        )
        """
        # use parameters to construct Vasicek log-bond
        vlz = VasicekLogBond(self.r, x[0], x[1], x[2])
        # compute vectorized differences of actual and Vasicek log-bond prices
        y = self.lz - vlz(self.t)
        # square + sum the differences to get the objective's loss
        return (y * y).sum()


def graph(df: pd.DataFrame):
    """Constructs forward curves + graphs the Vasicek discount factors per date.

    Each date is plotted separately, two plots per row, comparing at 500 points
    within the first and last maturities the piecewise cubic interpolated
    bootstrapped discount factors against the calibrated Vasicek discount
    factors recovered by exponentiating -A(t)f(t[0]) + C(t).
    """
    # get year fraction maturities from columns
    t = par_yield_maturities(df.columns)
    # points evaluate discount factors at
    t_plot = np.linspace(t[0], t[-1], 500)
    # create subplots
    # note: +1 and truncation ensures that we always have enough plots
    _, axs = plt.subplots(nrows=2, ncols=(len(df.index) + 1) // 2)
    # flatten axes and ignore any unused axis
    axs = [ax for row in axs for ax in row][:len(t)]
    # for each date + axis
    for date, ax in zip(df.index, axs):
        # interpolate bootstrapped discount factors + instantaneous forwards
        dfs = CubicSpline(t, df.loc[date].values)
        fwds = CubicSpline(t, -np.log(df.loc[date].values)).derivative()
        # calibrate Vasicek against f(t[0]) and discount factors
        # note: instantaneous forward at earliest maturity used as proxy
        r = fwds(t[0])
        obj = VasicekObjective(t, df.loc[date].values, r)
        res = minimize(
            obj,
            # guesses for a, b, sigma. these are fudges and it appears that
            # starting with a larger sigma guess improves stability
            (0.1, 0.02, 0.3),
            # supports the box constraints we need
            # note: we didn't provide a gradient function so the "gradient" will
            # be determined using 2-point finite difference w/ hardcoded step
            method="trust-constr",
            # require all parameters positive since bounds include endpoints
            bounds=((1e-8, None),) * 3
        )
        vlb = VasicekLogBond(r, res.x[0], res.x[1], res.x[2])
        # print optimization results for each date
        print(f"{date} Vasicek: {res.x} {'OK' if res.success else 'NOT OK'}")
        # set axis properties
        ax.set_xlabel("t")
        ax.set_ylabel("Z(t)")
        ax.set_title(f"{date} bootstrap vs. Vasicek Treasury discount curve")
        ax.grid(visible=True)
        # plot interpolated bootstrapped prices + Vasicek prices
        ax.plot(t_plot, dfs(t_plot), label=f"{type(dfs).__name__} Z(t)")
        ax.plot(t_plot, np.exp(vlb(t_plot)), label=f"Vasicek Z(t)")
        # show legend
        ax.legend()
    # show the figure
    plt.show()


def main() -> int:
    """Main function for the script.

    This bootstraps discount factors from par rates, prints the values, and
    produces a graph of the calibrated Vasicek discount factors.
    """
    dfs = calcDisFact(
        _script_dir /
        ".." /
        "csvs" /
        "daily-treasury-par-yield-curve-rates.csv"
    )
    print(dfs)
    graph(dfs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
