"""Plots multiple continuously compounded term forward rate curves.

Curve built from the original Treasury par rate bootstrapped discount factors.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from pathlib import Path
import sys
from typing import Iterable

from matplotlib import pyplot as plt
import numpy as np

# FIXME: should use actual Python package for organizing code
from discountFactorParYield import calcDisFact
from forward_curve import ForwardCurve, par_yield_maturities

# path to current directory + CSVs directory
_cur_dir = Path(__file__).parent
_csv_dir = (_cur_dir / ".." / "csvs").resolve()


def main(args: Iterable[str] | None = None) -> int:
    """Main function for the script.

    This produces a plot of 1y, 5y, 10y, and 20y continuously compounded
    Treasury term forward rates given bootstrapped par yield curve rates.

    Parameters
    ----------
    args : Iterable[str], default=None
        Command-line arguments to parse

    Returns
    -------
    int
        Exit code
    """
    # get discount factors
    df = calcDisFact(_csv_dir / "daily-treasury-par-yield-curve-rates.csv")
    # forward curve terms + axes subplots
    terms = [1, 5, 10, 20]
    _, ((ax1, ax2), (ax5, ax10)) = plt.subplots(nrows=2, ncols=2)
    # maturities + time points for smooth plots
    tenors = par_yield_maturities(df.columns)
    times = np.linspace(tenors[0], tenors[-1], num=500)
    # for each term
    for term, ax in zip(terms, (ax1, ax2, ax5, ax10)):
        # set axes properties
        ax.set_xlabel("t")
        ax.set_ylabel(f"f(0, t, t + {term})")
        ax.set_title(f"{term}y continuously compounded Treasury forward curves")
        ax.grid()
        # plot forward rate curve for each date
        for date in df.index:
            dfs = df.loc[date, :].values
            curve = ForwardCurve(dfs, tenors)
            ax.plot(times, curve(times, term), label=date)
        # display legend
        ax.legend()
    # display and return
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
