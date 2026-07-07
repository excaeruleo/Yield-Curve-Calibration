import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from discountFactorParYield import calcDisFact
from forward_curve import ForwardCurve, par_yield_maturities


def graph(df: pd.DataFrame):
    # tenors and times used in plot
    tenors = par_yield_maturities(df.columns)
    times = np.linspace(tenors[0], tenors[-1], num=500)
    # for each date
    for date in df.index:
        # discount factors + forward curve
        dfs = df.loc[date].values
        curve = ForwardCurve(dfs, tenors)
        # plot
        plt.plot(
            times,
            curve(times),
            label=f"{date} ({curve.interpolator().__name__})"
        )
    # set plot properties
    plt.xlabel("Maturity (Years)")
    plt.ylabel(f"Forward Rate")
    plt.title(f"Instantaneous Treasury Forward Curves")
    # show grid + legend + display
    plt.grid()
    plt.legend()
    plt.show()


def main() -> int:
    graph(calcDisFact('../csvs/daily-treasury-par-yield-curve-rates.csv'))
    return 0


if __name__ == "__main__":
    sys.exit(main())
