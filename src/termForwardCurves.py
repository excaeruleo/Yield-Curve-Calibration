import matplotlib.pyplot as plt
import numpy as np
from discountFactorParYield import calcDisFact
from forwardCurve import forwardCurve
from scipy.interpolate import CubicSpline


def graph(df):
  maturities = df.columns
  maturitiesNumeric = []
  for maturity in maturities:
    if "Mo" in maturity:
      str = maturity.split()
      maturity = round(float(str[0]) / 12, 3)
      maturitiesNumeric.append(maturity)
    else:
      str = maturity.split()
      maturity = round(float(str[0]))
      maturitiesNumeric.append(maturity)

  tenors = [1, 5, 10, 20]
  denseTimes = np.linspace(maturitiesNumeric[0], maturitiesNumeric[-1], 500)
  _, ((ax1, ax5), (ax10, ax20)) = plt.subplots(2, 2)

  for date in df.index:
    discount = df.loc[date].to_numpy(dtype = float)
    spline = CubicSpline(maturitiesNumeric, np.log(discount))

    for tau, ax in zip(tenors, (ax1, ax5, ax10, ax20)):
      t = denseTimes[denseTimes + tau <= maturitiesNumeric[-1]]
      forward = (spline(t) - spline(t + tau)) / tau
      ax.set_xlabel('t')
      ax.set_ylabel(f"f(0, t, t + {tau})")
      ax.set_title(f"{tau}y continuously compounded Treasury forward rates")
      ax.grid(True)
      ax.plot(t, forward, label = date)
      ax.legend()

  plt.show()


def main():
  forwardCurves = forwardCurve('../csvs/daily-treasury-par-yield-curve-rates.csv')
  print(forwardCurves)
  graph(forwardCurves)
  pass

if __name__ == "__main__":
  main()
