import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from typing import Iterable
from discountFactorParYield import calcDisFact
import math

class calcVectors:
  def __init__(self, r: float, a: float, b: float, sigma: float):
    self.r = r
    self.a = a
    self.b = b
    self.sigma = sigma

  def __call__(self, t:np.ndarray) -> np.ndarray:
    va = (1 - np.exp(-self.a * t)) / self.a
    vc = (self.b * (va - t) + 0.5 * np.power(self.sigma / self.a, 2) * (t + 0.5 * 
    (1 - np.exp(-2 * self.a * t)) / self.a - 2 * va))
    return -self.r * va + vc

class optimize:
  def __init__(self, t: Iterable[float], df: Iterable[float], r: float):
    self.t = np.array(t)
    self.df = np.log(df)
    self.r = r
  
  def __call__(self, x:np.ndarray) -> float:
    vlz = calcVectors(self. r, x[0], x[1], x[2])
    y = self.df - vlz(self.t)
    return (y * y).sum()
    
def graph(df: pd.DataFrame):
  maturities = df.columns
  maturitiesNumeric = []
  for maturity in maturities:
    if "Mo" in maturity:
      str = maturity.split()
      maturity = round(float(str[0]) / 12, 3)
      maturitiesNumeric.append(maturity)
    else:
      str = maturity.split()
      maturity = float(str[0])
      maturitiesNumeric.append(maturity)

  times = np.linspace(maturitiesNumeric[0], maturitiesNumeric[-1], 500)
  _, axs = plt.subplots(nrows = 2, ncols = (len(df.index) + 1) // 2)
  axs = [ax for row in axs for ax in row][:len(maturitiesNumeric)]

  for date, ax in zip(df.index, axs):
    dfs = CubicSpline(maturitiesNumeric, df.loc[date].values)
    fwds = CubicSpline(maturitiesNumeric, -np.log(df.loc[date].values)).derivative()
    r = fwds(maturitiesNumeric[0])
    objective = optimize(maturitiesNumeric, df.loc[date].values, r)
    res = minimize(objective, (0.1, 0.2, 0.3), method = "trust-constr", bounds = ((1e-8, None),) * 3)
    logBond = calcVectors(r, res.x[0], res.x[1], res.x[2])
    
    print(f"{date} Vasicek: {res.x} {'OK' if res.success else 'NOT OK'}")
    
    ax.set_xlabel("t")
    ax.set_ylabel("Z(t)")
    ax.set_title(f"{date} bootstrap vs Vasicek Treasury discount curve")
    ax.grid(visible = True)
    ax.plot(times, dfs(times), label = f"{type(dfs).__name__} Z(t)")
    ax.plot(times, np.exp(logBond(times)), label = f"Vasicek Z(t)")
    ax.legend()
  plt.show()

def main():
  dfs = calcDisFact('../csvs/daily-treasury-par-yield-curve-rates.csv')
  print(dfs)
  graph(dfs)  
  pass

if __name__ == "__main__":
  main()
