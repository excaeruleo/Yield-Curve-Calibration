import sys
sys.path.insert(0, "")
from discountFactorParYield import calcDisFact
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def forwardCurve(name: str):
  finalDisFacts = calcDisFact(name)
  forwardCurves = pd.DataFrame(index = finalDisFacts.index, columns = finalDisFacts.columns, dtype = float)
  #print(forwardCurves)
  maturities = np.array(forwardCurves.columns)
  #print(maturities) 
  for d, date in enumerate(forwardCurves.index):
    for i, col in enumerate(forwardCurves.columns):
      if "Mo" in col:
        str = col.split()
        year = float(str[0]) / 12
      elif "Yr" in col:
        str = col.split()
        year = float(str[0])
      #disFactDerivative = -math.log(finalDisFacts.loc[date, col])
      #forwardCurves.loc[date, col] = disFactDerivative
      forwardCurves.loc[date, col] = finalDisFacts.loc[date, col]
      #forwardCurves.loc[date, col] = np.interp(disFactDerivative, date, finalDisFacts.loc[date, col])

  return forwardCurves

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
      maturity = float(str[0])
      maturitiesNumeric.append(maturity)
 
  for date in df.index:
    forward = df.loc[date].to_numpy(dtype = float)
    forward = -np.log(forward)
    interp = CubicSpline(maturitiesNumeric, forward).derivative()
    denseTimes = np.linspace(maturitiesNumeric[0], maturitiesNumeric[-1], 500)
    denseForward = interp(denseTimes)
    plt.plot(denseTimes, denseForward, label = date + " Interpolated")
    #plt.plot(maturitiesNumeric, pd.loc[date], marker = 'o', label = date)

  plt.xlabel('Maturity (Converted to years for uniformity)')
  plt.ylabel('Forward Rate')
  plt.title('Forward Curves of Discount Factors')
  plt.legend()
  plt.grid(True)
  plt.show()

def main():
  #yieldsDF = calcYields('daily-treasury-par-yield-curve-rates.csv')
  forwardCurves = forwardCurve('../csvs/daily-treasury-par-yield-curve-rates.csv')
  print(forwardCurves)
  graph(forwardCurves)
  pass

if __name__ == "__main__":
  main()
