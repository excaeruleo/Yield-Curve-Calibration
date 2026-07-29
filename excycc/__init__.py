import sys
sys.path.insert(0, "../src")
from discountFactorParYield import calcDisFact
from discountFactorParYield import graph as graphDF
from contCompYields import calcYields
from contCompYields import graph as graphCC
from forwardCurve import forwardCurve
from forwardCurve import graph as graphFC
from termForwardCurves import graph as graphTFC
from vasicekCalibration import graph as graphVC

def main():
  periodDF = calcDisFact('../csvs/daily-treasury-par-yield-curve-rates.csv')
  print(periodDF)
  graphDF(periodDF)
  yieldsDF = calcYields('../csvs/daily-treasury-par-yield-curve-rates.csv')
  print(yieldsDF)
  graphCC(yieldsDF)
  forwardCurveDF = forwardCurve('../csvs/daily-treasury-par-yield-curve-rates.csv')
  print(forwardCurveDF)
  graphFC(forwardCurveDF)
  termForwardCurveDF = forwardCurve('../csvs/daily-treasury-par-yield-curve-rates.csv')
  graphTFC(termForwardCurveDF)
  vasicekCalibrationDF = calcDisFact('../csvs/daily-treasury-par-yield-curve-rates.csv')
  graphVC(vasicekCalibrationDF)
  pass

if __name__ == "__main__":
  main()
