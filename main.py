from Benchmarking import *
from ODE_Model_Function import *
from test_functions import *
from visualisation import *


if __name__ == "__main__":
  # Plotting code from benchmarks.py
  #plot_benchmark()

  # Plotting code from ODE_Model_Function.py
  # Plot just current time
  plot_initial_attempt(back_date = True)
  plot_second_attempt(back_date = True)

  plot_final_model()

  # Plot model without uncertainties
  Future_Productions = [10000, 0, 20000, 5000]
  Future_Time = 2080
  Labels = [
      "Current Production",
      "Cease all production",
      "Double current production",
      "Half Current Production",
  ]
  plot_model(Future_Productions, Future_Time, Labels, uncertainty = False)

  #Plot with uncertainties
  tT0, xT0, tP0, xP0 = plot_model(Future_Productions, Future_Time, Labels)
  
  plot_misfit(tP0, xP0, tT0, xT0)

  # Plotting code from visualisation.py
  plot_visualisations()
  

  

