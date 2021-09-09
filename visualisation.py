import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_visualisations():
  """
    Develops a matplotlib plot to visualise the data that was initially provided for this project 

    Parameters:
    -----------
    none

    Returns:
    -----------
    none

    Notes:
    -----------
    Referenced from - https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales

  """
  # Getting the data for analysis
  water_level_df = pd.read_csv("" + os.getcwd() + os.sep + "data" + os.sep + "gr_p.txt")

  total_production_df = pd.read_csv("" + os.getcwd() + os.sep + "data" + os.sep + "gr_q1.txt")

  rhyolite_production_df = pd.read_csv("" + os.getcwd() + os.sep + "data" + os.sep + "gr_q2.txt")

  temperature_df = pd.read_csv("" + os.getcwd() + os.sep + "data" + os.sep + "gr_T.txt")

  # Creating figure and subplot
  fig, host = plt.subplots(figsize=(15,6))
      
  # Duplicating y-axis to for the purposes of adding a third axis
  par1 = host.twinx()
  par2 = host.twinx()

  # Setting labels for the independent variable and the three dependent variables 
  host.set_xlabel("years")
  host.set_ylabel("production rate (tonnes/day)")
  par1.set_ylabel("water level (m)")
  par2.set_ylabel("temperature (degC)")

  # Setting colours for the four line plots to be made (total production rate, rhyolite production rate, water level and temperature)
  color1 = plt.cm.viridis(0)
  color2 = plt.cm.viridis(0.5)
  color3 = plt.cm.viridis(.9)
  color4 = plt.cm.magma(0)

  # Plotting the plots to their own set of axes
  # p1 refers to the x-axis (years)
  p1, = host.plot(total_production_df["year"], total_production_df.iloc[:,1], color=color1, label="total production rate (tonnes/day)")
  #p2 refers to the first y-axis (water level)
  p2, = par1.plot(water_level_df["year"], water_level_df.iloc[:,1], color=color2, label="water level (m)")
  #p3 refers to the secondary y-axis (temperature)
  p3, = par2.plot(temperature_df["year"], temperature_df.iloc[:,1], color=color3, label="temperature (degC)")
  #p4 refers to the third y-axis (production rate)
  p4, = host.plot(rhyolite_production_df["year"], rhyolite_production_df.iloc[:,1], color=color4, label="rhyolite production rate (tonnes/day)")

  # Adding in the legends to the pin the bottom right hand corner of the plot
  lns = [p1, p2, p3, p4]
  host.legend(handles=lns, loc='lower right')

  # Offsetting the temperature axis from the right hand side of the plot by 60 pixels
  par2.spines['right'].set_position(('outward', 60))

  # Assigning colours to each of the axes labels to distinguish them from each other
  host.yaxis.label.set_color(p1.get_color())
  par1.yaxis.label.set_color(p2.get_color())
  par2.yaxis.label.set_color(p3.get_color())

  # Adjust spacings w.r.t. figsize
  fig.tight_layout()

  # Saving / showing the plot
  save_figure = False
  if not save_figure:
      plt.show()
  else:
      fig.savefig("pyplot_multiple_y-axis.png")

