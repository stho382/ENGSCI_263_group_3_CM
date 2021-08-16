from ast import fix_missing_locations
import numpy as np
from matplotlib import pyplot as plt
import math as mt
import pandas as pd



if __name__ == "__main__":
    water_level_df = pd.read_csv("https://raw.githubusercontent.com/stho382/ENGSCI_263_group_3_CM/main/data/gr_p.txt?token=ATINDQZDQ52VEWZ2KCN2Y3DBDWUPK")
    water_level_df['Pressure'] = (water_level_df['water level (m)']-296.85)/10

    fx,ax1 = plt.subplots(nrows=1,ncols=1)   
    ax1.plot(water_level_df['year'],water_level_df['Pressure'],'ko')
    plt.show()