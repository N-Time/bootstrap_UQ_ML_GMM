# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:14:16 2023

@author: Vincent NT, ylpxdy@live.com

Modified subset for ML-GMM-QU
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
# %% Assign region label
# %%% original database
# # origin data with removing labels
# data0 = pd.read_excel('D:\Wen\Jupyter_working_path\Project\PEER_NGA_West2_selected\data\PEER_NGA_West2_8ex_SingleLabeled_230327.xlsx')
# removed data without removing labels
data0 = pd.read_excel('D:\Wen\Jupyter_working_path\Project\PEER_NGA_West2_selected\data\PEER_NGA_West2_size_14_6k_230602.xlsx')

# %%% subset of -1 & -2 with region label
data1 = pd.read_excel(r'D:\Wen\Jupyter_working_path\Project\Nonparametric_GMM\W2for3thpaper.xlsx')

# %%% region label for the original database
regionLabel = np.empty((len(data0),1))
regionLabel[:] = -999

for i in range(len(data1)):  # take a key
    # boolean of data1 key in data0
    temp = data0['Record Sequence Number'] == data1['Record Sequence Number'].iloc[i]
    regionLabel[temp] = data1['Region'].iloc[i]  #  assgin the value for data0

# %%% inset the region label column in the original database
data0['Region'] = regionLabel

# %% Check distance (error > 5km, assign -999)
data = data0
calculated_Repi=np.zeros((len(data),1))
for i in range(len(data)):
    if data['Hypocenter Latitude (deg)'].iloc[i] != -999 and data['Station Latitude'].iloc[i] != -999:
        calculated_Repi[i,0]=geodesic((data['Hypocenter Latitude (deg)'].iloc[i],data['Hypocenter Longitude (deg)'].iloc[i]),\
                                      (data['Station Latitude'].iloc[i],data['Station Longitude'].iloc[i])).km
    else:  # 震中经纬缺失
        calculated_Repi[i,0]=data['EpiD (km)'].iloc[i]
        print(data['Hypocenter Latitude (deg)'].iloc[i],data['Station Latitude'].iloc[i])  
          
Repi_residual=abs(data.loc[:,'EpiD (km)':'EpiD (km)']-calculated_Repi)
large_residual=Repi_residual['EpiD (km)']>5  # 超过 5 km Repi设为-999
data.loc[large_residual,['EpiD (km)']] = -999


# %% write the pd. to excel
# # filter by distance check
# data.to_excel('./PEER_NGA_West2_8exLabel_Region_230327.xlsx')

# no distance check
data0.to_excel('./PEER_NGA_West2_size_14_6k_Region_230602.xlsx')