# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:46:11 2023

@author: Vincent NT, ylpxdy@live.com

Functions for ML-GMMs UQ
"""

import numpy as np
import pandas as pd

def func_sigma_E(y_true, y_hat_b_im, pick_cvg_b=40, nan_flag=False):
    """
    Estimate the true predicted y_tile,
    Estimate the sigma_E
    """

    y_hat_b_cvg = y_hat_b_im[:,0:pick_cvg_b]
    
    # estimate y_tile
    if nan_flag == True:
        y_tile_cvg = np.nanmean(y_hat_b_cvg, axis=1)
    else:
        y_tile_cvg = np.mean(y_hat_b_cvg, axis=1)
    
    # estimate sigma(N, m) over B
    epsilon_E = y_hat_b_cvg - np.tile(y_tile_cvg, (y_hat_b_cvg.shape[1], 1)).T
    epsilon_T = np.tile(y_true, (y_hat_b_cvg.shape[1], 1)).T - y_hat_b_cvg
    
    if nan_flag == True:
        sigma_E = (np.nansum(epsilon_E**2, axis=1) / (epsilon_E.shape[1] - 1))**0.5
        sigma_T = (np.nansum(epsilon_T**2, axis=1) / (epsilon_T.shape[1] - 1))**0.5
    else:
        sigma_E = (np.sum(epsilon_E**2, axis=1) / (epsilon_E.shape[1] - 1))**0.5
        sigma_T = (np.sum(epsilon_T**2, axis=1) / (epsilon_T.shape[1] - 1))**0.5
    
    sigma_A2 = sigma_T**2 - sigma_E**2
    sigma_A2[sigma_A2 < 0] = 0
    sigma_A = sigma_A2**0.5
    
    return y_tile_cvg, sigma_T, sigma_E, sigma_A

def error_md_std(df, x_col, y_col, x_range, error_bins, sym=True,
                 minsamples=30, scale='linear'):
    """
    Calculate error bars
    """
    if scale=='linear':
        x_bin_edge = np.linspace(x_range[0],x_range[1],error_bins+1)
    elif scale=='log':
        x_bin_edge = np.logspace(np.log10(x_range[0]),
                                 np.log10(x_range[1]),
                                 error_bins+1)
    ymd = []
    yerr = []
    
    x_s_list = x_bin_edge[range(0,error_bins,1)]
    x_e_list = x_bin_edge[range(1,error_bins+1,1)]
    if error_bins % 2 == 0:  # even bins
        x_s_list = np.append(x_s_list, x_e_list[-1])
        x_e_list = np.append(x_e_list, x_bin_edge[-1])
    x_bin = (x_s_list + x_e_list)/2  # bin center
    
    for x_s, x_e in zip(x_s_list, x_e_list):
        samples_i = df.loc[(df[x_col] >= x_s) & (df[x_col] < x_e),
                           y_col]
        
        if samples_i.shape[0] >= minsamples:  # only for sufficient samples
            ymd.append(np.median(samples_i))
            
            if sym == True:
                yerr.append(np.std(samples_i))
            else:  # sym is a percentile tuple for (lower, upper)
                if len(samples_i) == 0:
                    yerr.append([np.nan, np.nan])
                else:
                    yerr.append([np.percentile(samples_i,sym[0]),
                                 np.percentile(samples_i,sym[1])])
        else:
            ymd.append(np.nan)
            yerr.append([np.nan, np.nan])
            
    ymd = np.array(ymd)
    yerr = np.array(yerr).T - ymd
    # yerr = abs(yerr)  # the errorbar in plt requires all values >= 0
        
    return x_bin, ymd, yerr

def get_sigma_df(im, Tmodel='Tαmodels', pick_cvg_b=40):
    """
    Read the bootstrap resultos y_hat_b of a training model of IM.
    Output the training set X and the corresponding sigma_E,A,T in pd.df
    """
    # choose an IM
    # im = 'PGA (g)'
    # Tmodel = 'Tαmodels'
    # num_bstp = 40  # determine by the source data y_hat_b columns
    
    # read from a specific folder for a T-model and im
    y_hat_b_all = pd.read_excel('./bootstrapping/'+ Tmodel +'/' + im +
                                '/bstp_train_yb_15k_500.xlsx',
                                index_col=[0])

    # sigma of Tα-models
    # pick_cvg_b = 40
    y_hat_b_im = y_hat_b_all.\
        iloc[:, y_hat_b_all.columns.get_loc(im) + 1:].values
    y_true = y_hat_b_all[im].values
    y_tile_cvg, sigma_T, sigma_E, sigma_A = \
        func_sigma_E(y_true, y_hat_b_im, pick_cvg_b)
    
    # Dataframe preparation
    # X, y_true and y_hat (i.e. y_pred on train) from the source data
    if Tmodel=='Tαmodels':
        insert_col = 'train_pred_'+im
    else:
        insert_col = im
    
    df = y_hat_b_all.iloc[:, :y_hat_b_all.columns.
                          get_loc(insert_col)+1]
    # y_unbias (i.e. avg. of y_hat_b), sigma_E,A,T
    df.insert(df.columns.get_loc(insert_col)+1,
              "y_unbias", y_tile_cvg)
    df.insert(df.columns.get_loc('y_unbias')+1, "sigma_E", sigma_E)
    df.insert(df.columns.get_loc('sigma_E')+1, "sigma_A", sigma_A)
    df.insert(df.columns.get_loc('sigma_A')+1, "sigma_T", sigma_T)
    
    
    # Region as text
    regionNum = sorted(df['Region'].unique())
    regionLabel = ['Other','WUS','Japan','WCN','Italy','Turkey','Taiwan']
    
    for i, num in enumerate(regionNum):
        df.loc[df['Region']==num,'Region'] = regionLabel[i]
    
    df['Region'] = df['Region'].astype('category')
    
    # Simplify the col. name
    df.rename(columns={'Northern CA/Southern CA - H11 Z2.5 (m)': 'Z2.5 (m)',
                       'Rx': 'Rx (km)',
                       'Ry 2': 'Ry (km)',
                       'Earthquake Magnitude': 'M',
                       'Rake Angle (deg)': 'Rake (deg)',
                       'Hypocenter Depth (km)': 'Dhyp (km)',
                       'Fault Rupture Width (km)': 'W (km)',
                       },
              inplace=True)
    
    # Reorder
    if Tmodel=='Tαmodels': 
        order = ['M', 'Dip (deg)', 'Rake (deg)',
                 'Dhyp (km)', 'Ztor (km)', 'W (km)',
                 'Rjb (km)', 'Rrup (km)', 'Rx (km)', 'Ry (km)',
                 'Vs30 (m/s)', 'Z2.5 (m)', 'Region',
                 im, 'train_pred_'+im, 'y_unbias',
                 'sigma_E', 'sigma_A', 'sigma_T']
    else:
        order = ['M', 'Dip (deg)', 'Rake (deg)',
                 'Dhyp (km)', 'Ztor (km)', 'W (km)',
                 'Rjb (km)', 'Rrup (km)', 'Rx (km)', 'Ry (km)',
                 'Vs30 (m/s)', 'Z2.5 (m)', 'Region',
                 im, 'y_unbias',
                 'sigma_E', 'sigma_A', 'sigma_T']
    df = df[order]
    
    return df