# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:46:11 2023

@author: Vincent NT, ylpxdy@live.com

Functions for ML-GMMs UQ
"""

import numpy as np

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

def error_md_std(df, x_col, y_col, x_range, error_bins, sym=True):
    """
    Calculate error bars
    """
    x_bin_edge = np.linspace(x_range[0],x_range[1],error_bins+1)
    ymd = []
    yerr = []
    
    x_s_list = x_bin_edge[range(0,error_bins,2)]
    x_e_list = x_bin_edge[range(1,error_bins+1,2)]
    if error_bins % 2 == 0:  # even bins
        x_s_list = np.append(x_s_list, x_e_list[-1])
        x_e_list = np.append(x_e_list, x_bin_edge[-1])
    x_bin = (x_s_list + x_e_list)/2  # bin center
    
    for x_s, x_e in zip(x_s_list, x_e_list):
        samples_i = df.loc[(df[x_col] >= x_s) & (df[x_col] < x_e),
                           y_col]
        
        ymd.append(np.median(samples_i))
        
        if sym == True:
            yerr.append(np.std(samples_i))
        else:
            if len(samples_i) == 0:
                yerr.append([np.nan, np.nan])
            else:
                yerr.append([np.percentile(samples_i,14),
                             np.percentile(samples_i,86)])
            
    
    return x_bin, np.array(ymd), np.array(yerr).T