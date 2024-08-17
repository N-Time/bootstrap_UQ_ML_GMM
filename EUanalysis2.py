# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:14:36 2023

@author: Vincent NT, ylpxdy@live.com

Perform parametric UQ in the paper,
Wen T, He J, Jiang L, Du Y, Jiang L. A simple and flexible bootstrap-based framework to quantify epistemic uncertainty of ground motion models by light gradient boosting machine. Applied Soft Computing 2023:111195. https://doi.org/10.1016/j.asoc.2023.111195.
"""


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlgmmuq import func_sigma_E
from mlgmmuq import error_md_std

custom_params = {"axes.spines.right": True, "axes.spines.top": True,
                 "text.usetex": False}
sns.set_theme(style="ticks", rc=custom_params)

# %% Compare sigma_E_i distribtuions of Tα-models and Tβ-models
# Input y_hat_b data for training set

num_bstp = 40

im_list = ['PGA (g)', 'T0.010S', 'T0.020S', 'T0.050S',
           'T0.100S', 'T0.200S', 'T0.500S', 
           'T1.000S', 'T2.000S', 'T5.000S', 'T10.000S']
period_list = [0.0, 0.01, 0.02, 0.05,
               0.1, 0.2, 0.5,
               1.0, 2.0, 5.0, 10.0]

gamma_median_list = []
gamma_std_list = []

for im in im_list:
    # im = 'PGA (g)'  #####
    
    y_hat_b_all_alpha = pd.read_excel('./bootstrapping/Tαmodels/' + im +
                               '/bstp_train_yb_15k_500.xlsx',
                                index_col=[0])
    
    y_hat_b_all_beta = pd.read_excel('./bootstrapping/Tβmodels/' + im +
                               '/bstp_train_yb_15k_500.xlsx',
                                index_col=[0])
    
    
    # estimate y and sigma_E
    pick_cvg_b = 40
    
    y_hat_b_im_alpha = y_hat_b_all_alpha.\
        iloc[:, y_hat_b_all_alpha.columns.get_loc(im) + 1:].values
    y_hat_b_im_beta = y_hat_b_all_beta.\
        iloc[:, y_hat_b_all_beta.columns.get_loc(im) + 1:].values
    
    y_tile_cvg_alpha, sigma_E_alpha = func_sigma_E(y_hat_b_im_alpha, pick_cvg_b)
    y_tile_cvg_beta, sigma_E_beta = func_sigma_E(y_hat_b_im_beta, pick_cvg_b)
    
    gamma_E = sigma_E_alpha / sigma_E_beta - 1
    
    
    # plot to compare the sigma_E of Tα- and Tβ- models
    gamma_E_m = np.median(gamma_E)
    gamma_E_s = np.std(gamma_E)
    gamma_median_list.append(gamma_E_m)
    gamma_std_list.append(gamma_E_s)
    
    # fig, ax = plt.subplots()
    # ax.hist(gamma_E, bins=100, linewidth=0.5, edgecolor="white")
    # ax.axvline(x=gamma_E_m, c='r' ,ls="--", lw=3)
    
    # plt.title(im+': Median and std. of gamma_E = %.3f, %.3f' 
    #           %(gamma_E_m,gamma_E_s), 
    #           fontsize=20)
    # plt.xlabel("gamma_E", fontsize=18)
    # plt.ylabel("Counts", fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)

fig, ax = plt.subplots()
ax.plot(period_list, gamma_median_list, lw=2)
ax.plot(period_list, gamma_std_list, lw=2)

plt.title('Median and std. of gamma_E with IMs', 
          fontsize=20)
plt.xlabel("Period (sec)", fontsize=18)
plt.ylabel("gamma median or std.", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# %% EU analysis of Tα-models
"""2
Assuming independence:
    v = var(delta_y_bji, delta_y_b'j'i'), b = B-model, j = X factor, i = IM
    1) generally, inter-B-model: v = 0, b' != b
    2) inter-factor and unbanlanced data: v != 0, j' != j, hetero- over factor
    3) inter-IM: v != 0, i' != i, hetero- over IM
"""
# choose an IM
im = 'PGA (g)'
Tmodel = 'Tαmodels'
num_bstp = 40

y_hat_b_all = pd.read_excel('./bootstrapping/'+ Tmodel +'/' + im +
                            '/bstp_train_yb_15k_500.xlsx',
                            index_col=[0])
# y_hat_b_all = pd.read_excel('./bootstrapping/' + im + '_B40_RS300' +
#                            '/bstp_train_yb_15k_500.xlsx',
#                             index_col=[0])

# sigma of Tα-models
pick_cvg_b = 40
y_hat_b_im = y_hat_b_all.\
    iloc[:, y_hat_b_all.columns.get_loc(im) + 1:].values
y_true = y_hat_b_all[im].values
y_tile_cvg, sigma_T, sigma_E, sigma_A = \
    func_sigma_E(y_true, y_hat_b_im, pick_cvg_b)

# Dataframe preparation
df = y_hat_b_all.iloc[:, :y_hat_b_all.columns.get_loc('Region')+1]
df.insert(df.columns.get_loc('Region')+1, "y_hat", y_tile_cvg)
df.insert(df.columns.get_loc('y_hat')+1, "sigma_E", sigma_E)

# Region as text
regionNum = sorted(df['Region'].unique())
regionLabel = ['Other','WUS','Japan','WCN','Italy','Turkey','Taiwan']

for i, num in enumerate(regionNum):
    df.loc[df['Region']==num,'Region'] = regionLabel[i]

df['Region'] = df['Region'].astype('category')

variable_symbol = ['Vs30 (m/s)', 'Z2.5 (m)' ,'Rjb (km)',
                   'Rrup (km)', 'Rx (km)', 'Ry (km)',
                   'M', 'Dip (deg)', 'Rake (deg)',
                   'Dhyp (km)', 'Ztor (km)', 'W (km)']
df.rename(columns={'Northern CA/Southern CA - H11 Z2.5 (m)': 'Z2.5 (m)',
                   'Rx': 'Rx (km)',
                   'Ry 2': 'Ry (km)',
                   'Earthquake Magnitude': 'M',
                   'Rake Angle (deg)': 'Rake (deg)',
                   'Hypocenter Depth (km)': 'Dhyp (km)',
                   'Fault Rupture Width (km)': 'W (km)',
                   },
          inplace=True)

# resort variable list by seismology
variable_list_resort = ['M', 'Dip (deg)', 'Rake (deg)',
                        'Dhyp (km)', 'Ztor (km)', 'W (km)',
                        'Rjb (km)', 'Rrup (km)', 'Rx (km)', 'Ry (km)',
                        'Vs30 (m/s)', 'Z2.5 (m)']

# %%% Homoscedastic of totla, aleatory and epistemic uncertainty
fig, ax = plt.subplots()
ax.hist(sigma_T, bins=100,
         alpha=0.5, linewidth=0.5, edgecolor="white", label="Total")
ax.hist(sigma_A, bins=100,
         alpha=0.5, linewidth=0.5, edgecolor="white", label="Aleatory")
ax.hist(sigma_E, bins=100,
         alpha=0.5, linewidth=0.5, edgecolor="white", label="Epistemic")

sigma_T_hm, sigma_A_hm, sigma_E_hm = \
    np.median(sigma_T), np.median(sigma_A), np.median(sigma_E)
ax.axvline(x=sigma_T_hm, c='b' ,ls="--", lw=3)
ax.axvline(x=sigma_A_hm, c='r' ,ls="--", lw=3)
ax.axvline(x=sigma_E_hm, c='g' ,ls="--", lw=3)
print('sigma_T = %.3f, sigma_A = %.3f, sigma_E = %.3f'
      %(sigma_T_hm, sigma_A_hm, sigma_E_hm))

plt.legend()
plt.title(im+': sigma_T, sigma_A, sigma_E = %.3f, %.3f, %.3f' 
          % (sigma_T_hm, sigma_A_hm, sigma_E_hm),
          fontsize=20)
ax.set_xlim((0, 1.2))
plt.xlabel("Sigma", fontsize=18)
plt.ylabel("Counts", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# %%%% Correlation of sigma_E and X
corr_spearman = []
for x in variable_list_resort:
    df_corr = df[[x, 'sigma_E']]
    df_corr.dropna(axis=0, how='any',
                   subset=[x, 'sigma_E'],
                   inplace=True)
        
    s = stats.spearmanr(df_corr[x].values,
                        df_corr['sigma_E'].values)
    
    corr_spearman.append(s)

corr_spearman = np.array(corr_spearman)

# plot
fig, ax = plt.subplots()
ax.barh(variable_list_resort, corr_spearman[:,0])
ax.axvline(x=-0.3, c='k' ,ls="--", lw=1)
ax.axvline(x=0.3, c='k' ,ls="--", lw=1)

ax.set_xlim((-1.0,1.0))
ax.set_yticks(variable_list_resort)
ax.set_xlabel('Spearman correlation coefficient')
ax.invert_yaxis()  # labels read top-to-bottom

plt.title('The spearman correlation coefficient of Sigma_E and metadata')
plt.savefig('./bootstrapping/Figure_sigma_X_corr_spearman.png',
            dpi=600, format="png")

# %%%% X hist
fig, axs = plt.subplots(3, 4, sharey=True)
for ax, v in zip(axs.flat, variable_list_resort):
    sns.histplot(data=df[v], ax=ax,
                 stat='probability', bins=20, kde=True)
    ax.set_xlabel(v)

plt.suptitle('Metadata distribution')
fig.tight_layout(pad=0.1)

plt.savefig('./bootstrapping/Figure_X_hist.png',
            dpi=600, format="png")

# %%%% sigma_E~X regplot
# scatter and 2-order tendency line
variable_range = [(3.0,8.0),(-180,180),(-180,180),(0.0,25.0),(0.0,18.0),(0.0,45.0),
                  (0,300),(0,300),(-250,250),(-500,500),(0,2000),(0,8000)]

# # scatter with linear model
# fig, axs = plt.subplots(3, 4, sharey=True, constrained_layout=True)
# fit_order = 4
# for ax, v, vr in zip(axs.flat, variable_list_resort, variable_range):
#     sns.regplot(data=df, x=v, y="sigma_E", ax=ax,
#                 order=fit_order, ci=None,
#                 scatter_kws=dict(alpha=0.3, edgecolors=None, linewidths=0),
#                 line_kws=dict(ls='--', lw=2.0, color='r'))
    
#     ax.set_xlabel(v)
#     ax.set_ylabel(None)
#     ax.set_xlim(vr)

# for i in [0, 1, 2]:
#     axs[i,0].set_ylabel('sigma_E')


# scatter with error bar
fig, axs = plt.subplots(3, 4, sharey=True, constrained_layout=True)
error_bins = 20
error_low = []  # to store 5% lower
error_16 = []
error_84 = []

for ax, v, vr in zip(axs.flat, variable_list_resort, variable_range):
    df_v = df[[v, 'sigma_E']]
    df_v.dropna(axis=0,how='any',
                subset=[v, 'sigma_E'],
                inplace=True)
    
    x_bin, ymd, yerr = error_md_std(df_v, v, 'sigma_E',
                                    vr, error_bins, sym=False)
    
    ax.scatter(df_v[v], df_v['sigma_E'],
               alpha=0.3, edgecolors=None, s=10)
    ax.errorbar(x_bin, ymd, yerr=yerr,
                fmt='o', ms=5, color='orange',
                ecolor='orange', elinewidth=2, capsize=6)
    
    ax.set_title('Overall median = %.3f' %(np.nanmedian(ymd)))
    ax.set_xlabel(v)
    ax.set_ylabel(None)
    ax.set_xlim(vr)
    
    # yerr_overall = np.nanmedian(yerr,axis=1)
    # print('Percentile 16 in %s = %.3f' %(v, yerr_overall[0]))
    # print('Percentile 84 in %s = %.3f' %(v, yerr_overall[1]))
    
    error_low.append(np.quantile(df_v['sigma_E'],0.05))
    error_16.append(np.quantile(df_v['sigma_E'],0.16))
    error_84.append(np.quantile(df_v['sigma_E'],0.84))

for i in [0, 1, 2]:
    axs[i,0].set_ylabel('sigma_E')


plt.suptitle('sigma_E with varying metadata')
# fig.tight_layout()

# plt.savefig('./bootstrapping/Figure_X_sigma_E_tendency_errorbar.png',
#             dpi=600, format="png")

# %%% Heteroscedastic
# X = y_hat_b_all.iloc[:, :y_hat_b_all.columns.get_loc('Region')+1].values
# V_E_X = sigma_E[:, np.newaxis].dot(sigma_E[np.newaxis, :])
# V_E_X = np.corrcoef(sigma_E[:, np.newaxis])

# sns.heatmap(V_E_X, annot=False)

# # scatter
# sns.set_theme(style="whitegrid")
# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
# g = sns.relplot(
#     data=df,
#     x="Rrup (km)", y="Earthquake Magnitude",
#     hue="Region", size="sigma_E",
#     palette=cmap, sizes=(10, 200),
# )
# g.set(xscale="log", yscale="log")
# g.ax.xaxis.grid(True, "minor", linewidth=.25)
# g.ax.yaxis.grid(True, "minor", linewidth=.25)
# g.despine(left=True, bottom=True)

# color_list=[]
# for i in range(6):
#     color_list.append((np.random.randint(0,255)/255,
#                        np.random.randint(0,255)/255,
#                        np.random.randint(0,255)/255))
# %%% Region
# plot theme
sns.set_theme(style="white")

# auxiliary X from the original 15k dataset
# import from the top of "testLGBMvsHGBR.py"
df1 = df.join(data15k["EQID"])

# %%%% hist
# fig, axs = plt.subplots()
# region_legend = []
# region_list = df.loc[:,'Region'].unique()
# sigma_E_md_region = []
# region_record_size = []
# region_event_size = []

# cmap = plt.get_cmap('Accent')
# for i, r in enumerate(region_list):
#     hist_data_group = df.loc[df['Region'] == r, 'sigma_E']
#     region_record_size.append(hist_data_group.size)
    
#     region_event_size.append(df1.loc[df1['Region'] == r, 'EQID'].unique().size)

#     axs.hist(hist_data_group, density=True,
#              histtype='stepfilled',
#              bins=20,
#              alpha=0.5, color=cmap(i))
    
#     sigma_E_md = np.median(hist_data_group.values)
#     sigma_E_md_region.append(sigma_E_md)
    
#     axs.axvline(x=sigma_E_md, c=cmap(i) ,ls="--", lw=3)
#     axs.set_xlim((0, 1.0))
    
#     print('The median sigma_E in %s = %.3f' %(r, sigma_E_md))
#     region_legend.append(r + '=' + str(round(sigma_E_md, 3)))

# plt.legend(region_legend, fontsize=18, title='Median sigma_E', title_fontsize=18)
# plt.xlabel("Sigma_E",fontsize=20)
# plt.ylabel("Norm. Counts",fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)

# violin plot
# sns.violinplot(data=df, x="sigma_E", y="Region")

# hist for each region
region_record_size = []
region_event_size = []
sigma_E_md_region = []
fig, axs = plt.subplots(2, 4, sharey=True, sharex=False)
for ax, r in zip(axs.flat, regionLabel):
    hist_data_group = df.loc[df['Region'] == r, 'sigma_E']
    sigma_E_md = np.median(hist_data_group.values)
    
    ax.set_title('%s: median sigma_E= %.3f' %(r,sigma_E_md))   
    
    sns.histplot(data=hist_data_group, ax=ax,
                 stat='probability', bins=20, kde=True)
    ax.axvline(x=sigma_E_md, c='k' ,ls="--", lw=2)
    
    ax.set_xlim((0.0,1.0))
    
    sigma_E_md_region.append(sigma_E_md)
    region_record_size.append(hist_data_group.size)
    region_event_size.append(df1.loc[df1['Region'] == r, 'EQID'].unique().size)

plt.suptitle('Regional sigma_E distribution')
axs[1,3].remove()
fig.tight_layout(pad=0.01)


# %%%% median sigma_E vs. regional dataset size
# y axis left = dataset size; y axis right = sigme_E

# Regional record size vs. sigma_E
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(regionLabel, region_record_size)
ax2.plot(regionLabel, sigma_E_md_region,
         color='r', linestyle='-', marker='x', lw=2)

ax1.set(yscale='log')

ax1.set_xlabel('Region')
ax1.set_ylabel('Records size', color = 'b')
ax2.set_ylabel('Regional median sigma_E', color = 'r')
ax2.set_ylim((0, 0.25))

plt.title(im+': Regional median sigma_E with record size')

# Regional event size vs. sigma_E
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(regionLabel,
        np.array(region_record_size) / np.array(region_event_size))
ax2.plot(regionLabel, sigma_E_md_region,
         color='r', linestyle='-', marker='x', lw=2)

ax1.set(yscale='log')

ax1.set_xlabel('Region')
ax1.set_ylabel('Record size per event', color = 'b')
ax2.set_ylabel('Regional median sigma_E', color = 'r')
ax2.set_ylim((0, 0.25))

plt.title(im+': Regional median sigma_E with record size per event')

# %%% Metadata
# df["log sigma_E"] = df["sigma_E"].apply(np.log)
# Tendency
# variable_list = df.columns.values[:-3]  # all predictiors

# cmap = sns.color_palette(palette="flare", n_colors=7)

# for v in variable_list:
# # v = variable_list[0]
#     g = sns.lmplot(data=df, x=v, y="sigma_E", hue="Region",
#                    palette=cmap,
#                    order=2,
#                    scatter_kws=dict(alpha=0.5, edgecolors=None, linewidths=0),
#                    legend_out=False,
#                    )
    
#     if v in ['Rjb (km)', 'Rrup (km)']:
#         g.set(xscale="log", yscale="linear")
#         g.set(xlabel='log '+v, ylabel='sigma_E')
#     else:
#         g.set(xscale="linear", yscale="linear")
#         g.set(xlabel=v, ylabel='sigma_E')
    
#     # g.set(title=v)

# scatter and 2-order tendency line
variable_range = [(0,2000),(0,8000),(0,300),(0,300),(-250,250),(-500,500),
                  (3.0,8.0),(-180,180),(-180,180),(0.0,25.0),(0.0,18.0),(0.0,45.0)]
for v, vr in zip(variable_symbol, variable_range):
    if v != 'Ztor (km)':
        # 'Ztor' matrix is not full-rank!
        grid = sns.FacetGrid(df, col="Region", hue="Region", palette="tab20c",
                              col_wrap=7, height=3.5, aspect=0.65,
                              sharex=False, sharey=True,
                              xlim=vr, ylim=(0.0,1.0))
        
        # Draw a line plot to show the trajectory of each random walk
        grid.map(sns.regplot, v, "sigma_E", order=1, ci=None,
                 scatter_kws=dict(alpha=0.3, edgecolors=None, linewidths=0),
                 )
        
        grid.fig.tight_layout(w_pad=0.1)
        
        v_name = v.split(' (')[0]  # the / should not be in file name
        plt.savefig('./bootstrapping/Figure_sigma_reginal_X_trend_linear_'+v_name+'.png',
                    dpi=600, format="png")

# %%% hist of X
for v in variable_symbol:
    fig, ax = plt.subplots()
    # ax.hist(df[v], bins=20, edgecolor='white')
    
    # if v in ['Rjb (km)', 'Rrup (km)']:
    #     ax.set(xscale="log", yscale="linear")
    #     ax.set(xlabel='log '+v, ylabel='sigma_E')
    # else:
    #     ax.set(xscale="linear", yscale="linear")
    #     ax.set(xlabel=v, ylabel='sigma_E')

    sns.histplot(data=df[[v,'Region']], x=v, hue="Region",
                 bins=20, stat='probability', log_scale=False)

# %%% Correlation coefficient
# plot
fig, axs = plt.subplots(2, 4, sharey=True)
for ax, r in zip(axs.flat, regionLabel):
    ax.set_title('Region: %s' %(r))
    region_meta_sigma = df[df['Region'] == r]  # select a specific region
    
    corr_spearman = []
    for x in variable_list_resort:
        df_corr = region_meta_sigma[[x, 'sigma_E']]
        df_corr.dropna(axis=0,how='any',
                       subset=[x, 'sigma_E'],
                       inplace=True)
            
        s = stats.spearmanr(df_corr[x].values,
                                   df_corr['sigma_E'].values)
        
        corr_spearman.append(s)
    
    corr_spearman = np.array(corr_spearman)
    
    ax.barh(variable_list_resort, corr_spearman[:,0])
    ax.axvline(x=-0.3, c='k' ,ls="--", lw=1)
    ax.axvline(x=0.3, c='k' ,ls="--", lw=1)
    
    ax.set_xlim((-1.0,1.0))
    ax.set_yticks(variable_list_resort)
    ax.invert_yaxis()  # labels read top-to-bottom
    
axs[1,3].remove()
# plt.tight_layout()
plt.suptitle('The spearman correlation coefficient of Sigma_E and metadata in different regions')

# %%%% regional distribution by pairplot

variable_range = [(0,2000),(0,8000),(0,300),(0,300),(-250,250),(-500,500),
                  (3.0,8.0),(-180,180),(-180,180),(0.0,25.0),(0.0,18.0),(0.0,45.0)]

for v, vr in zip(variable_symbol, variable_range):
    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="Region", hue="Region", palette="tab20c",
                          col_wrap=7, height=3.5, aspect=0.65,
                          sharex=False, sharey=True,
                          xlim=vr)
    
    # Draw a line plot to show the trajectory of each random walk
    grid.map(sns.histplot, v, stat='probability')
    
    grid.fig.tight_layout(w_pad=0.1)
    
    v_name = v.split(' (')[0]  # the / should not be in file name
    plt.savefig('./bootstrapping/Figure_sigma_reginal_X_hist_'+v_name+'.png',
                dpi=600, format="png")

