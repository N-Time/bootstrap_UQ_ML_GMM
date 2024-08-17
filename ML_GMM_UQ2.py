# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:40:05 2023

@author: Vincent NT, ylpxdy@live.com

Output the results in the paper,
Wen T, He J, Jiang L, Du Y, Jiang L. A simple and flexible bootstrap-based framework to quantify epistemic uncertainty of ground motion models by light gradient boosting machine. Applied Soft Computing 2023:111195. https://doi.org/10.1016/j.asoc.2023.111195.
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn import preprocessing

from mlgmmuq import func_sigma_E
from mlgmmuq import error_md_std
from mlgmmuq import get_sigma_df
from mlgmmuq import sigma_E_model_AY14
from mlgmmuq import get_event_list

custom_params = {"axes.spines.right": True, "axes.spines.top": True,
                 "text.usetex": False,
                 "axes.grid": True, "axes.grid.which": 'both',
                 "grid.alpha": 0.5}
sns.set_theme(style="ticks", font_scale=1.5,
              rc=custom_params)
plt.rcParams['font.sans-serif'] = 'times new roman'


# %% Input data

# # load a df of an IM for test
# im = 'PGA (g)'  # for a test
# Tmodel = 'Tαmodels'
# pick_cvg_b = 40
# df_test = get_sigma_df(im=im, Tmodel=Tmodel, pick_cvg_b=pick_cvg_b)

# im list
im_list = ['PGA (g)', 'PGV (cm sec^-1)',
           'T0.010S', 'T0.020S', 'T0.030S', 'T0.050S', 'T0.075S',
           'T0.100S', 'T0.150S', 'T0.200S', 'T0.300S', 
           'T0.400S', 'T0.500S', 'T0.750S',
           'T1.000S', 'T1.500S', 'T2.000S', 'T3.000S',
           'T4.000S', 'T5.000S', 'T7.500S', 'T10.000S']
im_values = [0.0, 0.0, 
             0.01, 0.02, 0.03, 0.05, 0.075,
             0.1, 0.15, 0.2, 0.3, 
             0.4, 0.5, 0.75,
             1.0, 1.5, 2.0, 3.0,
             4.0, 5.0, 7.5, 10.0]

im_list0 = ['PGA (g)',
           'T0.010S', 'T0.020S', 'T0.030S', 'T0.050S', 'T0.075S',
           'T0.100S', 'T0.150S', 'T0.200S', 'T0.300S', 
           'T0.400S', 'T0.500S', 'T0.750S',
           'T1.000S', 'T1.500S', 'T2.000S', 'T3.000S',
           'T4.000S', 'T5.000S', 'T7.500S', 'T10.000S']
im_values0 = [0.001, 
            0.01, 0.02, 0.03, 0.05, 0.075,
            0.1, 0.15, 0.2, 0.3, 
            0.4, 0.5, 0.75,
            1.0, 1.5, 2.0, 3.0,
            4.0, 5.0, 7.5, 10.0]

features = ['M', 'Dip (deg)', 'Rake (deg)',
            'Dhyp (km)', 'Ztor (km)', 'W (km)',
            'Rjb (km)', 'Rrup (km)', 'Rx (km)', 'Ry (km)',
            'Vs30 (m/s)', 'Z2.5 (m)', 'Region']

# %% Data set
"""
'data15k' get from 'LGBMbootstrap.py'
"""
# %%% 3. M-R plot
"""
Show M-R scatter with regional label
"""
cmap = sns.color_palette(n_colors=data15k['Region'].unique().size)
g = sns.relplot(
    data=data15k,
    x="Rrup (km)", y="Earthquake Magnitude",hue="Region",
    palette=cmap,
    )

g.set(xscale="log", yscale="linear")
g.despine(left=False, bottom=False, right=False, top=False)
g.set_ylabels('M')
g.set_xlabels('Rrup (km)')
# g.set_xlim((0.1,500))
# g.set_ylim((3.0,8.0))
# fig.get_size_inches()

# %%% 3.+ Data distribution: M-R plot for each IM
LUF = 'Lowest Usable Freq - Ave. Component (Hz)'

fig, axs = plt.subplots(5,4,sharex=True,sharey=True,constrained_layout=True)

for ax, im, im_val in zip(axs.flat, im_list, im_values):

    if im_val != 0:
        LUF_flag = 1/im_val > data15k[LUF]
    else:
        LUF_flag = np.ones((len(data15k),), dtype=bool)
    
    data = data15k[LUF_flag]
    
    ax.hist2d(data["Rrup (km)"],data["Earthquake Magnitude"],
              bins=50,cmap=cm.coolwarm)
    # ax.scatter(data[]data[im])
    
    ax.set(xscale="log", yscale="linear")
    ax.set_title(im)

# %%% 3.+ Data local trend: bin_M-R plot with linear trend for each IM
"""
Check the simple linear attenuation relationship of log10(IM) ~ log10(R) for M bins
To verify the sudden decrease of attenuation at ~60~70km by LGBM-based GMM,
for T > ~1.0s
"""
im = "T3.000S"  # T0.300S
im_val = 3.0  # 0.3

LUF = 'Lowest Usable Freq - Ave. Component (Hz)'
if im_val != 0:
    LUF_flag = 1/im_val > data15k[LUF]
else:
    LUF_flag = np.ones((len(data15k),), dtype=bool)

data = data15k[LUF_flag]
data = data[["Rrup (km)", "Earthquake Magnitude", im]]
bin_delta = 0.5
bins = np.arange(3.0,8.0,bin_delta)

for b in bins:
    # left edge as bin name
    bin_flag = (data["Earthquake Magnitude"] >= b) & (
                data["Earthquake Magnitude"] < (b + bin_delta))
    data.loc[bin_flag, "bin_flag"] = b

data["Rrup (km)"] = np.log10(data["Rrup (km)"])
data[im] = np.log10(data[im])
sns.lmplot(data=data, order=5,
           x="Rrup (km)", y=im, hue="bin_flag")

# %%% 3. LUF: Records, Events ~ Period
"""
Show Records, Events ~ Period
"""

# get IM values and IM names for 0.01s~10.0s
im_col_names = data15k.iloc[:, data15k.columns.get_loc('T0.010S'):
                            data15k.columns.get_loc('T10.000S')+1].columns.values
im_col_values = []
for i in im_col_names:
    im_col_values.append(float(i[1:-1]))

# count records and events by LUF
record_counts = []
event_counts = []
for n, v in zip(im_col_names, im_col_values):
    temp = data15k.loc[data15k['Lowest Usable Freq - Ave. Component (Hz)']
                         < 1/v, ['Record Sequence Number', 'EQID']]
    record_counts.append(temp.shape[0])
    
    event_counts.append(temp['EQID'].unique().shape[0])

# plot
fig, ax = plt.subplots(figsize=(6.4, 4.8))

ax.plot(im_col_values, record_counts,
        '-', lw=2, color='C0', label='records')
ax.set_xscale('log')
ax.set_ylabel('Record counts')
ax.set_xlim((0.01,10.0))

ax2 = ax.twinx()
ax2.plot(im_col_values, event_counts,
         '-', lw=2, color='C2', label='events')
ax2.set_ylabel('Event counts')

# %% 4.1 Central model

# %%% 4.1.1 Optimized hyperparameters of max_leaf_nodes & min_samples_leaf
"""
Show the contour of the 2 hyperparamters and its opt point.
"""
Tmodel = 'Tαmodels_stratify'
im_central_list = [
    'PGA (g)', 'PGV (cm sec^-1)','T0.010S','T0.030S',
    # 'T0.050S','T0.100S', 'T0.300S', 'T0.500S',
    # 'T1.000S', 'T3.000S', 'T5.000S','T10.000S',
    ]

fig, axs = plt.subplots(1, 4, sharex=True, sharey=True,
                        constrained_layout=True, figsize=(13.89,3.08))   # figsize=(15.8,3.9)
for ax, im_central in zip(axs.reshape(-1), im_central_list):
    # load data
    # im_central = 'T3.000S'
    df_op = pd.read_excel('./bootstrapping/'+ Tmodel +'/' + im_central +
                          '/'+im_central+'LGBMGrid_15k_500.xlsx',
                                index_col=[0])
    
    # cv grid
    x = df_op['param_max_leaf_nodes'].unique()  # LGBM: param_min_data_in_leaf
    y = df_op['param_min_samples_leaf'].unique()  # LGBM: param_num_leaves
    x, y = np.meshgrid(x, y)
    
    z = df_op['mean_test_score'].values.reshape(x.shape[1], -1).T
    
    # best point
    x_op, y_op = df_op.loc[df_op['rank_test_score'] == 1,
                           ['param_max_leaf_nodes',
                            'param_min_samples_leaf']].values[0,:]
    # y_op = df_op.loc[df_op['rank_test_score'] == 1, 'param_min_samples_leaf'].unique()
    z_op = df_op.loc[df_op['rank_test_score'] == 1, 'mean_test_score'].values
    
    # plot
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    ctf = ax.contourf(x, y, z, cmap=cm.PuBu_r)
    
    pt = ax.scatter(x_op, y_op, marker='*', c='k', s=100)
    
    ax.set_title(im_central)
    
    print('Optimized params. of %s = %s, %s' %(im_central, x_op, y_op))
    
fig.colorbar(ctf, aspect=40)

# %%% 4.1.2 Treat overfitting
"""
Show the n_iter_ ~ test and train score each stage.
"""
Tmodel = 'Tαmodels_stratify'
# Show IM
im_central_list = ['PGA (g)', 'PGV (cm sec^-1)',
           'T0.010S','T0.030S', 'T0.050S',
           'T0.100S', 'T0.300S', 'T0.500S',
           'T1.000S', 'T3.000S', 'T5.000S','T10.000S']  # for test ['PGA (g)','T3.000S'] 

# Read IM es_test_score and es_test_score, and best es
df_es_test = pd.read_excel('./bootstrapping/'+ Tmodel +
                      '/score_test_max_iter_15k_500.xlsx',
                      index_col=[0])
df_es_test.columns = im_list
df_es_train = pd.read_excel('./bootstrapping/'+ Tmodel +
                      '/score_train_max_iter_15k_500.xlsx',
                      index_col=[0])
df_es_train.columns = im_list
df_hp_opt = pd.read_excel('./bootstrapping/'+ Tmodel +
                          '/summaryHGBR_15k_500.xlsx',
                          index_col=[0])

# Plot 3x4 IMs n_iter_ ~ stage_score with indicating the 2 gaps
fig, axs = plt.subplots(3, 4, sharex=True, sharey=True,
                        constrained_layout=True, figsize=(17.5,9.66))
x_iter = np.arange(1,300+1)
for ax, im in zip(axs.flat,im_central_list):
    k = im_list.index(im)
    
    # gap at best es point
    best_n_iter = df_hp_opt.loc[df_hp_opt['IM']==im, 'n_iter'].values[0]
    gap_best = df_es_test.iloc[best_n_iter-1,k] - \
                df_es_train.iloc[best_n_iter-1,k]
    
    # gap at the max_iter
    stage_test_score = df_es_test.iloc[:,k].values
    stage_test_score = stage_test_score[~np.isnan(stage_test_score)]
    stage_train_score = df_es_train.iloc[:,k].values
    stage_train_score = stage_train_score[~np.isnan(stage_train_score)]
    gap_max = stage_test_score[-1] - stage_train_score[-1]
    
    print(im+': es at %.f, gap = %.3f %.3f' %(best_n_iter,gap_best,gap_max))
    
    ax.plot(x_iter, df_es_train.iloc[:,k], lw=1.5,
            label='Train score', zorder=2)
    ax.plot(x_iter, df_es_test.iloc[:,k], lw=1.5,
            label='Test score', zorder=2)
    ax.axvline(x=best_n_iter, c='k' ,ls="--", lw=1.5,
               label='Early stopping', zorder=3)
    ax.axhline(y=stage_test_score[-1], c='gray' ,ls="--", lw=1.5,
               label='max_test', zorder=1)
    ax.axhline(y=stage_train_score[-1], c='gray' ,ls="--", lw=1.5,
               label='max_train', zorder=1)
    
    ax.set_title(im+': es=%.f, gap=%.3f,%.3f' %(best_n_iter,gap_best,gap_max))
    ax.set_xlim((0,300))
    ax.set_ylim((0,2))

# ax.legend()

# %%% 4.1.2 Accuracy on training and testing sets
"""
In the section
'Output model: given params. and need to find earling stopping',
in 'LGBMbootstrap.py'

Check the assumption of bootstrap on Train0 or Test
Assuming: y_t by directively training on Train0 is less accuracy than
    y_hat by B predictions y_hat_b via bootstrap-based B-models
"""
check_subset = 'test'   # show the result on Train0-train or Test-test
Tmodel = 'Tαmodels_stratify'   # non-opt: insufficiently cv tunning

# im_central_list = ['PGA (g)', 'PGV (cm sec^-1)']   # for a test
im_central_list = im_list.copy()
im_central_list.remove('PGV (cm sec^-1)')  # remove PGV

yt0_R2_list = []
yt0_RMSE_list = []
yt0_MAE_list = []
yub_R2_list = []
yub_RMSE_list = []
yub_MAE_list = []

for ii, im_central in enumerate(im_central_list):
    # Load data of y_true, y_hat_train0, y_hat_b
    # im_central = 'PGA (g)'
    if check_subset == 'train':
        df_y = pd.read_excel('./bootstrapping/'+ Tmodel +'/' + im_central +
                              '/bstp_train_yb_15k_500.xlsx',
                                    index_col=[0])
    elif check_subset == 'test':
        df_y = pd.read_excel('./bootstrapping/'+ Tmodel +'/' + im_central +
                              '/'+im_central+'bstp_test_15k_500.xlsx',
                                    index_col=[0])
    y_true = df_y[im_central].values
    y_hat_train0 = df_y[check_subset+'_pred_'+im_central].values
    y_hat_b = df_y.iloc[:,df_y.columns.\
                        get_loc('b_0_'+check_subset+'_pred_'+im_central):].values
    
    # Calculate y_unbias by B y_hat_b s
    y_unbias = np.mean(y_hat_b,axis=1)
    
    # Calculate evl. (R2, RMSE, MAE) of y_hat_train0 and y_unbias
    yt0_R2, yt0_RMSE, yt0_MAE = r2_score(y_true,y_hat_train0),\
                                mean_squared_error(y_true,y_hat_train0)**0.5,\
                                mean_absolute_error(y_true,y_hat_train0)
    yub_R2, yub_RMSE, yub_MAE = r2_score(y_true,y_unbias),\
                                mean_squared_error(y_true,y_unbias)**0.5,\
                                mean_absolute_error(y_true,y_unbias)
    y_true_b_mat = np.tile(y_true.reshape((-1,1)), (1,y_hat_b.shape[1]))
    yb_R2, yb_RMSE, yb_MAE = r2_score(y_true_b_mat,y_hat_b,multioutput='raw_values'),\
                                mean_squared_error(y_true_b_mat,y_hat_b,multioutput='raw_values')**0.5,\
                                mean_absolute_error(y_true_b_mat,y_hat_b,multioutput='raw_values')
    
    # Store evl. results for each IM result
    yt0_R2_list.append(yt0_R2)
    yt0_RMSE_list.append(yt0_RMSE)
    yt0_MAE_list.append(yt0_MAE)
    yub_R2_list.append(yub_R2)
    yub_RMSE_list.append(yub_RMSE)
    yub_MAE_list.append(yub_MAE)
    if ii == 0:
        yb_R2_list = yb_R2.reshape((1,-1))
        yb_RMSE_list = yb_RMSE.reshape((1,-1))
        yb_MAE_list = yb_MAE.reshape((1,-1))
    else:
        yb_R2_list = np.concatenate((yb_R2_list, yb_R2.reshape((1,-1))),axis=0)
        yb_RMSE_list = np.concatenate((yb_RMSE_list, yb_RMSE.reshape((1,-1))),axis=0)
        yb_MAE_list = np.concatenate((yb_MAE_list, yb_MAE.reshape((1,-1))),axis=0)
    
    print('Finshed IM: %s' %(im_central))


# Plot: evl. ~ IM, for y_hat_train0, y_unbias and y_hat_b s
fig, axs = plt.subplots(1,3,sharex=True,
                        figsize=(15.8,3.9),constrained_layout=True)
# x_plt = np.arange(len(im_central_list))   # for a test
x_plt = im_values[1:]

evl_list = [[yt0_R2_list, yub_R2_list, yb_R2_list],
            [yt0_RMSE_list, yub_RMSE_list, yb_RMSE_list],
            [yt0_MAE_list, yub_MAE_list, yb_MAE_list]]
color_list = ['C0', 'C1', 'C2']

for ax, evls, c in zip(axs, evl_list, color_list):
    ax.plot(x_plt, evls[0], 'o--', color=c, label='y_hat_train0')
    ax.plot(x_plt, evls[1], 'o-', color=c, label='y_unbias')
    ax.plot(x_plt, evls[2],
             '-', color=c, alpha=0.3, label='y_hat_b')
    
    if c == 'C0':
        ax.set_ylim((0.90, 1.0))
    elif c == 'C1':
        ax.set_ylim((0.3, 0.8))
    elif c == 'C2':
        ax.set_ylim((0.2, 0.6))
        
    ax.set(xscale="log")
    ax.set_xlim((0.01,10))
    # ax.legend()


# %% 4.2. Uncertainty decomposition
"""
Decomposite uncertainty into:
    1) epsilon_T = epsilon_E + epsilon_A;
    2) epsilon_A = epsilon_inter + epsilon_intra
"""

Tmodel = 'Tαmodels_stratify/tol_3E-3_T1_T2' #  'Tαmodels'
stratify_flag = True  # Use stratify in data splitting of training model
im_list_t = ['T1.000S', 'T1.500S', 'T2.000S']   # for test im_list
im_val_t = [1.0, 1.5, 2.0]  # for test im_values

df_epsilon_stats = pd.DataFrame(columns=['IM', 'IM_value',
                                         'epsilon_T_md', 'epsilon_T_std',
                                         'epsilon_E_md', 'epsilon_E_std',
                                         'epsilon_A_md', 'epsilon_A_std',
                                         'sigma_inter_A_md', 'sigma_inter_A_std',
                                         'sigma_intra_A_md', 'sigma_intra_A_std'])
df_epsilon_stats2 = df_epsilon_stats.copy()

for im, im_val in zip(im_list_t, im_val_t):
    
    setFlags = ['train', 'test']
    datafile = ['bstp_train_yb_15k_500.xlsx',
            im+'bstp_test_15k_500.xlsx']

    for filename, setFlag in zip(datafile, setFlags):
        
        df = get_sigma_df(im=im, Tmodel=Tmodel, pick_cvg_b=40, region_text=True,
                          filename=filename, setFlag=setFlag)
        
        # event label
        event_train, event_test = get_event_list(
            data15k, LUF_col_name='Lowest Usable Freq - Ave. Component (Hz)',
            im_val=im_val, random_state=500, test_size=0.2,
            stratify_flag=stratify_flag)
        
        # epsilon components
        df["epsilon_T"] = df[im] - df[setFlag+'_pred_'+im]
        df["epsilon_E"] = df['y_unbias'] - df[setFlag+'_pred_'+im]
        df["epsilon_A"] = df[im] - df['y_unbias']
        
        # Group epsilon A components
        if setFlag == 'train':
            df["event"] = event_train["EQID"]
        elif setFlag == 'test':
            df["event"] = event_test["EQID"]
        df["event_avg_obs"] = df[[im, "event"]].groupby(["event"])[im].transform('mean')
        df["event_avg_est"] = df[["y_unbias", "event"]].groupby(["event"])["y_unbias"].transform('mean')
        # deltaB = 'avg. recorded im of an event' - 'avg. estmated im of an event'
        df["epsilon_inter"] = df["event_avg_obs"] - df["event_avg_est"]
        # deltaW = 'true im of a record' - 'avg. estmated im of an event'
        df["epsilon_intra"] = df["epsilon_A"] - df["epsilon_inter"]
        
        # statistics of epsilon
        epsilon_T_md, epsilon_T_std = stats.norm.fit(df["epsilon_T"].values)
        epsilon_E_md, epsilon_E_std = stats.norm.fit(df["epsilon_E"].values)
        epsilon_A_md, epsilon_A_std = stats.norm.fit(df["epsilon_A"].values)
        
        # for Delta_B and Delta_W
        # drop repeated row of an event
        df_unique_event = df.drop_duplicates(subset=['event'], keep='first')
        sigma_inter_A_md, sigma_inter_A_std = (
            stats.norm.fit(df_unique_event["epsilon_inter"].values))
        sigma_intra_A_md, sigma_intra_A_std = (
            stats.norm.fit(df["epsilon_intra"].values))
        
        epsilon_temp = {'IM': im,
                  'IM_value': im_val,
                  'epsilon_T_md': epsilon_T_md,
                  'epsilon_T_std': epsilon_T_std,
                  'epsilon_E_md': epsilon_E_md,
                  'epsilon_E_std': epsilon_E_std,
                  'epsilon_A_md': epsilon_A_md,
                  'epsilon_A_std': epsilon_A_std,
                  'sigma_inter_A_md': sigma_inter_A_md,
                  'sigma_inter_A_std': sigma_inter_A_std,
                  'sigma_intra_A_md': sigma_intra_A_md,
                  'sigma_intra_A_std': sigma_intra_A_std,
                 }
        
        if filename == 'bstp_train_yb_15k_500.xlsx':
            df_epsilon_stats = df_epsilon_stats.append(epsilon_temp, ignore_index=True)
        elif filename == im+'bstp_test_15k_500.xlsx':
            df_epsilon_stats2 = df_epsilon_stats2.append(epsilon_temp, ignore_index=True)
        
        print('Finished %s set: %s' %(setFlag,im))
        print('==>> epsilon_T_std, epsilon_E_std, epsilon_A_std = %.3f, %.3f, %.3f' %(
                epsilon_T_std, epsilon_E_std, epsilon_A_std))
        print('==>> sigma_inter_std, sigma_intra_std = %.3f, %.3f' %(
                sigma_inter_A_std, sigma_intra_A_std))
        
        df.to_excel('./bootstrapping/'+ Tmodel +'/' + im +
                    '/uq_'+filename)
    
df_epsilon_stats.to_excel('./bootstrapping/'+Tmodel+'/uq_est_train_set.xlsx')
df_epsilon_stats2.to_excel('./bootstrapping/'+Tmodel+'/uq_est_test_set.xlsx')


# %%% 4.2.1. plot inter- (vs. M) and intra- (vs. Rjb) residuals
im = 'PGA (g)'
Tmodel = 'Tαmodels'

df = pd.read_excel('./bootstrapping/'+Tmodel+'/' + im +
                   '/uq_'+'bstp_train_yb_15k_500.xlsx')
df_epsilon_stats = pd.read_excel('./bootstrapping/'+Tmodel
                                 +'/uq_est_train_set.xlsx')

fig, ax = plt.subplots(1,2,sharex=False,sharey=True,  # 2,5 for paper
                        constrained_layout=True)

ax[0].scatter(df["M"], df["epsilon_inter"],
            alpha=0.3, edgecolors=None, s=10)
ax[0].set_title('Overall median, std = %.3f, %.3f' %(
    df_epsilon_stats.loc[df_epsilon_stats['IM']==im,"sigma_inter_A_md"].values,
    df_epsilon_stats.loc[df_epsilon_stats['IM']==im,"sigma_inter_A_std"].values))
ax[0].set_xlabel("M")
ax[0].set_ylabel("Inter Residuals")
# ax[0].set_xlim(vr)
ax[0].set_ylim((-1.5,1.5))

ax[1].scatter(df["Rjb (km)"], df["epsilon_intra"],
            alpha=0.3, edgecolors=None, s=10)
ax[1].set_title('Overall median, std = %.3f, %.3f' %(
    df_epsilon_stats.loc[df_epsilon_stats['IM']==im,"sigma_intra_A_md"].values,
    df_epsilon_stats.loc[df_epsilon_stats['IM']==im,"sigma_intra_A_std"].values))
ax[1].set_xlabel("Rjb (km)")
ax[1].set_ylabel("Intra Residuals")
ax[1].set_xscale('log')
ax[1].set_xlim((1.0, 300))
ax[1].set_ylim((-1.5,1.5))

# %%% 4.2.2.1 Inter-event residuals for various X

im = 'PGA (g)'
Tmodel = 'Tαmodels_overfit'
epsilon_comp = "epsilon_inter"
df_train_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                            '/uq_bstp_train_yb_15k_500.xlsx').iloc[:,1:]
df_test_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                           '/uq_'+im+'bstp_test_15k_500.xlsx').iloc[:,1:]
df_train_coef = pd.read_excel('./bootstrapping/'+Tmodel+'/uq_est_train_set.xlsx')
# only train coef. due to the really similar coef. for test and train.
metadata_list = ['M', 'Dip (deg)', 'Rake (deg)',
                 'Dhyp (km)', 'Ztor (km)', 'W (km)']
metadata_range = [(3.0,8.0),(10.0,90.0),(-180,180),
                  (0.0,20.0),(0.0,18.0),(0.0,45.0)]

fig, axs = plt.subplots(2, 3, sharey=True, constrained_layout=True,
                        figsize=(12.47, 7.23))
error_bins = 10
color_list = ['red','orange','green']

for ax, v, vr in zip(axs.flat, metadata_list, metadata_range):
    # v(i.e. one of metadata) ~ epsilon_intra or epsilon_inter
    if v == "M":
        df_v = pd.concat([df_train_uq.loc[:,["Rake (deg)",v,epsilon_comp]],
                         df_test_uq.loc[:,["Rake (deg)",v,epsilon_comp]]])
    elif v == "Rake (deg)":
        df_v = pd.concat([df_train_uq.loc[:,["M",v,epsilon_comp]],
                          df_test_uq.loc[:,["M",v,epsilon_comp]]])
    else:
        df_v = pd.concat([df_train_uq.loc[:,["M","Rake (deg)",v,epsilon_comp]],
                         df_test_uq.loc[:,["M","Rake (deg)",v,epsilon_comp]]])

    df_v.dropna(axis=0,how='any',
                subset=[v, epsilon_comp],
                inplace=True)
    
    # Log scale for R
    scale = 'linear'
    df_v_hb = df_v
    if (v == 'Rjb (km)') | (v == 'Rrup (km)'):
        ax.set_xscale('log')
        scale = 'log'
        df_v_hb = df_v[df_v[v]>0]
        
    # Error bar values
    x_bin, ymd, yerr = error_md_std(df=df_v[[v,epsilon_comp]],
                                    x_col=v, y_col=epsilon_comp,
                                    x_range=vr, error_bins=error_bins,
                                    sym=True, minsamples=30, scale=scale)
    
    # Plot residual points
    ax.scatter(df_v[v], df_v[epsilon_comp],
               alpha=0.3, edgecolors=None, s=10, zorder=1)
    # ax.hexbin(df_v_hb[v].values, df_v_hb[epsilon_comp].values,
    #           gridsize=100, xscale=scale, cmap='summer')
    
    # Plot errorbars
    # require the abs of error for both lower and higher
    if yerr.ndim > 1:  # for 'sym' = False
        yerr_abs_fc = np.concatenate((np.ones((1,yerr.shape[1]))*(-1),
                                      np.ones((1,yerr.shape[1]))), axis=0)
        yerr = yerr*yerr_abs_fc
    ax.errorbar(x_bin, ymd, yerr=yerr,
                fmt='o', ms=5, color='k',
                ecolor='k', elinewidth=2, capsize=6, zorder=4)
    
    # Plot sigma_E models
    # overall sigma_E by MLE in this paper
    sigma_E_MLE = df_train_coef.loc[df_train_coef['IM']==im,
                                    "sigma_inter_A_std"].values
    ax.hlines([sigma_E_MLE, -sigma_E_MLE], vr[0], vr[1], label='MLE',
              linestyles='--', colors=color_list[0], lw=2.0, zorder=3)
    
    # Plot setting
    # ax.set_title('Overall median = %.3f' %(np.nanmedian(ymd)))
    ax.set_xlabel(v)
    ax.set_ylabel(None)
    ax.set_xlim(vr)
    ax.set_ylim((-1.5,1.5))

# ax.legend()

# %%% 4.2.2.2 Inter-event residuals for various IM

im_list1 = ['PGA (g)', 'PGV (cm sec^-1)', 'T0.030S', 'T0.300S', 
            'T1.000S', 'T2.000S', 'T3.000S', 'T10.000S']
Tmodel = 'Tαmodels_stratify'
epsilon_comp = "epsilon_inter"

df_train_coef = pd.read_excel('./bootstrapping/'+Tmodel+
                              '/uq_est_train_set.xlsx')

v = "M"
vr = (3.0,8.0)

fig, axs = plt.subplots(2, 4, sharey=True, constrained_layout=True,
                        figsize=(15.97,6.44))
error_bins = 8
color_list = ['red','orange','green']

for ax, im in zip(axs.flat, im_list1):
    df_train_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                            '/uq_bstp_train_yb_15k_500.xlsx').iloc[:,1:]
    df_test_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                               '/uq_'+im+'bstp_test_15k_500.xlsx').iloc[:,1:]
    # drop repeated row of an event
    df_train_uq = df_train_uq.drop_duplicates(subset=['event'], keep='first')
    df_test_uq = df_test_uq.drop_duplicates(subset=['event'], keep='first')
    
    # v(i.e. one of metadata) ~ epsilon_intra or epsilon_inter
    df_v = pd.concat([df_train_uq.loc[:,["Rake (deg)","M",epsilon_comp]],
                       df_test_uq.loc[:,["Rake (deg)","M",epsilon_comp]],
                      ])

    df_v.dropna(axis=0,how='any',
                subset=[v, epsilon_comp],
                inplace=True)
    
    # Log scale for R
    scale = 'linear'
    df_v_hb = df_v
    if (v == 'Rjb (km)') | (v == 'Rrup (km)'):
        ax.set_xscale('log')
        scale = 'log'
        df_v_hb = df_v[df_v[v]>0]
        
    # Error bar values
    x_bin, ymd, yerr = error_md_std(df=df_v[[v,epsilon_comp]],
                                    x_col=v, y_col=epsilon_comp,
                                    x_range=vr, error_bins=error_bins,
                                    sym=True, minsamples=20, scale=scale)
    
    # Plot residual points
    ax.scatter(df_v[v], df_v[epsilon_comp],
               alpha=0.3, edgecolors=None, s=10, zorder=1)
    # ax.hexbin(df_v_hb[v].values, df_v_hb[epsilon_comp].values,
    #           gridsize=100, xscale=scale, cmap='summer')
    
    # Plot errorbars
    # require the abs of error for both lower and higher
    if yerr.ndim > 1:  # for 'sym' = False
        yerr_abs_fc = np.concatenate((np.ones((1,yerr.shape[1]))*(-1),
                                      np.ones((1,yerr.shape[1]))), axis=0)
        yerr = yerr*yerr_abs_fc
    # ax.errorbar(x_bin, ymd, yerr=yerr,
    #             fmt='o', ms=5, color='k',
    #             ecolor='k', elinewidth=2, capsize=6, zorder=4)
    ax.scatter(x_bin,ymd,c='k',s=30,edgecolors='k',marker='d',
               zorder=4,label='bin mean')
    ax.scatter(x_bin,ymd+yerr,c='k',s=30,edgecolors='k',marker='P',
               zorder=4,label='bin sigma')
    ax.scatter(x_bin,ymd-yerr,c='k',s=30,edgecolors='k',marker='P',
               zorder=4,label='bin sigma')

    # Plot sigma_E models
    # overall sigma_E by MLE in this paper
    sigma_E_MLE = df_train_coef.loc[df_train_coef['IM']==im,
                                    "sigma_inter_A_std"].values
    ax.hlines([sigma_E_MLE, -sigma_E_MLE], vr[0], vr[1], label='MLE',
              linestyles='--', colors=color_list[0], lw=2.0, zorder=3)
    
    # Plot setting
    # ax.set_title('Overall median = %.3f' %(np.nanmedian(ymd)))
    ax.set_xlabel(v)
    ax.set_ylabel(None)
    ax.set_xlim(vr)
    ax.set_ylim((-2.0,2.0))

# ax.legend()

# %%% 4.2.3. Intra-event residuals

im_list1 = ['PGA (g)', 'PGV (cm sec^-1)', 'T0.300S',
            'T1.000S', 'T3.000S', 'T10.000S']
Tmodel = 'Tαmodels_stratify'
epsilon_comp = "epsilon_intra"

for im in im_list1:
    df_train_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                                '/uq_bstp_train_yb_15k_500.xlsx').iloc[:,1:]
    df_test_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                               '/uq_'+im+'bstp_test_15k_500.xlsx').iloc[:,1:]
    df_train_coef = pd.read_excel('./bootstrapping/'+Tmodel+'/uq_est_train_set.xlsx')
    # only train coef. due to the really similar coef. for test and train.
    
    # metadata_list = df_train_uq.columns.values[:df_train_uq.columns.get_loc(im)]
    metadata_list = ['Rjb (km)', 'Rx (km)',
                     'Vs30 (m/s)', 'Z2.5 (m)']
    metadata_range = [(1,300),(-250,250),(0,1200),(0,3000)]
    
    fig, axs = plt.subplots(1, 4, sharey=True, constrained_layout=True,
                            figsize=(15.97,3.22))
    error_bins = 10
    color_list = ['red','orange','green']
    
    for ax, v, vr in zip(axs.flat, metadata_list, metadata_range):
        # v(i.e. one of metadata) ~ epsilon_intra or epsilon_inter
        if v == "M":
            df_v = pd.concat([df_test_uq.loc[:,["Rake (deg)",v,epsilon_comp]],
                              df_train_uq.loc[:,["Rake (deg)",v,epsilon_comp]],
                             ])
        elif v == "Rake (deg)":
            df_v = pd.concat([df_test_uq.loc[:,["M",v,epsilon_comp]],
                               df_train_uq.loc[:,["M",v,epsilon_comp]],
                              ])
        else:
            df_v = pd.concat([df_test_uq.loc[:,["M","Rake (deg)",v,epsilon_comp]],
                              df_train_uq.loc[:,["M","Rake (deg)",v,epsilon_comp]],
                             ])
    
        df_v.dropna(axis=0,how='any',
                    subset=[v, epsilon_comp],
                    inplace=True)
        
        # Log scale for R
        scale = 'linear'
        df_v_hb = df_v
        if (v == 'Rjb (km)') | (v == 'Rrup (km)'):
            ax.set_xscale('log')
            scale = 'log'
            df_v_hb = df_v[df_v[v]>0]
            
        # Error bar values
        x_bin, ymd, yerr = error_md_std(df=df_v[[v,epsilon_comp]],
                                        x_col=v, y_col=epsilon_comp,
                                        x_range=vr, error_bins=error_bins,
                                        sym=True, minsamples=30, scale=scale)
        
        # Plot residual points
        ax.scatter(df_v[v], df_v[epsilon_comp],
                   alpha=0.3, edgecolors=None, s=10, zorder=1)
        # ax.hexbin(df_v_hb[v].values, df_v_hb[epsilon_comp].values,
        #           gridsize=100, xscale=scale, cmap='summer')
        
        # Plot errorbars
        # require the abs of error for both lower and higher
        if yerr.ndim > 1:  # for 'sym' = False
            yerr_abs_fc = np.concatenate((np.ones((1,yerr.shape[1]))*(-1),
                                          np.ones((1,yerr.shape[1]))), axis=0)
            yerr = yerr*yerr_abs_fc
        # ax.errorbar(x_bin, ymd, yerr=yerr,
        #             fmt='o', ms=5, color='k',
        #             ecolor='k', elinewidth=2, capsize=6, zorder=4)
        ax.scatter(x_bin,ymd,c='k',s=30,edgecolors='k',marker='d',
           zorder=4,label='bin mean')
        ax.scatter(x_bin,ymd+yerr,c='k',s=30,edgecolors='k',marker='P',
           zorder=4,label='bin sigma')
        ax.scatter(x_bin,ymd-yerr,c='k',s=30,edgecolors='k',marker='P',
           zorder=4,label='bin sigma')
        
        
        # Plot sigma_E models
        # overall sigma_E by MLE in this paper
        sigma_E_MLE = df_train_coef.loc[df_train_coef['IM']==im,
                                        "sigma_intra_A_std"].values
        ax.hlines([sigma_E_MLE, -sigma_E_MLE], vr[0], vr[1], label='MLE',
                  linestyles='--', colors=color_list[0], lw=2.0, zorder=3)
        
        # Plot setting
        # ax.set_title('Overall median = %.3f' %(np.nanmedian(ymd)))
        ax.set_xlabel(v)
        ax.set_ylabel(None)
        ax.set_xlim(vr)
        ax.set_ylim((-4.0,4.0))

# ax.legend()

# %%% 4.2.4. Boosting implies overfitting-induced epistemic uncertainty
"""
Show the score on test and train set to indicate underfitting and overfitting,
while show the uncertainty quantification varying with boosting, including
sigma_T, sigma_E, sigma_A_inter and sigma_A_intra.
"""
im_list3 = ['PGA (g)', 'T0.300S', 'T3.000S']  # 'T1.000S', 'T3.000S', 'T10.000S'
row_num = 3
col_num = 3
im_list3 = im_list3*row_num
Tmodel = 'Tαmodels_stratify'

gampe_range = (8.0, 20.0)  # GMPEs corresponds weak model

# load as-saved opt. es.
early_stopping_opt_df = pd.read_excel('./bootstrapping/'+Tmodel+
                                      '/summaryHGBR_15k_500.xlsx')

fig, axs = plt.subplots(row_num,col_num,sharex=True,
                        constrained_layout=True, figsize=(16.3 ,  8.38))
ax_idx = np.arange(0,row_num*col_num,1)

for ax, im, idx in zip(axs.flat, im_list3, ax_idx):
    df_mf = pd.read_excel('./bootstrapping/Flexibility/'+
                          im+'/summary_flexibility.xlsx')
    
    df_mf.sort_values(by='max_iter', ascending=True, inplace=True)
    
    es_best_point = early_stopping_opt_df.loc[early_stopping_opt_df['IM']==im,
                                              'n_iter'].values
    
    if idx < col_num:  # upper rows of axs for scores
        ax.plot(df_mf['max_iter'], df_mf['test_MAE'],
                '^-', color='red', label='test_MAE', zorder=3)
        ax.plot(df_mf['max_iter'], df_mf['train_MAE'],
                'o--', color='red', label='train_MAE', zorder=3)
        ax.set_xscale('log')
        
        ax2 = ax.twinx()
        ax2.plot(df_mf['max_iter'], df_mf['test_R2'],
                 '^-', color='gray', label='test_R2', zorder=3)
        ax2.plot(df_mf['max_iter'], df_mf['train_R2'],
                 'o--', color='gray', label='train_R2', zorder=3)
        # ax2.set_ylabel('MAE')
        
        ax.axvline(x=es_best_point[0], c='k' ,ls="--", lw=1.5,
                   label='best_es', zorder=2)
        
        # ax.legend(loc='upper right')
        # ax2.legend(loc='upper left')
        # ax.minorticks_on()
        ax.tick_params(axis="y",which="minor")
        ax.set_ylim((0.0,2.0))
        # ax2.minorticks_on()
        ax2.tick_params(axis="y",which="minor")
        ax2.set_ylim((0.0,1.0))
         
    elif (idx > col_num-1) & (idx < 2*col_num):  # mid row of axs for uncertainty components
        ax.plot(df_mf['max_iter'],
                df_mf[['test_epsilon_E_std','test_epsilon_A_std',
                       'test_epsilon_T_std']],
                '^-', label='test', zorder=3)
        ax.plot(df_mf['max_iter'],
                df_mf[['train_epsilon_E_std','train_epsilon_A_std',
                       'train_epsilon_T_std']],
                'o--', label='train', zorder=3)
        ax.set_xscale('log')
        
        ax.axvline(x=es_best_point[0], c='k' ,ls="--", lw=1.5,
                   label='best_es', zorder=2)
        
        # xlim = ax.get_xlim()
        ax.set_ylim((0.0,2.5))
        ylim = ax.get_ylim()
        # xrec = [gampe_range[0], gampe_range[1], gampe_range[1], gampe_range[0]]
        # yrec = [ylim[0], ylim[0], ylim[1], ylim[1]]
        # ax.fill(xrec, yrec, facecolor='.9', alpha=1.0, zorder=1)
        
        # ax.minorticks_on()
        ax.tick_params(axis="y",which="minor")
        # ax.legend()
        
    else:  # lower row of axs for aleatory uncertainty components
        ax.plot(df_mf['max_iter'],
                df_mf[['test_epsilon_E_std',
                       'test_sigma_inter_A_std','test_sigma_intra_A_std']],
                '^-', label='test', zorder=3)
        ax.plot(df_mf['max_iter'],
                df_mf[['train_epsilon_E_std',
                       'train_sigma_inter_A_std','train_sigma_intra_A_std']],
                'o--', label='train', zorder=3)
        ax.set_xscale('log')
        
        ax.axvline(x=es_best_point[0], c='k' ,ls="--", lw=1.5,
                   label='best_es', zorder=2)
        
        # xlim = ax.get_xlim()
        ax.set_ylim((0.0,2.5))
        ylim = ax.get_ylim()
        # xrec = [gampe_range[0], gampe_range[1], gampe_range[1], gampe_range[0]]
        # yrec = [ylim[0], ylim[0], ylim[1], ylim[1]]
        # ax.fill(xrec, yrec, facecolor='.9', alpha=1.0, zorder=1)
        
        # ax.minorticks_on()
        ax.tick_params(axis="y",which="minor")
        # ax.legend()

ax.set_xlim((1.0,10000))
# plt.tight_layout()

# %% 4.3. Compared to present epistemic model
"""
1) Plot the general epsilon_E for various IMs
2) Plot the epsilon_E vs. various X with errorbar to show the Homogeneity Of Variance.
    X = 12 numerical and 1 categorical
3) Plot the benchmark model in the plot 2):
    a) A minimal epsilon_E_std of 0.083 for NGA-West2;
    b) Atik and Young 2014 within-model epistemic uncertainty model.
"""

# %%% 4.3.1. epsilon_E overall distribtion
"""
Demonstrate the dist. of epsilon_E with the estimated Median and Std by MLE
"""
Tmodel = 'Tαmodels_stratify'

fig, axs = plt.subplots(2,5,sharex=True,sharey=True,
                        constrained_layout=True, figsize=(16.69,5.72))

im_list_t = ['PGA (g)', 'T0.030S', 'T0.050S',
             'T0.100S', 'T0.300S', 'T0.500S',
             'T1.000S', 'T3.000S', 'T5.000S', 'T10.000S']  # for test
im_values_t = [0, 0.03, 0.05,
               0.1, 0.3, 0.5,
               1.0, 3.0, 5.0, 10.0]

# Load y_true, y_pred_train, y_unbias, sigma_E,A,T
df_train = pd.read_excel('./bootstrapping/'+Tmodel+'/uq_est_train_set.xlsx')
df_test = pd.read_excel('./bootstrapping/'+Tmodel+'/uq_est_test_set.xlsx')

for ax, im, im_val in zip(axs.flat, im_list_t, im_values_t):
    df_train_im = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                                '/uq_bstp_train_yb_15k_500.xlsx')
    df_test_im = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                               '/uq_'+im+'bstp_test_15k_500.xlsx')
    
    ax.hist(df_train_im["epsilon_E"], bins=50, color='C0',
            alpha=0.5, linewidth=0.5, edgecolor="white", label='Train')
    ax.hist(df_test_im["epsilon_E"], bins=100, color='C1',
            alpha=0.5, linewidth=0.5, edgecolor="white", label='Test')
    
    ep_E_std_train = df_train.loc[df_train['IM']==im, "epsilon_E_std"].values
    ep_E_std_test = df_test.loc[df_test['IM']==im, "epsilon_E_std"].values
    
    ax.axvline(x=df_train.loc[df_train['IM']==im, "epsilon_E_md"].values,
               c='C0' ,ls="-", lw=1.5, label='Train')
    ax.axvline(x=df_test.loc[df_test['IM']==im, "epsilon_E_md"].values,
               c='C1' ,ls="-", lw=1.5, label='Test')
    ax.axvline(x=ep_E_std_train,
               c='C0' ,ls="--", lw=1.5, label='Train')
    ax.axvline(x=ep_E_std_test,
               c='C1' ,ls="--", lw=1.5, label='Test')
    ax.axvline(x=-ep_E_std_train,
               c='C0' ,ls="--", lw=1.5, label='Train')
    ax.axvline(x=-ep_E_std_test,
               c='C1' ,ls="--", lw=1.5, label='Test')
        
    print(im+' Std. = %.3f, %.3f' %(ep_E_std_train, ep_E_std_test))
    
    ax.set_title(im)
    # ax.set_xlabel(None)
    # ax.set_ylabel(None)
    ax.set_xlim((-1.0,1.0))
    # ax.set_ylim((0.0,0.5))

# ax.legend()

# %%% 4.3.2. epsilon_E with various X for ALL 15k data
"""
1) Over ALL 15k data including the test and train
    X = 12 numerical and 1 categorical
2) Plot 0.083 and AY14 model
"""

im = 'T3.000S'
im_val = 3.0
Tmodel = 'Tαmodels_stratify'
df_train_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                            '/uq_bstp_train_yb_15k_500.xlsx').iloc[:,1:]
df_test_uq = pd.read_excel('./bootstrapping/'+Tmodel+'/'+im+
                           '/uq_'+im+'bstp_test_15k_500.xlsx').iloc[:,1:]
df_train_coef = pd.read_excel('./bootstrapping/'+Tmodel+'/uq_est_train_set.xlsx')
# only train coef. due to the really similar coef. for test and train.

metadata_list = df_train_uq.columns.values[:df_train_uq.columns.get_loc(im)]
metadata_range = [(3.0,8.0),(0.0,100),(-180,180),(0.0,25.0),(0.0,18.0),(0.0,45.0),
                  (1,300),(1,300),(-250,250),(-500,500),(0,2000),(0,8000)]

fig, axs = plt.subplots(3, 4, sharey=True, constrained_layout=True,
                        figsize=(15.97,9.66))
error_bins = 10
color_list = ['red','orange','green']

for ax, v, vr in zip(axs.flat, metadata_list, metadata_range):
    # v(i.e. one of metadata) ~ epsilon_E
    if v == "M":
        df_v = pd.concat([df_train_uq.loc[:,["Rake (deg)",v,"epsilon_E"]],
                         df_test_uq.loc[:,["Rake (deg)",v,"epsilon_E"]]])
    elif v == "Rake (deg)":
        df_v = pd.concat([df_train_uq.loc[:,["M",v,"epsilon_E"]],
                          df_test_uq.loc[:,["M",v,"epsilon_E"]]])
    else:
        df_v = pd.concat([df_train_uq.loc[:,["M","Rake (deg)",v,"epsilon_E"]],
                         df_test_uq.loc[:,["M","Rake (deg)",v,"epsilon_E"]]])

    df_v.dropna(axis=0,how='any',
                subset=[v, 'epsilon_E'],
                inplace=True)
    
    # Log scale for R
    scale = 'linear'
    df_v_hb = df_v
    if (v == 'Rjb (km)') | (v == 'Rrup (km)'):
        ax.set_xscale('log')
        scale = 'log'
        df_v_hb = df_v[df_v[v]>0]
        
    # Error bar values
    x_bin, ymd, yerr = error_md_std(df=df_v[[v,'epsilon_E']],
                                    x_col=v, y_col='epsilon_E',
                                    x_range=vr, error_bins=error_bins,
                                    sym=True, minsamples=10, scale=scale)
    
    # Plot residual points
    ax.scatter(df_v[v], df_v['epsilon_E'],
               alpha=0.3, edgecolors=None, s=10, zorder=1)
    # ax.hexbin(df_v_hb[v].values, df_v_hb['epsilon_E'].values,
    #           gridsize=100, xscale=scale, cmap='summer')
    
    # Plot errorbars
    # require the abs of error for both lower and higher
    if yerr.ndim > 1:  # for 'sym' = False
        yerr_abs_fc = np.concatenate((np.ones((1,yerr.shape[1]))*(-1),
                                      np.ones((1,yerr.shape[1]))), axis=0)
        yerr = yerr*yerr_abs_fc
    # ax.errorbar(x_bin, ymd, yerr=yerr,
    #             fmt='o', ms=5, color='k',
    #             ecolor='k', elinewidth=2, capsize=6, zorder=4)
    ax.scatter(x_bin,ymd,c='k',s=30,edgecolors='k',marker='d',
               zorder=4,label='bin mean')
    ax.scatter(x_bin,ymd+yerr,c='k',s=30,edgecolors='k',marker='P',zorder=4)
    ax.scatter(x_bin,ymd-yerr,c='k',s=30,edgecolors='k',marker='P',
               zorder=4,label='bin sigma')
    
    # Plot sigma_E models
    # 1) overall sigma_E by MLE in this paper
    sigma_E_MLE = df_train_coef.loc[df_train_coef['IM']==im,'epsilon_E_std'].values
    ax.hlines([sigma_E_MLE, -sigma_E_MLE], vr[0], vr[1], label='MLE',
              linestyles='--', colors=color_list[0], lw=2.0, zorder=3)
    # 2) 0.083
    ax.hlines([0.083, -0.083], vr[0], vr[1], label='0.083',
              linestyles='--', colors=color_list[1], lw=2.0, zorder=3)
    # 3) AY14
    df_v.dropna(axis=0,how='any',
            subset=['Rake (deg)', 'M'],
            inplace=True)
    sigma_E_AY14 = sigma_E_model_AY14(rake=df_v["Rake (deg)"].values, T=im_val,
                                      M=df_v["M"].values)
    ax.scatter(df_v[v],sigma_E_AY14,marker='s', label='AY14',
               alpha=0.5,edgecolors=None,s=10,c=color_list[2],zorder=2)
    ax.scatter(df_v[v],-sigma_E_AY14,marker='s',
               alpha=0.5,edgecolors=None,s=10,c=color_list[2],zorder=2)
    
    # Plot setting
    # ax.set_title('Overall median = %.3f' %(np.nanmedian(ymd)))
    ax.set_xlabel(v)
    ax.set_ylabel(None)
    ax.set_xlim(vr)
    ax.set_ylim((-0.6,0.6))

# ax.legend()

# %%% 4.3.2. epsilon_E with Regions for ALL 15k data
"""
Plot epsilon_E with various regions, with 3 kinds of sigma_E model
"""
Tmodels = 'Tαmodels_stratify'

im_list1 = ['PGA (g)', 'T0.030S', 'T0.300S',
            'T1.000S', 'T3.000S', 'T10.000S']
im_values1 = [0.001, 0.03, 0.3, 1.0, 3.0, 10.0]

df_train_coef = pd.read_excel('./bootstrapping/'+Tmodels+'/uq_est_train_set.xlsx')

fig, axs = plt.subplots(2,3,sharex=True,sharey=True,
                        constrained_layout=True, figsize=(16.31,7.93))

color_list = ['red','orange','pink']

for ax, im, im_val in zip(axs.flat, im_list1, im_values1):
    # Load data
    df_train_uq = pd.read_excel('./bootstrapping/'+Tmodels+'/'+im+
                            '/uq_bstp_train_yb_15k_500.xlsx').iloc[:,1:]
    df_test_uq = pd.read_excel('./bootstrapping/'+Tmodels+'/'+im+
                               '/uq_'+im+'bstp_test_15k_500.xlsx').iloc[:,1:]
    df_global = pd.concat([df_train_uq.loc[:,["Region","epsilon_E"]],
                           df_test_uq.loc[:,["Region","epsilon_E"]]])
    df_global_AY14 = pd.concat([df_train_uq.loc[:,["M","Rake (deg)","Region","epsilon_E"]],
                           df_test_uq.loc[:,["M","Rake (deg)","Region","epsilon_E"]]])
    df_global.dropna(axis=0,how='any',subset=['epsilon_E'],inplace=True)
    df_global_AY14.dropna(axis=0,how='any',subset=['M','Rake (deg)','epsilon_E'],inplace=True)
    
    # Plot epsilon_E for various regions
    sns.violinplot(data=df_global, x="Region", y="epsilon_E",
                   ax=ax, color="C0", inner=None, zorder=1)
    rg_array = df_global['Region'].unique()
    rg_mean = np.zeros(rg_array.shape)
    rg_std = np.zeros(rg_array.shape)
    for i,rg in enumerate(rg_array):
        rg_mean[i] = df_global.loc[df_global['Region']==rg,'epsilon_E'].mean()
        rg_std[i] = df_global.loc[df_global['Region']==rg,'epsilon_E'].std()
        
    ax.scatter(rg_array,rg_mean,c='k',s=30,edgecolors='k',marker='d',
                zorder=4,label='bin mean')
    ax.scatter(rg_array,rg_mean+rg_std,c='k',s=30,edgecolors='k',marker='P',zorder=4)
    ax.scatter(rg_array,rg_mean-rg_std,c='k',s=30,edgecolors='k',marker='P',
                zorder=4,label='bin sigma')
    
    
    # Plot sigma_E model
    xlim_values = ax.get_xlim()
    # 1) overall sigma_E by MLE in this paper
    sigma_E_MLE = df_train_coef.loc[df_train_coef['IM']==im,'epsilon_E_std'].values
    ax.hlines([sigma_E_MLE, -sigma_E_MLE], xlim_values[0], xlim_values[1], label='MLE',
              linestyles='--', colors=color_list[0], lw=2.0, zorder=3)
    # 2) 0.083
    ax.hlines([0.083, -0.083], xlim_values[0], xlim_values[1], label='0.083',
              linestyles='--', colors=color_list[1], lw=2.0, zorder=3)
    # 3) AY14
    sigma_E_AY14 = sigma_E_model_AY14(rake=df_global_AY14["Rake (deg)"].values,
                                      T=im_val,
                                      M=df_global_AY14["M"].values)
    ax.scatter(df_global_AY14["Region"],sigma_E_AY14,marker='s', label='AY14',
               alpha=0.5,edgecolors=None,s=15,c=color_list[2],zorder=2)
    ax.scatter(df_global_AY14["Region"],-sigma_E_AY14,marker='s',
               alpha=0.5,edgecolors=None,s=15,c=color_list[2],zorder=2)
    
    
    # Plot setting
    ax.set_title(im)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

ax.set_xlim((-0.5,6.5))
ax.set_ylim((-0.6,0.6))
# ax.legend()

# %% 4.4. Case study: conceptional scenario approach
"""
1) Predefine case study: X input samples
2) Plot the case study results over Rx for FM x3, M x7
    How to unify different Rjb,Rrup into Rx?
"""

# %%% 4.4.1. IM-R plot with various M, FM
from mlgmmuq import case_study
import time

Tmodel = 'Tαmodels_stratify'
random_state = 500
num_bstp = 40

Rx_start = 1.0
Rx_end = 100.0
Rx_num = 100
M_list = np.arange(5.0,7.5,0.5)
FM_list = ['SS', 'RS', 'NM']  # ['SS', 'RS', 'NM']
Vs30 = 700.0  # 260.0, 760.0
region = 1 # Region label: 0=Other, 1=WUS, 2=Japan, 3=WCN, 4=Italy, 5=Turkey, 6=Taiwan
color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

im_list1 = ['PGA (g)', 'T0.300S', 'T1.000S', 'T3.000S']
# ['PGA (g)', 'T0.300S', 'T1.000S', 'T3.000S']
period_value1 = [0.0, 0.3, 1.0, 3.0]
# [0.0, 0.3, 1.0, 3.0]

for im, period_value in zip(im_list1, period_value1):
    fig, axs = plt.subplots(1,3,sharex=True,sharey=True,
                            figsize=(15.8,3.9),constrained_layout=True)
    start_time = time.time()
    
    for ax, FM in zip(axs.flat, FM_list):
        for M, c in zip(M_list, color_list):
            # Case study params
            df_case = case_study(M=M, FM=FM, Vs30=Vs30, region=region, features=features,
                                 Rx_start=Rx_start, Rx_end=Rx_end, Rx_num=Rx_num)
            
            # Model params
            # load hyperparameters
            model_coef = pd.read_excel('./bootstrapping/'+Tmodel+'/summaryHGBR_15k_500.xlsx')
            common_params = dict(
                loss='absolute_error', l2_regularization=2,
                max_iter=model_coef.loc[model_coef["IM"]==im, "n_iter"].values[0],
                min_samples_leaf=model_coef.loc[model_coef["IM"]==im, "min_samples_leaf"].values[0],
                max_leaf_nodes=model_coef.loc[model_coef["IM"]==im, "max_leaf_nodes"].values[0],
                categorical_features=df_case.columns.get_loc('Region'),
                max_depth=None, learning_rate=0.1,
                random_state=random_state)
            sourceFeatures = ['Earthquake Magnitude',
                              'Dip (deg)','Rake Angle (deg)',
                              'Hypocenter Depth (km)','Ztor (km)',
                              'Fault Rupture Width (km)',
                              'Region',
                              ]
            pathFeatures = ['Rjb (km)', 
                            'Rrup (km)','Ry 2', 'Rx',
                            ]
            siteFeatures = ['Vs30 (m/s)',
                            'Northern CA/Southern CA - H11 Z2.5 (m)',
                            ]
            LUF = 'Lowest Usable Freq - Ave. Component (Hz)'
            features0 = siteFeatures + pathFeatures + sourceFeatures
            
            # Modeling
            # data split
            if period_value != 0:
                LUF_flag = 1/period_value > data15k[LUF]
            else:
                LUF_flag = np.ones((len(data15k),), dtype=bool)
            X = data15k.loc[LUF_flag, features0]
            y = np.log(data15k.loc[LUF_flag, im])
            X.rename(columns={'Northern CA/Southern CA - H11 Z2.5 (m)': 'Z2.5 (m)',
                              'Rx': 'Rx (km)',
                              'Ry 2': 'Ry (km)',
                               'Earthquake Magnitude': 'M',
                               'Rake Angle (deg)': 'Rake (deg)',
                               'Hypocenter Depth (km)': 'Dhyp (km)',
                               'Fault Rupture Width (km)': 'W (km)',
                               },
                      inplace=True)
            # X_train0, X_test, y_train0, y_test = train_test_split(X,y,test_size=0.2,
            #                                                       random_state=random_state,
            #                                                       shuffle=True)
            X_train0, y_train0 = X, y  # modeling over the whole dataset
            df_pred = pd.DataFrame()
            for b in np.arange(num_bstp):
                # bootstrapping subset Db*
                train_set_B = pd.concat([X_train0, y_train0],axis=1).\
                    sample(frac=1.0, axis=0, replace=True, random_state=b)
                X_train = train_set_B.loc[:, features]
                y_train = train_set_B.loc[:, im]
                
                # train Tα-model on Db* with the optimized hyperparameters
                reg_b = HistGradientBoostingRegressor(**common_params).\
                    fit(X_train,y_train)
                
                # prediction
                df_pred[im+'_b'+str(b)] = np.exp(reg_b.predict(df_case))
                
                print('Finished '+im+' M='+str(M)+', '+FM+' ,b='+
                      str(b)+'/'+str(num_bstp-1))
            
            # Plot
            ax.plot(df_case["Rx (km)"], df_pred, label='bootstrap',
                    ls='-', lw=1.0, c=c, alpha=0.5, zorder=1)
            ax.plot(df_case["Rx (km)"], np.mean(df_pred.values,axis=1), label='y_unbiased',
                    ls='-', lw=1.5, c='black', zorder=2)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim((Rx_start,Rx_end))
            ax.set_ylim((1e-4,1e0))
    
    end_time = time.time()
    print('Time consuming: %.2f' %(end_time - start_time))

# %%% 4.4.2. IM-R plot with various M, FM
from mlgmmuq import case_study
import time

Tmodel = 'Tαmodels_stratify'
random_state = 500
num_bstp = 40
Rx = 30.0
Vs30_list = [200.0, 400.0, 760.0, 1000.0, 1500.0]# np.arange(200.0,1600.0,200.0), [200.0, 400.0, 760.0, 1000.0, 1500.0]
M_list = [5.0, 6.0, 7.0]

color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

im_list1 = im_list0  # im_list0, ['T0.300S', 'T3.000S']
period_value1 = im_values0  # im_values0, [0.3, 3.0]

start_time = time.time()

# Params.
df_case = case_study(M=7.0, FM='SS', Vs30=760.0, region=1, features=features,
             Rx_start=Rx, Rx_end=Rx, Rx_num=1)
# df_case["Ztor (km)"] = np.array([0.0])
# df_case["W (km)"] = np.array([None])  # W should not be None
for i in range(0,len(Vs30_list)*len(M_list)-1): # copy row for Vs30 list
    df_case.loc[i+1] = df_case.loc[0]
xv, yv = np.meshgrid(M_list, Vs30_list)
df_case["M"] = xv.ravel()
df_case["Vs30 (m/s)"] = yv.ravel()

# Initialize
pred_array = np.zeros((len(Vs30_list)*len(M_list),len(im_list1),num_bstp))

for im, period_value, i in zip(im_list1, period_value1, np.arange(len(im_list1))):
    # Model params
    # load hyperparameters
    model_coef = pd.read_excel('./bootstrapping/'+Tmodel+'/summaryHGBR_15k_500.xlsx')
    common_params = dict(
        loss='absolute_error', l2_regularization=2,
        max_iter=model_coef.loc[model_coef["IM"]==im, "n_iter"].values[0],
        min_samples_leaf=model_coef.loc[model_coef["IM"]==im, "min_samples_leaf"].values[0],
        max_leaf_nodes=model_coef.loc[model_coef["IM"]==im, "max_leaf_nodes"].values[0],
        categorical_features=df_case.columns.get_loc('Region'),
        max_depth=None, learning_rate=0.1,
        random_state=random_state)
    sourceFeatures = ['Earthquake Magnitude',
                      'Dip (deg)','Rake Angle (deg)',
                      'Hypocenter Depth (km)','Ztor (km)',
                      'Fault Rupture Width (km)',
                      'Region',
                      ]
    pathFeatures = ['Rjb (km)', 
                    'Rrup (km)','Ry 2', 'Rx',
                    ]
    siteFeatures = ['Vs30 (m/s)',
                    'Northern CA/Southern CA - H11 Z2.5 (m)',
                    ]
    LUF = 'Lowest Usable Freq - Ave. Component (Hz)'
    features0 = siteFeatures + pathFeatures + sourceFeatures
    
    # Modeling
    # data split
    if period_value > 0.001:
        LUF_flag = 1/period_value > data15k[LUF]
    else:
        LUF_flag = np.ones((len(data15k),), dtype=bool)
    X = data15k.loc[LUF_flag, features0]
    y = np.log(data15k.loc[LUF_flag, im])
    X.rename(columns={'Northern CA/Southern CA - H11 Z2.5 (m)': 'Z2.5 (m)',
                      'Rx': 'Rx (km)',
                      'Ry 2': 'Ry (km)',
                       'Earthquake Magnitude': 'M',
                       'Rake Angle (deg)': 'Rake (deg)',
                       'Hypocenter Depth (km)': 'Dhyp (km)',
                       'Fault Rupture Width (km)': 'W (km)',
                       },
              inplace=True)
    # X_train0, X_test, y_train0, y_test = train_test_split(X,y,test_size=0.2,
    #                                                       random_state=random_state,
    #                                                       shuffle=True)
    X_train0, y_train0 = X, y  # modeling over the whole dataset
    for ib,b in enumerate(np.arange(num_bstp)):
        # bootstrapping subset Db*
        train_set_B = pd.concat([X_train0, y_train0],axis=1).\
            sample(frac=1.0, axis=0, replace=True, random_state=b)
        X_train = train_set_B.loc[:, features]
        y_train = train_set_B.loc[:, im]
        
        # train Tα-model on Db* with the optimized hyperparameters
        reg_b = HistGradientBoostingRegressor(**common_params).\
            fit(X_train,y_train)
        
        # prediction
        pred_array[:,i,ib] = np.exp(reg_b.predict(df_case))
        
        print('Finished '+im+',b='+str(b)+'/'+str(num_bstp-1))

end_time = time.time()
print('Time consuming: %.2f' %(end_time - start_time))

# Plot
fig, axs = plt.subplots(1,2,sharex=True,sharey=True,
                        figsize=(10.8,3.9),constrained_layout=True)
# fig, axs = plt.subplots(1,3,sharex=True,sharey=True,constrained_layout=True)
# for ax, M, i in zip(axs.flat, M_list, np.arange(len(M_list))):
for ax, M, i in zip(axs.flat, [6.0,7.0], [1,2]):
    plot_pred = pred_array[np.arange(i,len(Vs30_list)*len(M_list),len(M_list)),:,:]
    for Vs30, c, v in zip(Vs30_list, color_list, np.arange(len(Vs30_list))):
        # Plot for a Vs30
        ax.plot(period_value1, plot_pred[v,:,:],
                ls='-', lw=1.0, c=c, alpha=0.1, zorder=1)
        ax.plot(period_value1, np.mean(plot_pred[v,:,:],axis=1), label='Vs30='+str(Vs30),
                ls='-', lw=2, c=c, zorder=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((0.01,10.0))
    ax.set_ylim((1e-3,1e0))
    
# ax.legend()