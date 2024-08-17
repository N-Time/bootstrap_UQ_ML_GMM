# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:49:17 2023

@author: Vincent NT, ylpxdy@live.com, for Bootstrap framework
@author: Jianguang He, for LightGBM implementation

Train the model in the paper,
Wen T, He J, Jiang L, Du Y, Jiang L. A simple and flexible bootstrap-based framework to quantify epistemic uncertainty of ground motion models by light gradient boosting machine. Applied Soft Computing 2023:111195. https://doi.org/10.1016/j.asoc.2023.111195.

A test on UQ of ML-based GMMs, y = f(x;c) + e
1) UQ includes: Total error, Prediction intervals ~ sigma^2
      [Khosravi, 2011, IEEE: NN]
      a) Aleatory:  P(y|x) ~ sigma_e^2
      b) Epistemic: P(c|D_train) ~ sigma_yhat^2
2) Method for P(y|x): generally sigma^2 - sigma_yhat^2 assuming their independence
      a) Sample regression residual covariance [A. Do, ES, 2020]
      b) Iterative process for random effect [Abrahamson, BSSA, 1992; F. Khosravikia, CG, 2021]
      c) Prediction intervals [Meinshausen, JMLR, 2006]
3) Method for P(c|D_train):
      a) Asymptotic standard errors [L.A. Atik, EqSp, 2014]
      b) Bootstrapping
      c) Bayesian regression [Sreenath, EESD, 2022]
4) Method for UQ of y_ij given f_i(x,c_i) and f_j(x,c_j)
      the covariance structure of IMs in GMM

https://machinelearningmastery.com/prediction-intervals-for-machine-learning/

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from joblib import dump, load

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,median_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor


# %% Data preparation
# # Read original NGA-West2
# data0 = pd.read_excel(r'D:\Wen\Jupyter_working_path\Project\Nonparametric_GMM\PEER_NGA_West2_8exLabel_Region_230327.xlsx')

# # Filter and get NW2SD-15k
# data15k = data0[data0['Excluding Label'] == 0]

# Filtered data
data15k = pd.read_excel(r'D:\Wen\Jupyter_working_path\Project\Nonparametric_GMM\PEER_NGA_West2_size_14_6k_Region_230602.xlsx')

# Assign -999 as np.nan
data15k = data15k.replace(to_replace=[-999,'-999'],value=[np.nan,np.nan])

# Column list
columnList = data15k.columns.to_list()

data15k.rename(columns={'PGV (cm/sec)': 'PGV (cm sec^-1)'}, inplace=True)
label = data15k.columns.to_list()[data15k.columns.get_loc('PGA (g)'):
                                  data15k.columns.get_loc('T20.000S')]
    
sourceFeatures = ['Earthquake Magnitude',
                  'Dip (deg)','Rake Angle (deg)',
                  'Hypocenter Depth (km)','Ztor (km)',
                  'Fault Rupture Width (km)',
                  'Region',
                  ]
pathFeatures = ['Rjb (km)','Rrup (km)',
                'Rx','Ry 2',
                ]
siteFeatures = ['Vs30 (m/s)',
                'Northern CA/Southern CA - H11 Z2.5 (m)',
                ]
LUF = 'Lowest Usable Freq - Ave. Component (Hz)'
features = siteFeatures + pathFeatures + sourceFeatures

# Drop the metadata
key_metadata_list = []
data15k.dropna(axis=0,how='any',\
                subset=['Earthquake Magnitude',
                        'Rjb (km)',
                        'EpiD (km)',
                        'Vs30 (m/s)',
                        'Lowest Usable Freq - Ave. Component (Hz)']+label,
                inplace=True)

data15k['Region'] = data15k['Region'].astype('category')
# %% Bootstrapping Tα-model: B-model for only paramters
"""
The Tα-model needs only 1 run CV for the training set Dtr, then
    with the optimized hyperparameters to 
    train model on all B Bootstrapping dataset Db*.
"""

im_list = [
            'PGA (g)',
            'PGV (cm sec^-1)',
            'T0.010S', 'T0.020S', 'T0.030S', 'T0.050S', 'T0.075S',
            'T0.100S', 'T0.150S', 'T0.200S', 'T0.300S', 
            'T0.400S', 'T0.500S', 'T0.750S',
            'T1.000S', 'T1.500S', 'T2.000S',
            'T3.000S', 'T4.000S', 'T5.000S', 'T7.500S', 'T10.000S',
            ]
im_values = [
            0.0,
            0.0, 
            0.01, 0.02, 0.03, 0.05, 0.075,
            0.1, 0.15, 0.2, 0.3, 
            0.4, 0.5, 0.75,
            1.0, 1.5, 2.0, 
            3.0, 4.0, 5.0, 7.5, 10.0,
            ]

# ####To find the early-stopping threshold by the CI of training score
# rs_list = np.arange(30,630,30)  # obtain CI of MAE by various data split
# res_score_test = np.nan*np.zeros((300+1,rs_list.shape[0],len(im_list)), dtype=np.float64)
# res_score_train = np.nan*np.zeros((300+1,rs_list.shape[0],len(im_list)), dtype=np.float64)
# for j, random_state in enumerate(rs_list):

main_folder = './bootstrapping/savemodel/'
random_state = 500   # a specific state of 500 for a repeatable test
# This rs is used in
#   1) Original dataset D splitting into Dtr and Dte;
#   2) An additional splitting of Dtr for optimize early stopping;
es_way = 2   # earlpy-stopping way: 0 = False, 1 = es in cv, 2 = es with cv best HPs, 3 = es by cutting cv ensembles
es_show = False  # early-stopping: 0 = False, 1 = True
max_iter = 100  # for es_way = 0 or 1
es_tol = [6e-3, 3E-3, 1e-3]   # a step-wise tol for various IM samples, ref: 'EarlyStoppingThreshold.py'
n_iter_no_change = 4   # about 10% of max_iter, e.g. 4 for ~40
num_bstp = 40  # >0 for Boostrap: generally 20~200, no significant change for B > 40
# Early-stopping setting:
#   1) for bootstrap modeling, es_way=2, max_iter=100, num_bstp=40
#   2) for stage prediction, es_way=0, max_iter=300, num_bstp=0
save_model = True
loss_func = 'absolute_error'
score_metric = 'neg_mean_absolute_error'

es_test_score = np.nan*np.zeros((max_iter,len(im_list)))
es_train_score = np.nan*np.zeros((max_iter,len(im_list)))
df_im_results = pd.DataFrame(columns=['IM',
                                'max_leaf_nodes', 'min_samples_leaf', 'n_iter',
                                'train_R2', 'train_RMSE', 'train_MAE',
                                'cv_best_mean_test_score',
                                'cv_best_mean_train_score',
                                'test_R2','test_RMSE','test_MAE',
                                ])

start0time = time.time()
for im, period_value, k in zip(im_list, im_values, np.arange(0,len(im_list))):

    if period_value != 0:
        LUF_flag = 1/period_value > data15k[LUF]
    else:
        LUF_flag = np.ones((len(data15k),), dtype=bool)
    
    # Check the output fold and build it
    os.path.exists(main_folder+im)
    if not os.path.exists(main_folder+im):
        os.makedirs(main_folder+im)
    
    # Remove events with N = 1
    data15k_LUF = data15k.loc[LUF_flag,:]
    df0 = data15k_LUF.set_index('EQID', drop=False)  # index to nature as origin data
    few_rc = data15k_LUF['EQID'].value_counts()
    few_rc = few_rc[few_rc < 2].index   # index list of N < 2
    df1 = df0.drop(few_rc)
    label_few = np.setdiff1d(df0['Record Sequence Number'].values,
                             df1['Record Sequence Number'].values)
    for i in label_few:
        data15k_LUF = data15k_LUF.loc[~(data15k_LUF['Record Sequence Number']==i)]
    
    # Data splitting
    categorical_columns = data15k_LUF.select_dtypes(include="category").columns
    X = data15k_LUF.loc[:, features]
    y = np.log(data15k_LUF.loc[:, im])
    
    # Data splitting
    X_train0, X_test, y_train0, y_test = train_test_split(X,y,test_size=0.2,
                                                         random_state=random_state,
                                                         shuffle=True,
                                                         stratify=data15k_LUF["EQID"])
    
    # Save train0, X and y, to concat y_hat_b for all B
    y_hat_b_all = pd.concat([X_train0, y_train0], axis=1)
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Current training the original Tα-model on Dtr')
    starttime = time.time()
    
    # CV for grid search
    if es_way == 1:
        esFlag = True
    else:
        esFlag = False
        
    common_params = dict(
        loss=loss_func, l2_regularization=2,
        max_iter=max_iter,  # manually define early-stopping
        categorical_features=np.array(X_train0.columns.get_loc('Region')),
        max_depth=None, learning_rate=0.1,
        early_stopping=esFlag,
        # validation_fraction=0.1, n_iter_no_change=10, tol=1e-7,
        random_state=random_state)
    
    param = {'min_samples_leaf':[16,20,24,28,32,36,40],#默认20, 'min_data_in_leaf' in LightGBM
             'max_leaf_nodes':[25,27,29,31,33,35,37]}#默认31,公式=2^(max_depth),max_depth默认-1, 'num_leaves' in LightGBM
    # 30,40,50,60,65,70,75,80  # leads less n_iter_ but more data split random
    
    # Find the optimized hyperparameters for X_train0 by 1 run of CV
    grid = GridSearchCV(estimator = HistGradientBoostingRegressor(**common_params),
                        param_grid=param,
                        cv=10,
                        verbose = 0,
                        scoring =score_metric,
                        return_train_score = True,
                        refit = False,
                        n_jobs=12)
    grid.fit(X_train0, y_train0)
    print(grid.best_params_, grid.best_score_)
    
    common_params.update(grid.best_params_)  # load best param. from CV
    # reg0 = HistGradientBoostingRegressor(**common_params).fit(X_train0,y_train0)
    
    common_params1 = common_params.copy()
    if es_way == 2:
        # Set early-stopping, es_show = 2 or 3
        if type(es_tol) == list:
            # step-wise tol: 0.0~1.0~3.0~10.0
            if period_value < 1.0:
                es_tol_app = es_tol[0]
            elif (period_value < 3.0) and (period_value >= 1.0):
                es_tol_app = es_tol[1]
            else:
                es_tol_app = es_tol[2]
        else:
            # single tol
            es_tol_app = es_tol

        common_params1.update(dict(
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=n_iter_no_change, tol=es_tol_app,
            ))
    
    # es_way = 0, 1, 2
    reg = HistGradientBoostingRegressor(**common_params1).fit(X_train0,y_train0)
    n_estimator = reg.n_iter_  ### replace all 'reg.n_iter_'
    
    if es_way == 3:  # es by cutting first specific estimators
        print('!!~~Not Completed~~!!')
    
    print('The best iteration: %s' % reg.n_iter_)
    
    # show Early stopping
    # stage prediction
    test_score = np.zeros((reg.n_iter_,), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = mean_absolute_error(y_test, y_pred)
    
    train_score = np.zeros((reg.n_iter_,), dtype=np.float64)  # full train set
    for i, y_pred in enumerate(reg.staged_predict(X_train0)):
        train_score[i] = mean_absolute_error(y_train0, y_pred)
    
    es_test_score[0:len(test_score),k] = test_score
    es_train_score[0:len(train_score),k] = train_score
    # plot
    if (es_way > 0) & (es_show == True):
        fig, ax = plt.subplots(1,1,figsize=(7.0, 6.0))
        ax.set_title(score_metric)
        ax.plot(np.arange(reg.n_iter_+1), -reg.train_score_,  # train set in es, e.g. 0.9
                "b-",label="Training Set Deviance")
        ax.plot(np.arange(reg.n_iter_)+1, test_score,
                "r-", label="Test Set Deviance")
        ax.legend(loc="upper right")
        ax.set_xlabel("Boosting Iterations")
        ax.set_ylabel("Deviance")
        
    
    pdCV = pd.DataFrame(grid.cv_results_)
    pdCV.to_excel(main_folder+im+'/'+
                  im+'LGBMGrid_15k_'+str(random_state)+'.xlsx')

    # Test result
    X_test_out = pd.concat([X_test, y_test], axis=1)
    test_set_predict = reg.predict(X_test)
    X_test_out.insert(loc=X_test_out.shape[1],
                      column='test_pred_'+im, value=test_set_predict)
    test_R2, test_RMSE, test_MAE = r2_score(y_test, test_set_predict),\
                    mean_squared_error(y_test, test_set_predict)**0.5,\
                    mean_absolute_error(y_test, test_set_predict)
    print('R2 RMSE MAE: Test vs Train')
    print('%.3f %.3f %.3f' % (test_R2, test_RMSE, test_MAE))

    
    # Train result
    train_set_predict = reg.predict(X_train0)
    train_R2, train_RMSE, train_MAE = r2_score(y_train0, train_set_predict),\
                    mean_squared_error(y_train0, train_set_predict)**0.5,\
                    mean_absolute_error(y_train0, train_set_predict)
    y_hat_b_all.insert(loc=y_hat_b_all.shape[1],
                   column='train_pred_'+im, value=train_set_predict)
    print('%.3f %.3f %.3f' % (train_R2, train_RMSE, train_MAE))
    
    # ###### for CI of MAE
    # for i, y_pred in enumerate(reg.staged_predict(X_test)):
    #     res_score_test[i+1,j,k] = mean_absolute_error(y_test, y_pred)
    # res_score_train[:reg.train_score_.shape[0],j,k] = -reg.train_score_
    
    # Network
    pdCV_best = pdCV.loc[pdCV['rank_test_score']==1,
                         ['mean_test_score','mean_train_score']]
    df_im_temp = {'IM': im,
              'max_leaf_nodes': reg.get_params()['max_leaf_nodes'],
              'min_samples_leaf': reg.get_params()['min_samples_leaf'],
              'n_iter': reg.n_iter_,
              'train_R2': train_R2,'train_RMSE': train_RMSE,'train_MAE': train_MAE,
              'cv_best_mean_test_score': pdCV_best.iloc[0,0],
              'cv_best_mean_train_score': pdCV_best.iloc[0,1],
              'test_R2': test_R2,'test_RMSE': test_RMSE,'test_MAE': test_MAE,
              }
    df_im_results = df_im_results.append(df_im_temp, ignore_index=True)
    
    endtime = time.time()
    print('=============================')
    print('Time in 1 CV for optimized parameters and hypterparameters: %.2f sec' %(endtime - starttime))
    
    # Save model
    if save_model: 
        dump(reg, main_folder+im+'/'+
                  im+'_LGBM_15k_'+str(random_state)+'.joblib')
    
    # Bootstrapping for Tα-model
    starttime = time.time()
    if num_bstp > 0:
        # Talpha-model:
        common_params.update(dict(max_iter=reg.n_iter_))  # fix HP: n_iter_
        for b in np.arange(num_bstp):
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Current training Tα-model of '+im+' on Db*: %s' %b)
            
            # Bootstrapping subset Db*
            train_set_B = pd.concat([X_train0, y_train0],axis=1).\
                sample(frac=1.0, axis=0, replace=True, random_state=b)
            X_train = train_set_B.loc[:, features]
            y_train = train_set_B.loc[:, im]
            
            # Train Tα-model on Db* with the optimized hyperparameters
            reg_b = HistGradientBoostingRegressor(**common_params).\
                fit(X_train,y_train)
            
            # Output result
            # on Test subset
            test_set_predict_b = reg_b.predict(X_test)
            X_test_out.insert(loc=X_test_out.shape[1],
                              column='b_'+str(b)+'_test_pred_'+im,
                              value=test_set_predict_b)
            
            # on Train0 subset
            train_set_predict = reg_b.predict(X_train0)
            y_hat_b_all.insert(loc=y_hat_b_all.shape[1],
                               column='b_'+str(b)+'_train_pred_'+im,
                               value=train_set_predict)
            
            # Save model
            if save_model:
                dump(reg_b, main_folder+im+'/'+
                          im+'_LGBM_15k_'+str(random_state)+
                          '_b_'+str(b)+'.joblib')
                    
    endtime = time.time()
    print('!=============================!')
    print('Time in B-models of '+im+': %.2f sec' %(endtime - starttime))
    
    # Output Bootstrapping result
    X_test_out.to_excel(main_folder+im+'/'+im+'bstp_test_15k_'+str(random_state)+'.xlsx')
    y_hat_b_all.to_excel(main_folder+im+'/'+'bstp_train_yb_15k_500.xlsx')

pd.DataFrame(es_test_score).to_excel(main_folder+'score_test_max_iter_15k_500.xlsx')
pd.DataFrame(es_train_score).to_excel(main_folder+'score_train_max_iter_15k_500.xlsx')
df_im_results.to_excel(main_folder+'summaryHGBR_15k_'+
              str(random_state)+'.xlsx')
end0time = time.time()
print('!!!=============================!!!')
print('Total time in Tα-model: %.2f sec' %(end0time - start0time))


# %% Permutation importance
from sklearn.inspection import permutation_importance

result = permutation_importance(
    reg, X, y, n_repeats=10, random_state=random_state, n_jobs=12
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(X.feature_names)[sorted_idx],
)
plt.title("Permutation Importance (test set)")


# %% Weaker model: Epistemic uncertainty of Tα-model
"""
Check the impact of model flexibility on sigma_E,A,T

Try a weaker model to attend to obtain less within-model variability:
    1) Re-train a weak model with less number of learner by 'n_estimators',
        in which the addtional optimization of 'early stopping' is shut down;
    2) Clipping the optimized Tα-model to weaken its capability,
        in which the trained model is reloaded and delected the finnal learners.
"""
from mlgmmuq import get_event_list
from mlgmmuq import get_residual_analysis

random_state = 500   # a specific state of 500 for a repeatable test
# This rs is used in
#   1) Original dataset D splitting into Dtr and Dte;
#   2) An additional splitting of Dtr for optimize early stopping;
num_bstp = 40  # generally 20~200, no significant change for B > 40
Tmodel = 'Tαmodels_stratify'
loss_func = 'absolute_error'

im_list = ['T0.300S']
period_list = [0.3]

early_stopping_list = [
                        # 1,2,  # for test
                        # 3,5,8,15,
                        # 20,30,35,40,50,75,
                        # 100,200,300,400,500,
                        # 1000,
                        2000,5000,
                        # 10000,
                        ]
early_stopping_opt_df = pd.read_excel('./bootstrapping/'+Tmodel+
                                      '/summaryHGBR_15k_500.xlsx')

start0time = time.time()

for im, period_value in zip(im_list, period_list):
    # Filter by LUF
    if period_value != 0:
        LUF_flag = 1/period_value > data15k[LUF]
    else:
        LUF_flag = np.ones((len(data15k),), dtype=bool)
        
    # Check the output fold and build it
    os.path.exists('./bootstrapping/'+im)
    if not os.path.exists('./bootstrapping/'+im):
        os.makedirs('./bootstrapping/'+im)
        
    # Remove N < 2
    data15k_LUF = data15k.loc[LUF_flag,:]
    df0 = data15k_LUF.set_index('EQID', drop=False)  # index to nature as origin data
    few_rc = data15k_LUF['EQID'].value_counts()
    few_rc = few_rc[few_rc < 2].index   # index list of N < 2
    df1 = df0.drop(few_rc)
    label_few = np.setdiff1d(df0['Record Sequence Number'].values,
                             df1['Record Sequence Number'].values)
    for i in label_few:
        data15k_LUF = data15k_LUF.loc[~(data15k_LUF['Record Sequence Number']==i)]
    
    # Data splitting
    X = data15k_LUF.loc[:, features]
    X['Region'] = X['Region'].astype('category')
    y = np.log(data15k_LUF.loc[:,im])
    
    X_train0, X_test, y_train0, y_test = train_test_split(X,y,test_size=0.2,
                                                        random_state=random_state,
                                                        shuffle=True,
                                                        stratify=data15k_LUF["EQID"])
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(im+': Current training the original Tα-model on Dtr')
    starttime = time.time()
    
    # Params initialized
    common_params = dict(
        loss=loss_func, l2_regularization=2,
        # max_iter=100,  # manually define early-stopping
        categorical_features=X_train0.columns.get_loc('Region'),
        max_depth=None, learning_rate=0.1,
        early_stopping=False,
        # validation_fraction=0.1, n_iter_no_change=10, tol=1e-7,
        random_state=random_state)
    
    # Load the saved cv result
    param = {'max_leaf_nodes': early_stopping_opt_df.loc[
        early_stopping_opt_df['IM']==im, 'max_leaf_nodes'].values[0],
             'min_samples_leaf': early_stopping_opt_df.loc[
        early_stopping_opt_df['IM']==im, 'min_samples_leaf'].values[0]}

    common_params.update(param)
    
    # load saved es opt value
    early_stopping_opt_value = early_stopping_opt_df.loc[early_stopping_opt_df['IM']==im,
                                                     'n_iter'].values[0]
    temp_es = early_stopping_list.copy()
    temp_es.append(early_stopping_opt_value)
    
    for ii, early_stopping_num in enumerate(temp_es):        
        # Save train0, X and y, to concat y_hat_b for all B
        y_hat_b_all = pd.concat([X_train0, y_train0], axis=1)
            
        # Manually set num_iterations by 'n_iter'
        common_params.update({'max_iter': early_stopping_num})
        reg = HistGradientBoostingRegressor(**common_params).fit(X_train0,y_train0)
        n_iter = reg.n_iter_  ### replace all 'reg.n_iter_'
        
        # Show the test result on Test and optimized hyperparameters on the Train0
        # Test result
        X_test_out = pd.concat([X_test, y_test], axis=1)
        test_set_predict = reg.predict(X_test)
        X_test_out.insert(loc=X_test_out.shape[1],
                          column='test_pred_'+im, value=test_set_predict)
        df_test_set = X_test_out.copy()
        
        R2, RMSE, MAE = r2_score(y_test, test_set_predict),\
                        mean_squared_error(y_test, test_set_predict)**0.5,\
                        mean_absolute_error(y_test, test_set_predict)
        test_result = {'test_R2':R2, 'test_RMSE':RMSE, 'test_MAE':MAE}
        
        # Train result
        train_set_predict = reg.predict(X_train0)
        y_hat_b_all.insert(loc=y_hat_b_all.shape[1],
               column='train_pred_'+im, value=train_set_predict)
        df_train_set = y_hat_b_all.copy()
        
        R2, RMSE, MAE = r2_score(y_train0, train_set_predict),\
                        mean_squared_error(y_train0, train_set_predict)**0.5,\
                        mean_absolute_error(y_train0, train_set_predict)
        train_result = {'train_R2':R2, 'train_RMSE':RMSE, 'train_MAE':MAE}
        
        endtime = time.time()
        print('=============================')
        print('Time in 1 CV for optimized parameters and hypterparameters: %.2f sec' %(endtime - starttime))
        
        # Bootstrapping for Tα-model
        for b in np.arange(num_bstp):
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('IM = %s, Early-stop model = %s' %(im,early_stopping_num))
            print('Current training Tα-model on Db*: %s' %b)
            starttime = time.time()
            
            # Bootstrapping subset Db*
            train_set_B = pd.concat([X_train0, y_train0],axis=1).\
                sample(frac=1.0, axis=0, replace=True, random_state=b)
            X_train = train_set_B.loc[:, features]
            y_train = train_set_B.loc[:, im]
            
            # Train Tα-model on Db* with the optimized hyperparameters
            reg_b = HistGradientBoostingRegressor(**common_params).\
                fit(X_train,y_train)
            
            # Output result
            # on Test subset
            test_set_predict_b = reg_b.predict(X_test)
            X_test_out.insert(loc=X_test_out.shape[1],
                              column='b_'+str(b)+'_test_pred_'+im,
                              value=test_set_predict_b)
            
            # on Train0 subset
            train_set_predict = reg_b.predict(X_train0)
            y_hat_b_all.insert(loc=y_hat_b_all.shape[1],
                               column='b_'+str(b)+'_train_pred_'+im,
                               value=train_set_predict)
            
            
        endtime = time.time()
        print('=============================')
        print('Time in Tα-model of '+im+': %.2f sec' %(endtime - starttime))
        
        
        # Output Bootstrapping result
        X_test_out.to_excel('./bootstrapping/'+im+'/es_'+\
                            str(early_stopping_num)+\
                            '_bstp_test_15k_'+str(random_state)+'.xlsx')
        y_hat_b_all.to_excel('./bootstrapping/'+im+'/es_'+\
                             str(early_stopping_num)+\
                             '_bstp_train_yb_15k_500.xlsx')

        # Show the result of an early-stopping setting
        print('=============================')
        print(reg.get_params())
        print('=============================')
        print('Test result: R2 = %.5f, RMSE = %.5f, MAE = %.5f' %\
              (test_result['test_R2'], test_result['test_RMSE'], test_result['test_MAE']))
        print('Train result: R2 = %.5f, RMSE = %.5f, MAE = %.5f' %\
              (train_result['train_R2'], train_result['train_RMSE'], train_result['train_MAE']))
        print('=============================')
        
        # event label
        event_train, event_test = get_event_list(
            data15k, 'Lowest Usable Freq - Ave. Component (Hz)',
            period_value, random_state=random_state, test_size=0.2,
            stratify_flag=True)
        
        # y_unbias
        y_hat_b_im_tr = y_hat_b_all.\
            iloc[:, y_hat_b_all.columns.get_loc(im) + 1:].values  # Bmodels pred.
        y_unbias_tr = np.mean(y_hat_b_im_tr[:,0:num_bstp], axis=1)  # Unbiased pred.
        y_hat_b_im_te = X_test_out.\
            iloc[:, X_test_out.columns.get_loc(im) + 1:].values  # Bmodels pred.
        y_unbias_te = np.mean(y_hat_b_im_te[:,0:num_bstp], axis=1)  # Unbiased pred.

        
        train_epsilon_im = get_residual_analysis(df_train_set, event_train, im,
                                           setFlag='train', y_unbias=y_unbias_tr)
        test_epsilon_im = get_residual_analysis(df_test_set, event_test, im,
                                           setFlag='test', y_unbias=y_unbias_te)
        
        # save the flexibility information
        flexibility_out = {'IM': im,'IM_value': period_value}
        flexibility_out.update(reg.get_params())
        flexibility_out.update(test_result)
        flexibility_out.update(train_result)
        flexibility_out.update(test_epsilon_im)
        flexibility_out.update(train_epsilon_im)
        
        flexibility_output_i = pd.DataFrame(flexibility_out, index=[0])
        if ii == 0:
            flexibility_output = flexibility_output_i
        else:
            flexibility_output = pd.concat([flexibility_output, flexibility_output_i], axis=0)
    
    # For an IM
    end0time = time.time()
    print(im+': Total time in Tα-model: %.2f sec' %(end0time - start0time))

    flexibility_output.to_excel('./bootstrapping/'+im+'/summary_flexibility.xlsx')

