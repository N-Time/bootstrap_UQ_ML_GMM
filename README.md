# bootstrap_UQ_ML_GMM

A Bootstrap-based uncertainty quantification for machine-learning-based ground motion model.

Ground motion models (GMM) incorporating proper epistemic uncertainty quantification are fundamental for seismic hazard analysis, structural seismic design, and earthquake risk management. However, this critical uncertainty has not been systematically addressed in recently prevalent machine learning (ML)-based GMMs. To rectify this gap, a straightforward and flexible framework based on bootstrapping is proposed for generic ML algorithms. Specifically, a group of datasets are generated through bootstrap resampling, each employed to develop an individual GMM. In the inference phase, the epistemic uncertainty is represented by the prediction samples from this ensemble of models. In case study, the Light Gradient Boosting Machine (LGBM), is applied to establish a GMM on the NGA-West2 database. Overfitting is mitigated by early-stopping, and its implications on uncertainty components and random-effect variability are thoroughly examined. Compared to artificial neural network (ANN) in literature, our work demonstrates a decrease in the standard deviation of inter- and intra-event variability by an average of 0.051 and 0.140, respectively. Moreover, our model adeptly captures the heteroscedasticity of epistemic uncertainty, particularly over different regions and within the range of the metadata with limited data. The significance of epistemic uncertainty is highlighted by its potential to diminish the statistical significance of seismic scaling effects at short-period pseudo spectral acceleration (PSA) for great events and at rock sites.

# Framework

<img width="695" alt="Boostrap_UQ" src="https://github.com/user-attachments/assets/441e90db-e0a5-40ec-a00a-4d869fb1b9dc">

# Prediction

![Case_M-R_PGA_T0 3](https://github.com/user-attachments/assets/0110c1f2-31bd-46be-b18f-6b98334040df)
![Case_M-R_T1 0_T3 0](https://github.com/user-attachments/assets/7ba9a97e-04e7-4053-bd91-061fc1ae1b16)
![Case_Vs30_Rx30_T](https://github.com/user-attachments/assets/7159045b-66af-45c8-a09d-365e0c712435)

# How to use

Run the following scripts in Spyder.

1) Data preparation: 'ngaw2_15k.py', 'assignRegionLabel.py';
2) Model training: 'LGBMbootstrap2.py';
3) Parameter uncertainty quantification: 'EUanalysis2.py', 'mlgmmuq.py';
4) Output the results in the paper: 'ML_GMM_UQ2.py'.

# equired Packages

numpy, pandas, geopy, scipy, matplotlib, joblib, sklearn, seaborn

# Reference

[1] Wen T, He J, Jiang L, Du Y, Jiang L. A simple and flexible bootstrap-based framework to quantify epistemic uncertainty of ground motion models by light gradient boosting machine. Applied Soft Computing 2023:111195. https://doi.org/10.1016/j.asoc.2023.111195.
