# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:10:24 2020

@author: benja

Find competition and data under the following link:
    https://www.kaggle.com/c/house-prices-advanced-regression-techniques
"""


# Load all libraries
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from scipy.stats import norm   #for some statistics
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV

# =============================================================================
# Load Data
# =============================================================================
def get_data():
    #get train data
    train_data_path ='data/train.csv'
    train = pd.read_csv(train_data_path)
    #get test data
    test_data_path ='data/test.csv'
    test = pd.read_csv(test_data_path)    
    return train , test


def get_combined_data():
  #reading train data
  train , test = get_data()
  target = train.SalePrice # define the variable to predict 
  train.drop(['SalePrice'],axis = 1 , inplace = True)
  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop(['index', 'Id'], inplace=True, axis=1)
  return combined, target

#Load train and test data into pandas DataFrames
train_data, test_data = get_data()

#Combine train and test data to process them together in order check their
# statistical properties over the entire set
combined, target = get_combined_data()


# =============================================================================
# Missing values
# =============================================================================
# Handle categorial variables manually

combined.loc[:, "MasVnrType"]       = combined.loc[:, "MasVnrType"].fillna("None")
combined.loc[:, "GarageQual"]       = combined.loc[:, "GarageQual"].fillna("No")
combined.loc[:, "GarageCond"]       = combined.loc[:, "GarageCond"].fillna("No")
combined.loc[:, "Utilities"]        = combined.loc[:, "Utilities"].fillna("AllPub")
combined.loc[:, "SaleCondition"]    = combined.loc[:, "SaleCondition"].fillna("Normal")
combined.loc[:, "KitchenQual"]      = combined.loc[:, "KitchenQual"].fillna("TA")
combined.loc[:, "BsmtExposure"]     = combined.loc[:, "BsmtExposure"].fillna("No")
combined.BsmtFinSF1.astype('str')
combined.loc[:, 'MSZoning']         = combined.loc[:, 'MSZoning'].fillna('RH')
combined.loc[:, 'SaleType']         = combined.loc[:, 'SaleType'].fillna('Oth')
combined.loc[:, 'Exterior1st']      = combined.loc[:, 'Exterior1st'].fillna('Other')


# Fill the NaNs of metrical variables with zeros
for col in ('LotFrontage', 'MasVnrArea', 'OpenPorchSF','GarageArea',
            'GarageCars', 'HalfBath', 'WoodDeckSF', 'ScreenPorch','KitchenAbvGr',
            'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', "BsmtFinSF2"):
    combined[col] = combined[col].fillna(0)
del col   


# Assort according to num and cat
def assort_cols(df):
    num_cols = df.select_dtypes(exclude=['object']).columns
    num_cols = num_cols.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = cat_cols.tolist()
    df1 = df[num_cols + cat_cols]
    return num_cols, cat_cols, df1
[num_cols, cat_cols, combined] = assort_cols(combined)


# identify the number of observations in the dependent data set
training_data_len       = len(target)


# =============================================================================
# Group some categories of some variables
# =============================================================================
# If there are too many categories, group some and ecode new
combined = combined.replace({'SaleCondition' : {'AdjLand' : 1, 'Abnorml' : 2, 'Family' :2,
                                                'Alloca' :3, 'Normal':3, 'Partial': 4}})

combined = combined.replace({'MSZoning' : {'C (all)' : 1, 'RM' : 2, 'RH' :2,
                                                'RL' :3, 'FV':3}})
# encode 'Neighborhood' as ordinale scaled var (I choose 4 categories)
combined = combined.replace({'Neighborhood' : {'MeadowV' : 1, 'BrDale' : 1, 'IDOTRR' : 1,
                                                'BrkSide' : 1, 'Edwards' : 1, 'OldTown' : 1, 'Sawyer' : 1,
                                                'Blueste' : 1, 'SWISU' : 1, 'NPkVill' : 1, 'NAmes' : 1, 
                                                'Mitchel' : 2, 'SawyerW' : 2, 'NWAmes' : 2, 'Gilbert' : 2,
                                                'Blmngtn' : 2, 'CollgCr' : 2, 'Crawfor' : 3, 'ClearCr' : 3,
                                                'Somerst' : 3, 'Veenker' : 3, 'Timber' : 3, 'StoneBr': 4, 'NridgHt':4,
                                                'NoRidge' : 4}})

# =============================================================================
# Age of the building
# =============================================================================

combined['Age'] = combined['YearBuilt'].max()+1 - combined['YearBuilt']
combined['Years_sold'] = combined['YrSold'].max()+1 - combined['YearBuilt']
combined['Years_since_remod'] = combined['YrSold'] - combined['YearRemodAdd']
combined.drop(['YearBuilt', 'YearRemodAdd','YrSold'], inplace=True, axis = 1)

# =============================================================================
# Remove obvious Outliers
# =============================================================================
combined = combined[combined.GrLivArea < 4000]

# =============================================================================
# Drop some vars (which do not show a clear explanatory power)
# =============================================================================
combined.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtFullBath',
               'BsmtHalfBath', 'GarageYrBlt', 'GarageType', 'GarageFinish', 'BsmtQual',
               'BsmtFinType1', 'BsmtFinType2','BsmtCond', 'MoSold'],
              inplace=True, axis=1)


# =============================================================================
# Delete some rows (with single missing values)
# =============================================================================
def where_nan_index(df, col_nam): 
    x = df[col_nam].isnull().values
    x = np.where(x==True)
    x = np.array(x)
    x = x.T
    return x

El_nan          = where_nan_index(combined, 'Electrical')
Functional      = where_nan_index(combined, 'Functional')

combined['Target']  = target
combined            = combined.drop(combined.index[El_nan])
combined            = combined.drop(combined.index[Functional])
combined.reset_index(inplace=True)
combined.drop(['index'], inplace = True, axis=1)



t_nan               = np.array(np.where(combined['Target'].isnull().values ==True)).T
target              = combined['Target'].drop(combined['Target'].index[t_nan])
combined.drop(['Target'], inplace = True, axis=1)

training_data_len   = target.shape[0]

del El_nan, Functional, t_nan


# =============================================================================
# Combine Variables (in a suitable manner)
# =============================================================================
combined['Porch']       = combined['OpenPorchSF'] + combined['EnclosedPorch']+ combined['3SsnPorch']+combined['ScreenPorch']
combined.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Street'], inplace = True, axis=1)

combined['BathroomNo']  = combined['FullBath']+combined['HalfBath']

combined.drop(['BsmtFinSF1','BsmtFinSF2', 'LowQualFinSF', 'PoolArea',
               'FullBath', 'HalfBath'], inplace = True, axis=1)

# Assort again
[num_cols, cat_cols, combined] = assort_cols(combined)

# =============================================================================
# One-Hot-Encoder
# =============================================================================
combined_cat        = pd.get_dummies(combined[cat_cols])
combined_cat_col    = combined_cat.columns

combined_num        = combined[num_cols]
combined_num_col    = combined_num.columns

combined_hot        = pd.concat([combined_num, combined_cat], axis = 1)
comb_hot_columns    = combined_hot.columns


# =============================================================================
# Show the Distribution of the SalesPrice Raw Data (y)
# =============================================================================

sb.distplot(target , fit = norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(target)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('(Log) SalePrice distribution')
plt.savefig('figures/LogSalesPrice_Dist.eps')
plt.show()

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(target, plot=plt)
plt.title('Q-Q Plot (log(y))')
plt.savefig('figures/QQPlot_logy.eps')
plt.show()



# Take logs of the dependent variable (due to skewness)
target = np.log(target)



# =============================================================================
# Plot log-SalesPrice Distribution
# =============================================================================
sb.distplot(target , fit = norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(target)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('(Log) SalePrice distribution')
plt.savefig('figures/LogSalesPrice_Dist.eps')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(target, plot=plt)
plt.title('Q-Q Plot (log(y))')
plt.savefig('figures/QQPlot_logy.eps')
plt.show()


# =============================================================================
# Pre-Processing Standardization
# =============================================================================

scaler_x_num            = StandardScaler()
X_num                                          = combined_hot.values
X_num[:,:np.array(combined_num_col).shape[0]]  = scaler_x_num.fit_transform(X_num[:,:np.array(combined_num_col).shape[0]])

# =============================================================================
# PCA 
# =============================================================================
pca                 = PCA(svd_solver='full') # n_components = None, random_state = 1

pca.fit_transform(X_num)

pca_ex_var_ratio    = pca.explained_variance_ratio_
pca_components      = pca.components_

# =============================================================================
# Identify the loading
# =============================================================================

# Plot the cumulative explained variance
pca_CumExpVar = np.cumsum(pca_ex_var_ratio)

plt.plot(range(1,len(pca_ex_var_ratio)+1), pca_CumExpVar, label = 'CumExpVar')
plt.hlines(y=0.95, xmin = 1, xmax = len(pca_ex_var_ratio)+1, 
           lw=1.0, color = 'red',linestyle='dashed', label = '95% CI')
plt.xlim([1,len(pca_ex_var_ratio)+1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('figures/CumExpVar.eps')
plt.savefig('figures/CumExpVar.png')
plt.legend(loc = 'lower right')
plt.show()


# Identify how much features explain 95% of the variance
# Choose number of components (as index)
topX = len(pca_CumExpVar[pca_CumExpVar<= 0.95])-1
pca_loading_scores          = pd.Series(pca_components[0,:])
pca_sorted_loading_scores   = pca_loading_scores.abs().sort_values(ascending=False)

pca_topX_index              = pca_sorted_loading_scores[0:topX].index

# Identify PCs
combined_hot_cols           = combined_hot.columns
pca_topX_cols               = np.array(combined_hot_cols[pca_topX_index]).reshape(-1,1)

# Create Panda Series containing the top loadings and feature names
pca_topX                    = pd.Series(np.array(pca_sorted_loading_scores[0:topX]).reshape(-1,1)[:,0], 
                                        index = pca_topX_cols[:,0])


labels = ['PC' + str(i) for i in range(1,len(pca_topX_index)+1)]
plt.gcf().subplots_adjust(bottom=0.4)
plt.bar(x = range(0,len(pca_topX_index)) , height = pca_ex_var_ratio[0:topX]*100,
        tick_label = labels)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal Components')
plt.xticks(rotation=90, fontsize=7)
plt.title('Scree Plot')
plt.savefig('figures/ScreePlot.eps')
plt.savefig('figures/ScreePlot.png')
plt.tight_layout()
plt.show()

# Keep the house clean
del pca_loading_scores, pca_sorted_loading_scores, labels, pca_topX_index, pca_topX_cols

topX = len(pca_CumExpVar[pca_CumExpVar<= 0.95])-1


# =============================================================================
# Step 2: Ridge Regression on (topX) selected principal components
# =============================================================================

scaler_y = StandardScaler() 
scaler_x = StandardScaler() 
X        = combined_hot.values

X[:,:np.array(combined_num_col).shape[0]]  = scaler_x.fit_transform(X[:,:np.array(combined_num_col).shape[0]])

X_std   = X[:training_data_len,:]
target  = np.array(target).reshape(-1,1)
y_std   = scaler_y.fit_transform(target)

# Split standardized '(test)Data' into test and trainings data
tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X_std):
    X_train, X_test = X_std[train_index,:], X_std[test_index,:]
    y_train, y_test = y_std[train_index], y_std[test_index]



# =============================================================================
# Learning Curve - PCA Selection
# =============================================================================
from sklearn.metrics import mean_squared_error
num_pca = [np.arange(1,topX)][0]

lc_score_train  = []
lc_score_test   = []
mse_train       = []
mse_test        = []

for i in range(1,topX):
    pca_topX        = PCA(n_components = i, random_state = 1) 
    X_train_pca     = pca_topX.fit_transform(X_train)
    X_test_pca      = pca_topX.fit_transform(X_test)
    
    lc              = LinearRegression()
    lc.fit(X_train_pca, y_train)
    lc_pred_test         = lc.predict(X_test_pca)
    lc_pred_train        = lc.predict(X_train_pca)
    lc_score_train.append(lc.score(X_train_pca, y_train))
    lc_score_test.append(lc.score(X_test_pca, y_test))
    mse_train.append(mean_squared_error(y_train, lc_pred_train))
    mse_test.append(mean_squared_error(y_test,lc_pred_test))

mse_train   = np.array(mse_train)
mse_test    = np.array(mse_test)

lc_score_train  = np.array(lc_score_train)
lc_score_test   = np.array(lc_score_test)

# Set linewidth
lw = 2

# Plot the Learning Curve
plt.subplot(211)
plt.title('Learning Curve(s)')
plt.ylabel('MSE')
plt.plot(num_pca, mse_train, label="Training score",
              color="darkorange", lw=lw)
plt.plot(num_pca, mse_test, label="Test score",
              color="navy", lw=lw)
plt.xlabel('Number of PCs')
plt.legend(loc="best")
plt.subplot(212)
plt.ylabel('$R^2$')
plt.plot(num_pca, lc_score_train, label="Training score",
              color="darkorange", lw=lw)

plt.plot(num_pca, lc_score_test, label="Test score",
              color="navy", lw=lw)
plt.xlabel('Number of PCs')
plt.savefig('figures/PCA_selection_LearnCurve.eps')
plt.show()

# Select the number of PCA that minimizes the margin of the MSE of train-test-set
pca_sel = np.argmin(np.abs(mse_train - mse_test))+1
# +1 due to zero-base counting


# =============================================================================
# Choose the number of principle components 
# =============================================================================
# Choose PC's that explain at least 5% of the variance
# Otherwise, the model will strongly overfit the data
# pca_sel = len(pca_ex_var_ratio[pca_ex_var_ratio>= 0.05])

   
# Reduce the dimension using PCA
pca_topX        = PCA(n_components = pca_sel, random_state = 1) 
X_train_pca     = pca_topX.fit_transform(X_train)
X_test_pca      = pca_topX.fit_transform(X_test)

# =============================================================================
# Linear Regression
# =============================================================================
lr = LinearRegression()

lr.fit(X_train_pca, y_train)

y_pred_lr_test         = lr.predict(X_test_pca)
y_pred_lr_train        = lr.predict(X_train_pca)

lr_score_train         = lr.score(X_train_pca, y_train)
lr_score_test          = lr.score(X_test_pca, y_test)
# =============================================================================
# Ridge Regression with PCA
# =============================================================================

ridgereg = RidgeCV(alphas=[1e-3,1e-2/2, 1e-2,1e-1/2, 1e-1, 1/2, 1, 5, 10], store_cv_values= True)
ridgereg.fit(X_train_pca, y_train)



# Predictions
y_pred_ridgereg_test         = ridgereg.predict(X_test_pca)
y_pred_ridgereg_train        = ridgereg.predict(X_train_pca)

# Check the fit
ridgereg_score_train         = ridgereg.score(X_train_pca, y_train)
ridgereg_score_test          = ridgereg.score(X_test_pca, y_test)
# --> overfitting

ridgereg_resid_test          = y_test - y_pred_ridgereg_test
ridgereg_resid_train         = y_train - y_pred_ridgereg_train

ridgereg_cv_values           = ridgereg.cv_values_ 
# Illustrate
fig, ax = plt.subplots()
ax.scatter(y_pred_ridgereg_train, ridgereg_resid_train,
            c = 'red',
            edgecolor = 'black',
            marker = 'o', label = 'Train Data')
ax.scatter(y_pred_ridgereg_test, ridgereg_resid_test,
            c = 'black',
            edgecolor = 'white',
            marker = 'o', label = 'Test Data')
ax.hlines(y=0, xmin = -10, xmax = 10, lw=1.0, color = 'black', linestyle = 'dashed')
ax.legend(loc= 'upper left')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title('Residual Plot (PC-Ridge Regression)')
fig.savefig('figures/ReisdualPlot_PC_Ridge.eps',  
            bbox_inches='tight', format='eps')

# Evaluate using RMSE
rmse_ridge_pca = np.sqrt(np.mean(y_pred_ridgereg_test  - y_test)**2)


print('Score of Test Set:', ridgereg_score_test)
print('Score of Train Set:',ridgereg_score_train)
print('RMSE of Ridge Regression:',rmse_ridge_pca)


# =============================================================================
# Elastic Net Regression
# =============================================================================
en = ElasticNetCV(cv=10, random_state=1)
en.fit(X_train_pca, y_train)


# Predictions
y_pred_en_test         = en.predict(X_test_pca).reshape(-1,1)
y_pred_en_train        = en.predict(X_train_pca).reshape(-1,1)

# Check the fit
en_score_train         = en.score(X_train_pca, y_train)
en_score_test          = en.score(X_test_pca, y_test)

en_resid_test          = y_test - y_pred_en_test
en_resid_train         = y_train - y_pred_en_train

en_l1_ratio           = en.l1_ratio_
# Illustrate
fig, ax = plt.subplots()
ax.scatter(y_pred_en_train, en_resid_train,
            c = 'red',
            edgecolor = 'black',
            marker = 'o', label = 'Train Data')
ax.scatter(y_pred_en_test, en_resid_test,
            c = 'black',
            edgecolor = 'white',
            marker = 'o', label = 'Test Data')
ax.hlines(y=0, xmin = -10, xmax = 10, lw=1.0, color = 'black', linestyle = 'dashed')
ax.legend(loc= 'upper left')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title('Residual Plot (PC-ElasticNet Regression)')
fig.savefig('figures/ReisdualPlot_PC_ENet.eps',  
            bbox_inches='tight', format='eps')

# Evaluate using RMSE
rmse_en_pca = np.sqrt(np.mean(y_pred_ridgereg_test  - y_test)**2)


print('Score of Test Set:', en_score_test)
print('Score of Train Set:',en_score_train)
print('RMSE of ENet Regression:',rmse_ridge_pca)

