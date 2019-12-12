# testing xgboost, submit GDBT w/o grid search
import sklearn
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn as sk
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from parser_1 import apply_flops, apply_ops_hist, apply_init_params, diff_avg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def test_to_csv(Regr_train_err, Regr_validation_err, Data_train_std, Data_validation_std, filename='test_result.csv'):
    pt = Regr_train_err.predict(Data_train_std)
    pv = Regr_validation_err.predict(Data_validation_std)
    df_test = pd.read_csv("data/test.csv")
    df_ex = pd.DataFrame(
        {'id': ['test_' + str(i // 2) + '_val_error' if i % 2 == 0 else 'test_' + str(i // 2) + '_train_error'
                for i in range(df_test.shape[0] * 2)],
         'Predicted': [pv[i // 2] if i % 2 == 0 else pt[i // 2]
                       for i in range(df_test.shape[0] * 2)]})
    df_ex.to_csv(filename, header=True, index=False)


df = pd.read_csv("data/train-1185.csv")
D = df.shape[0]
feature_all = list(df.columns)
# for i in range(len(feature_all)):
#     print(i, feature_all[i])
# flops = apply_flops(df)
# flops.to_csv('flops.csv', index=None)
flops = pd.read_csv('flops.csv')
y_val = df["val_error"]
y_tr = df["train_error"]

X = df[feature_all[17:] + ['number_parameters', 'epochs']].\
    join(flops).\
    join(apply_ops_hist(df)).\
    join(apply_init_params(df))\
    .\
    join(diff_avg(df[feature_all[17:67]], 'val_accs_diff')).\
    join(diff_avg(df[feature_all[67:117]], 'val_losses_diff')).\
    join(diff_avg(df[feature_all[117:167]], 'train_accs_diff')).\
    join(diff_avg(df[feature_all[167:217]], 'train_losses_diff'))  # train acc
pd.DataFrame(X).to_csv('data matrix.csv')
model_features = X.columns
X = preprocessing.scale(X)

X_train_tr, X_test_tr, y_train_tr, y_test_tr = model_selection.train_test_split(X, y_tr)
# regr_tr = sk.linear_model.LinearRegression()
# regr_tr.fit(X_train_tr, y_train_tr)
# y_pred_tr = regr_tr.predict(X_test_tr)
# print('Linear regression training err: R2 metric = ', sk.metrics.r2_score(y_test_tr, y_pred_tr))
parameter_candidates = {
    "subsample":[0.5, 0.55, 0.6, 0.65, 0.7]
}


# regr_tr = GridSearchCV(estimator=xgb.XGBRegressor(objective ='reg:logistic',missing=np.nan, max_depth=4, learning_rate=0.08, n_estimators=190, subsample=0.5),
#                     param_grid=parameter_candidates,
#                     cv=5,
#                     refit=True,
#                     error_score=0,
#                     n_jobs=-1)
# regr_tr = sk.linear_model.LinearRegression()
# regr_tr.fit(X_train_tr, y_train_tr)  # cross validation
# regr_tr = ensemble.GradientBoostingRegressor(loss="ls", max_depth=4, subsample=0.9, n_estimators=140)
regr_tr = xgb.XGBRegressor(objective ='reg:logistic',missing=np.nan, max_depth=4, learning_rate=0.08, n_estimators=190, subsample=0.5)

regr_tr.fit(X, y_tr)  # submission
y_pred_tr = regr_tr.predict(X_test_tr)
# print('the optimal params we found are: ', regr_tr.best_params_)
print('training err: R2 metric = ', sk.metrics.r2_score(y_test_tr, y_pred_tr))

#############
# Use same X: Yes
# X = df[feature_all[57:67] + ['number_parameters', 'epochs']].join(flops) # val acc
# X = preprocessing.scale(X)
# only 1 feature survived pca 95% var, acc decreases?
# print(X.shape)
# pca = decomposition.PCA(.95)
# X = pca.fit_transform(X)
# print(X.shape)

X_train_val, X_test_val, y_train_val, y_test_val = model_selection.train_test_split(X, y_val)
# regr_val = sk.linear_model.LinearRegression()
# regr_val.fit(X_train_val, y_train_val)
# y_pred_val = regr_val.predict(X_test_val)
# print('Linear regression benchmark validation err: R2 metric = ', sk.metrics.r2_score(y_test_val, y_pred_val))

###

# regr_val = GridSearchCV(estimator=xgb.XGBRegressor(objective ='reg:logistic',missing=np.nan, max_depth=4,learning_rate=0.08, n_estimators=200, subsample=0.7),
#                     param_grid=parameter_candidates,
#                     cv=5,
#                     refit=True,
#                     error_score=0,
#                     n_jobs=-1)
# regr_val = ensemble.GradientBoostingRegressor(loss="huber", max_depth=5, subsample=1.0)
regr_val = xgb.XGBRegressor(objective ='reg:logistic',missing=np.nan, max_depth=4,learning_rate=0.08, n_estimators=200, subsample=0.7)

# regr_val = sk.linear_model.LinearRegression()
# regr_val.fit(X_train_val, y_train_val)  # cross validation
regr_val.fit(X, y_val)  # submission
y_pred_val = regr_val.predict(X_test_val)
# print('the optimal params we found are: ', regr_val.best_params_)
print('validation err: R2 metric = ', sk.metrics.r2_score(y_test_val, y_pred_val))

##### CV
scores = cross_val_score(regr_tr, X, y_tr)
print("training Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(regr_val, X, y_val)
print("validation Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))








# # # ######
# df_t = pd.read_csv("data/test.csv")
# # for i in range(len(df_t.columns)):
# #     print(i, df_t.columns[i])
# flops_t = apply_flops(df_t)
# op_hist = apply_ops_hist(df_t)
# # X_ex_tr = preprocessing.scale(df_t[list(df_t.columns[153:163]) + ['number_parameters', 'epochs']].join(flops_t))
# X_ex = preprocessing.scale(df_t[feature_all[17:] + ['number_parameters', 'epochs']].
#                            join(flops_t).
#                            join(apply_ops_hist(df_t)).
#                            join(apply_init_params(df_t)).
#                            join(diff_avg(df_t[feature_all[17:67]], 'val_accs_diff')).\
#     join(diff_avg(df_t[feature_all[67:117]], 'val_losses_diff')).\
#     join(diff_avg(df_t[feature_all[117:167]], 'train_accs_diff')).\
#     join(diff_avg(df_t[feature_all[167:217]], 'train_losses_diff')))
# test_to_csv(regr_tr, regr_val, X_ex, X_ex, filename="v18_1.csv")


# indices = np.argsort(regr_tr.feature_importances_)[-15:]
# # plt.figure(figsize=[10, 15], dpi=200)
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), regr_tr.feature_importances_[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [model_features[i] for i in indices], fontsize=5)
# ### explore xgboost
# # xgb_train = xgb.DMatrix(X_train_val, label=y_train_val)
# # xgb_test = xgb.DMatrix(X_test_val, label=y_test_val)
# # param = {'max_depth': 5, 'silent': 1, 'objective': 'reg:squarederror'}
# # evallist = [(xgb_test, 'eval'), (xgb_train, 'train')]
# # num_round = 100
# # bst = xgb.train(param, xgb_train, num_round, evallist)
# # y_pred_val = bst.predict(xgb_test)
# # print('validation err: R2 metric = ', sk.metrics.r2_score(y_test_val, y_pred_val))
# # fig, ax = plt.subplots(figsize=(10, 15))
# # plot_importance(bst, height=0.5, max_num_features=64, ax=ax)
# plt.savefig('tuned_model_tr_imp.png', dpi=300)
# # plt.show()

