import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.DataFrame.from_csv("data/train.csv")
D = df.shape[0]
plt.scatter(np.array(df['val_accs_49'], dtype=float), df['val_error'])
plt.scatter(np.array(df['val_accs_49'], dtype=float), df['train_error'])
plt.show()
y_val = np.array(df['val_error'])[:D//2]
y_tr = np.array(df['train_error'])[:D//2]
X = np.array(df[['val_accs_49', 'val_accs_48', 'val_accs_47']], dtype=float)[:D//2]
X = sk.preprocessing.scale(X)

regr_val = sk.linear_model.LinearRegression()
regr_val.fit(X, y_val)


poly = sk.preprocessing.PolynomialFeatures(degree=2)
X_quad = poly.fit_transform(X)
X_quad = sk.preprocessing.scale(X_quad)
# regr_tr = svm.SVR(kernel='linear')

regr_tr = sk.linear_model.LinearRegression()
regr_tr.fit(X_quad, y_tr)

# regr_tr = ensemble.GradientBoostingRegressor(n_estimators=20)
# regr_val = ensemble.GradientBoostingRegressor(n_estimators=20)


#### CV / split testing #####
y_test_tr = np.array(df['train_error'])[D//2:]
y_test_val = np.array(df['val_error'])[D//2:]
X_test = np.array(df[['val_accs_49', 'val_accs_48', 'val_accs_47']], dtype=float)[D//2:]
X_test_quad = poly.fit_transform(X_test)
X_test = sk.preprocessing.scale(X_test)
X_test_quad = sk.preprocessing.scale(X_test_quad)
pred_tr = regr_tr.predict(X_test_quad)
pred_val = regr_val.predict(X_test)
print('training err: R2 metric = ', sk.metrics.r2_score(y_test_tr, pred_tr))
print('validation err: R2 metric = ', sk.metrics.r2_score(y_test_val, pred_val))




# plt.scatter(np.array(df['epochs'], dtype=float), df['val_error'])
# print(df.columns[15:])

# pca = sk.decomposition.PCA()
# pca.fit_transform(X)
# print(X.mean(axis=0), X.std(axis=0))

# score = sk.metrics.r2_score(y_test_tr, pred)
# print(score)

#### validation #####

# df_test = pd.DataFrame.from_csv("data/test.csv")
# X_test = np.array(df_test[['number_parameters', 'epochs', 'val_accs_49']], dtype=float)
# X_test = sk.preprocessing.scale(X_test)
# pred_val = regr_val.predict(X_test)
# pred_tr = regr_tr.predict(X_test)

#### export data ####
# df_ex = pd.DataFrame({'id': ['test_'+str(i//2)+'_val_error' if i % 2 == 0 else 'test_'+str(i//2)+'_train_error'
#                              for i in range(df_test.shape[0]*2)],
#                       'Predicted':[pred_val[i//2] if i % 2 == 0 else pred_tr[i//2]
#                                    for i in range(df_test.shape[0]*2)]})
# print(df_ex)
# df_ex.to_csv('v0.csv', header=True, index=False)