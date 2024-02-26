## 1. Gradient Boosting with LOWESS



```python
import numpy as np
import pandas as pd
import xgboost
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy import linalg
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
```


```python
# Gaussian Kernel
def Gaussian(x):
  return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))
```


```python
def weight_function(u,v,kern=Gaussian,tau=0.5):
    return kern(cdist(u, v, metric='euclidean')/(2*tau))
```


```python
class Lowess:
    def __init__(self, kernel = Gaussian, tau=0.05):
        self.kernel = kernel
        self.tau = tau

    def fit(self, x, y):
        kernel = self.kernel
        tau = self.tau
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        lm = linear_model.Ridge(alpha=0.001)
        w = weight_function(x,x_new,self.kernel,self.tau)

        if np.isscalar(x_new):
          lm.fit(np.diag(w)@(x.reshape(-1,1)),np.diag(w)@(y.reshape(-1,1)))
          yest = lm.predict([[x_new]])[0][0]
        else:
          n = len(x_new)
          yest_test = []
          for i in range(n):
            lm.fit(np.diag(w[:,i])@x,np.diag(w[:,i])@y)
            yest_test.append(lm.predict([x_new[i]]))
        return np.array(yest_test).flatten()
```

Work in progress...


```python
class GBLowess:

  def __init__(self, lr, n_estimators):
    self.lr = lr
    self.n_estimators = n_estimators
    self.models = []

  def fit(self, xtrain, ytrain):
    self.pred0 = y.mean()
    pred = self.pred0
    for i in range(self.n_estimators):
      res = ytrain - pred
      model = Lowess()
      model.fit(xtrain, res)
      y0 = model.predict(xtrain)
      pred += self.lr * y0
      self.models.append(model)

  def predict(self, xtest):
    pred = self.pred0
    for i in range(self.n_estimators):
      pred += self.lr * self.models[i].predict(xtest)

      return pred
```


```python
data = pd.read_csv('/content/drive/MyDrive/AML/concrete.csv')
X = data.loc[:,'cement':'age'].values
y = data['strength'].values
xtrain, xtest, ytrain, ytest = tts(X, y, test_size=0.3,shuffle=True, random_state=123)
```


```python
gb_lowess = GBLowess(lr=0.1,n_estimators=100)
gb_lowess.fit(xtrain,ytrain)
mse(gb_lowess.predict(xtest),ytest)
```




    258.69897412387536



### some hyperparameter summary:
n_estimators: Increasing the number of boosting steps can improves the model's ability to fit the data, potentially lowering the MSE up to a point.

tau: The tau parameter controls the bandwidth for the LOWESS smoothing. Potentially leading to a higher MSE, if the model does not follow the data well enough.

learning_rate: A smaller learning rate means the model makes slower progress (needs higher n_estimator), but it can potentially achieve a smaller mse that prevents overfitting.

### Scaling
it appears that scaling increased the mse value, most likely a function error considering that it should decrease the mse value.


```python
scale1 = MinMaxScaler()
xtrain1 = scale1.fit_transform(xtrain)
xtest1 = scale1.transform(xtest)
gb_lowess.fit(xtrain1,ytrain)
yhat1 = gb_lowess.predict(xtest1)
(mse(ytest,yhat1))
```




    263.99866035011985




```python
scale2 = StandardScaler()
xtrain2 = scale2.fit_transform(xtrain)
xtest2 = scale2.transform(xtest)
gb_lowess.fit(xtrain2,ytrain)
yhat2 = gb_lowess.predict(xtest2)
(mse(ytest,yhat2))
```




    262.62985206806707




```python
scale3 = QuantileTransformer()
xtrain3 = scale3.fit_transform(xtrain)
xtest3 = scale3.transform(xtest)
gb_lowess.fit(xtrain3,ytrain)
yhat3 = gb_lowess.predict(xtest3)
(mse(ytest,yhat3))
```

    /usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_data.py:2627: UserWarning: n_quantiles (1000) is greater than the total number of samples (927). n_quantiles is set to n_samples.
      warnings.warn(





    262.62985206806707



the result from gradient boosting with lowess is definitely much higher in comparison to the xgboost package. could be a potential functionality issue from the class.


```python
model_xgboost = xgboost.XGBRFRegressor(n_estimators=100,max_depth=7)
model_xgboost.fit(xtrain,ytrain)
mse(model_xgboost.predict(xtest),ytest)
```




    37.643205071403095



attemp to run kfold, but cell times out


```python
scale = MinMaxScaler()
mse_gblwr = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_gblwr = CustomGradientBoostingRegressor(learning_rate=0.1,n_estimators=100)

for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain].ravel()
  ytest = y[idxtest].ravel()
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model_gblwr.fit(xtrain,ytrain)
  yhat_gblwr = model_gblwr.predict(xtest)

  mse_gblwr.append(mse(ytest,yhat_gblwr))
```

## 2. KNN/usearch library

### Review KNN from scratch without usearch


```python
class knn_for_regression:
  def __init__(self,k_neighbor,xtrain,ytrain):
    self.k = k_neighbor
    self.xtrain = xtrain
    self.ytrain = ytrain
  def euclidean_dist(self,pt1,pt2):
    dist = np.sqrt(np.sum((pt1-pt2)**2))
    return dist
  def predict(self, xtest):
    preds = []
    for val in xtest:
        dists = [self.euclidean_dist(val,x) for x in self.xtrain]
        ind = np.argsort(dists)[:self.k]
        labels = [self.ytrain[ind] for i in ind]
        pred = np.mean(labels)
        preds.append(pred)
    return np.array(preds)
```


```python
knn0 = knn_for_regression(2,xtrain,ytrain)
knn0.predict(xtrain)
mse(knn0.predict(xtest),ytest)
```




    72.48015744336568



### Compare with sklearn


```python
from sklearn.neighbors import KNeighborsRegressor
sk_model = KNeighborsRegressor(n_neighbors=2)
sk_model.fit(xtrain,ytrain)
sk_model.predict(xtest)
mse(sk_model.predict(xtest),ytest)
```




    72.51706391585759



### KNN with usearch


```python
pip install usearch
```

    Collecting usearch
      Downloading usearch-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl (2.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m21.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from usearch) (1.25.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from usearch) (4.66.2)
    Installing collected packages: usearch
    Successfully installed usearch-2.9.0



```python
from usearch.index import search, MetricKind, Matches, BatchMatches
```


```python
class knn_for_regression_usearch:
  def __init__(self, k, metric, xtrain, ytrain):
    self.k = k
    self.metric = metric
    self.xtrain = xtrain
    self.ytrain = ytrain

  def predict(self, xtest):
    preds = []
    for val in xtest:
      u_search = search(self.xtrain,val,self.k,self.metric,exact=True)
      output = u_search.to_list()
      ind = np.array(output)[:,0].astype('int64')
      lbls = [ytrain[i] for i in ind]
      pred = np.mean(lbls)
      preds.append(pred)
    return np.array(preds)

```


```python
knn1 = knn_for_regression_usearch(7,MetricKind.L2sq,xtrain,ytrain)
knn1.predict(xtrain)
mse(knn1.predict(xtest),ytest)
```




    73.92345972524933



### Compare with sklearn


```python
sk_model1 = KNeighborsRegressor(n_neighbors=7)
sk_model1.fit(xtrain,ytrain)
sk_model1.predict(xtest)
mse(sk_model1.predict(xtest),ytest)
```




    73.92345972524933



mse values were similar with both knn without usearch and with usearch when compared with sklearn
