```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import toeplitz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,r2_score as r2

```


```python
device = torch.device('cpu')
dtype = torch.double
```

# 1. SCAD class

## SCAD with Pytorch
#### https://andrewcharlesjones.github.io/journal/scad.html

introduce scad:


```python
def scad_penalty(beta_hat, lambda_val, a_val):
  is_linear = (torch.abs(beta_hat) <= lambda_val)
  is_quadratic = torch.logical_and(lambda_val < torch.abs(beta_hat), torch.abs(beta_hat) <= a_val * lambda_val)
  is_constant = (a_val * lambda_val) < torch.abs(beta_hat)

  linear_part = lambda_val * torch.abs(beta_hat) * is_linear
  quadratic_part = (2 * a_val * lambda_val * torch.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
  constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
  return linear_part + quadratic_part + constant_part

def scad_derivative(beta_hat, lambda_val, a_val):
  return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))

```

## SCAD model for regression



```python
class scad_regression(nn.Module):

  def __init__(self, input_size, alpha=3, lambda_val=0.2):
    super(scad_regression, self).__init__()
    self.input_size = input_size
    self.alpha = alpha
    self.lambda_val = lambda_val
    self.linear = nn.Linear(input_size,1,device=device,dtype=dtype,bias=False)

  def forward(self, x):
    return self.linear(x)

  def loss(self, y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    weights = self.linear.weight
    is_linear = (torch.abs(weights) <= self.lambda_val)
    is_quadratic = torch.logical_and(self.lambda_val < torch.abs(weights), torch.abs(weights) <= self.alpha * self.lambda_val)
    is_constant = (self.alpha * self.lambda_val) < torch.abs(weights)
    linear_part = (self.lambda_val * torch.abs(weights) * is_linear).sum()
    quadratic_part = ((2 * self.alpha * self.lambda_val * torch.abs(weights) - weights**2 - self.lambda_val**2) / (2 * (self.alpha - 1)) * is_quadratic).sum()
    constant_part = ((self.lambda_val**2 * (self.alpha + 1)) / 2 * is_constant).sum()
    return linear_part + quadratic_part + constant_part + mse_loss

  def fit(self, X, y, num_epochs=200, learning_rate=0.001):
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
      self.train()
      optimizer.zero_grad()
      y_pred = self(X)
      loss = self.loss(y_pred.flatten(), y.flatten())
      loss.backward()
      optimizer.step()
      if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

  def predict(self, X):
    self.eval()
    with torch.no_grad():
      y_pred = self(X)
    return y_pred

  def get_coefficients(self):
    return self.linear.weight
```


```python
data = pd.read_csv('/content/drive/MyDrive/AML/concrete.csv')
x = data.drop(columns=['strength']).values
y = data["strength"].values
# quantile transformer
x = scaler.fit_transform(x)
x_torch = torch.tensor(x, device=device)
y_torch = torch.tensor(y, device=device)
```


```python
scad0 = scad_regression(input_size=x_torch.shape[1], alpha=3, lambda_val=0.2)
scad0.fit(x_torch, y_torch, num_epochs=15000, learning_rate=0.03)
```

    Epoch [1000/15000], Loss: 149.96982318544428
    Epoch [2000/15000], Loss: 84.71348758368362
    Epoch [3000/15000], Loss: 61.49926305578025
    Epoch [4000/15000], Loss: 55.02279794023234
    Epoch [5000/15000], Loss: 53.45368317899484
    Epoch [6000/15000], Loss: 53.011802369440936
    Epoch [7000/15000], Loss: 52.97588203800688
    Epoch [8000/15000], Loss: 52.97537444345149
    Epoch [9000/15000], Loss: 52.97537404492489
    Epoch [10000/15000], Loss: 52.97537404492266
    Epoch [11000/15000], Loss: 52.975374044922646
    Epoch [12000/15000], Loss: 52.975374044922646
    Epoch [13000/15000], Loss: 52.97537404492265
    Epoch [14000/15000], Loss: 52.97537404492265
    Epoch [15000/15000], Loss: 52.97537404492265



```python
X_train,X_test,y_train,y_test = train_test_split(x_torch,y_torch,test_size=0.2,
                                                 random_state=441)
mse(scad0.predict(X_test),y_test)
```




    45.96884127036795




```python
scad0.get_coefficients()
```




    Parameter containing:
    tensor([[ 33.0397,  14.9935,   3.8154, -14.8133,   5.0444,   0.8927,  -2.6222,
              36.1784]], dtype=torch.float64, requires_grad=True)



### taking a look at the dataframe, the coefficients with the heaviest weights are cement, slag, and age.


```python
data
```





  <div id="df-20bbb4a6-db8c-4a83-bc1b-6d031663c768" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cement</th>
      <th>slag</th>
      <th>ash</th>
      <th>water</th>
      <th>superplastic</th>
      <th>coarseagg</th>
      <th>fineagg</th>
      <th>age</th>
      <th>strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>276.4</td>
      <td>116.0</td>
      <td>90.3</td>
      <td>179.6</td>
      <td>8.9</td>
      <td>870.1</td>
      <td>768.3</td>
      <td>28</td>
      <td>44.28</td>
    </tr>
    <tr>
      <th>1026</th>
      <td>322.2</td>
      <td>0.0</td>
      <td>115.6</td>
      <td>196.0</td>
      <td>10.4</td>
      <td>817.9</td>
      <td>813.4</td>
      <td>28</td>
      <td>31.18</td>
    </tr>
    <tr>
      <th>1027</th>
      <td>148.5</td>
      <td>139.4</td>
      <td>108.6</td>
      <td>192.7</td>
      <td>6.1</td>
      <td>892.4</td>
      <td>780.0</td>
      <td>28</td>
      <td>23.70</td>
    </tr>
    <tr>
      <th>1028</th>
      <td>159.1</td>
      <td>186.7</td>
      <td>0.0</td>
      <td>175.6</td>
      <td>11.3</td>
      <td>989.6</td>
      <td>788.9</td>
      <td>28</td>
      <td>32.77</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>260.9</td>
      <td>100.5</td>
      <td>78.3</td>
      <td>200.6</td>
      <td>8.6</td>
      <td>864.5</td>
      <td>761.5</td>
      <td>28</td>
      <td>32.40</td>
    </tr>
  </tbody>
</table>
<p>1030 rows × 9 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-20bbb4a6-db8c-4a83-bc1b-6d031663c768')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-20bbb4a6-db8c-4a83-bc1b-6d031663c768 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-20bbb4a6-db8c-4a83-bc1b-6d031663c768');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4e9dbf9c-6bff-46cb-a7c8-aa8373618788">
  <button class="colab-df-quickchart" onclick="quickchart('df-4e9dbf9c-6bff-46cb-a7c8-aa8373618788')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4e9dbf9c-6bff-46cb-a7c8-aa8373618788 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_32427e92-2787-416f-972a-623ca8725f65">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('data')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_32427e92-2787-416f-972a-623ca8725f65 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('data');
      }
      })();
    </script>
  </div>

    </div>
  </div>




# 2. ElasticNet vs SqrtLasso vs SCAD

## ElasticNet


```python
class ElasticNet(nn.Module):

  def __init__(self, input_size, alpha=1.0, l1_ratio=0.5):
    super(ElasticNet, self).__init__()
    self.input_size = input_size
    self.alpha = alpha
    self.l1_ratio = l1_ratio
    self.linear = nn.Linear(input_size, 1,bias=False,device=device,dtype=dtype)

  def forward(self, x):
    return self.linear(x)

  def loss(self, y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    l1_reg = torch.norm(self.linear.weight, p=1)
    l2_reg = torch.norm(self.linear.weight, p=2)

    objective = (1/2) * mse_loss + self.alpha * (
        self.l1_ratio * l1_reg + (1 - self.l1_ratio) * (1/2)*l2_reg**2)

    return objective

  def fit(self, X, y, num_epochs=100, learning_rate=0.01):
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        self.train()
        optimizer.zero_grad()
        y_pred = self(X)
        loss = self.loss(y_pred, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

  def predict(self, X):
    self.eval()
    with torch.no_grad():
        y_pred = self(X)
    return y_pred

  def get_coefficients(self):
    return self.linear.weight
```

## SqrtLasso


```python
class SqrtLasso(nn.Module):
  def __init__(self, input_size, alpha=0.1):
    super(SqrtLasso, self).__init__()
    self.input_size = input_size
    self.alpha = alpha
    self.linear = nn.Linear(input_size, 1,bias=False,device=device,dtype=dtype)

  def forward(self, x):
    return self.linear(x)

  def loss(self, y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    l1_reg = torch.norm(self.linear.weight, p=1,dtype=torch.float64)
    loss = torch.sqrt(mse_loss) + self.alpha * (l1_reg)
    return loss

  def fit(self, X, y, num_epochs=200, learning_rate=0.01):
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        self.train()
        optimizer.zero_grad()
        y_pred = self(X)
        loss = self.loss(y_pred, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

  def predict(self, X):
    self.eval()
    with torch.no_grad():
        y_pred = self(X)
    return y_pred

  def get_coefficients(self):

    return self.linear.weight
```

## SCAD


```python
class scad_regression(nn.Module):

  def __init__(self, input_size, alpha=3, lambda_val=0.2):
    super(scad_regression, self).__init__()
    self.input_size = input_size
    self.alpha = alpha
    self.lambda_val = lambda_val
    self.linear = nn.Linear(input_size,1,device=device,dtype=dtype,bias=False)

  def forward(self, x):
    return self.linear(x)

  def loss(self, y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    weights = self.linear.weight
    is_linear = (torch.abs(weights) <= self.lambda_val)
    is_quadratic = torch.logical_and(self.lambda_val < torch.abs(weights), torch.abs(weights) <= self.alpha * self.lambda_val)
    is_constant = (self.alpha * self.lambda_val) < torch.abs(weights)
    linear_part = (self.lambda_val * torch.abs(weights) * is_linear).sum()
    quadratic_part = ((2 * self.alpha * self.lambda_val * torch.abs(weights) - weights**2 - self.lambda_val**2) / (2 * (self.alpha - 1)) * is_quadratic).sum()
    constant_part = ((self.lambda_val**2 * (self.alpha + 1)) / 2 * is_constant).sum()
    return linear_part + quadratic_part + constant_part + mse_loss

  def fit(self, X, y, num_epochs=200, learning_rate=0.001):
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
      self.train()
      optimizer.zero_grad()
      y_pred = self(X)
      loss = self.loss(y_pred.flatten(), y.flatten())
      loss.backward()
      optimizer.step()
      if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

  def predict(self, X):
    self.eval()
    with torch.no_grad():
      y_pred = self(X)
    return y_pred

  def get_coefficients(self):
    return self.linear.weight
```

## Data simulation


```python
def make_correlated_features(num_samples,p,rho):
  vcor = []
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  x = np.random.multivariate_normal(mu, r, size=num_samples)
  return x
```


```python
rho =0.9
p = 200
n = 150
```


```python
x1 = make_correlated_features(n,p,rho)
# sparsity
beta =np.array([1,-1,2,3,0,0,0,0,2,-1,2,3,4])
beta = beta.reshape(-1,1)
betastar = np.concatenate([beta,np.repeat(0,p-len(beta)).reshape(-1,1)],axis=0)
y1 = x1@betastar + 1.5*np.random.normal(size=(n,1))

```


```python
x1_scaled = scaler.fit_transform(x1)
x1_torch = torch.tensor(x1_scaled,device=device)
y1_torch = torch.tensor(y1,device=device)
```

    /usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_data.py:2627: UserWarning: n_quantiles (1000) is greater than the total number of samples (150). n_quantiles is set to n_samples.
      warnings.warn(



```python
import warnings
warnings.filterwarnings('ignore')
```

### Overall, sqrtlasso's predictions (lower mse) are closer to the true values, indicating better performance.

ElasticNet


```python
mse_elastic = []
for i in range(500):
  elasticnet_model = ElasticNet(x1_torch.shape[1])
  elasticnet_model.fit(x1_torch,y1_torch)
  coef_elastic = elasticnet_model.get_coefficients().cpu().detach().numpy()
  mse_elastic.append(mse(betastar,coef_elastic.reshape(-1,1)))
print("Mean MSE for elastic", np.mean(mse_elastic))
```

    Mean MSE for elastic 0.19711147198655868


SqrtLasso


```python
mse_sqrtlasso = []
for i in range(500):
  sqrtlasso_model = SqrtLasso(x1_torch.shape[1])
  sqrtlasso_model.fit(x1_torch,y1_torch)
  coef_sqrtlasso = sqrtlasso_model.get_coefficients().cpu().detach().numpy()
  mse_sqrtlasso.append(mse(betastar,coef_sqrtlasso.reshape(-1,1)))
print("Mean MSE for sqrtlasso", np.mean(mse_sqrtlasso))
```

    Mean MSE for sqrtlasso 0.1808115682383928


SCAD


```python
mse_scad = []
for i in range(500):
  scad_model = scad_regression(x1_torch.shape[1])
  scad_model.fit(x1_torch,y1_torch)
  coef_scad = scad_model.get_coefficients().cpu().detach().numpy()
  mse_scad.append(mse(betastar,coef_scad.reshape(-1,1)))
print("Mean MSE for scad", np.mean(mse_scad))
```

    Mean MSE for scad 0.23274134315692926

