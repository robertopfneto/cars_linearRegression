#%%
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from cars_linearRegression.model import LinearRegressionTorch

#%% Importando dataset
arquivo_carro = 'car_data.csv'
carros = pd.read_csv(arquivo_carro)

# Visualizando dados
sns.scatterplot(x='Present_Price', y='Selling_Price', data=carros)
sns.regplot(x='Present_Price', y='Selling_Price', data=carros)

#%% Pré-processamento e normalização
X_np = carros['Present_Price'].values.astype(np.float32).reshape(-1,1)
y_np = carros['Selling_Price'].values.astype(np.float32).reshape(-1,1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_np)
y_scaled = scaler_y.fit_transform(y_np)

X = torch.from_numpy(X_scaled)
y_true = torch.from_numpy(y_scaled)

# Inicializa modelo
model = LinearRegressionTorch(1, 1)

#%% Função de perda e otimizador
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#%% Treinamento
EPOCHS = 1000
losses, slope, bias = [], [], []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_func(y_pred, y_true)
    loss.backward()
    optimizer.step()

    # Armazenando parâmetros
    for name, param in model.named_parameters():
        if name == 'linear.weight':
            slope.append(param.item())
        elif name == 'linear.bias':
            bias.append(param.item())

    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

#%% Visualizações do treinamento
sns.lineplot(x=range(EPOCHS), y=losses).set(title='Perda durante o treinamento')
plt.show()

sns.lineplot(x=range(EPOCHS), y=slope).set(title='Inclinação (peso) ao longo das épocas')
plt.show()

sns.lineplot(x=range(EPOCHS), y=bias).set(title='Viés (bias) ao longo das épocas')
plt.show()

#%% Visualização final da regressão aprendida
# Reconvertendo para escala original
X_plot = scaler_X.inverse_transform(X_scaled)
y_pred_plot = model(X).detach().numpy()
y_pred_plot = scaler_y.inverse_transform(y_pred_plot)

sns.scatterplot(x=X_np.reshape(-1), y=y_np.reshape(-1), label='Dados reais')
sns.lineplot(x=X_plot.reshape(-1), y=y_pred_plot.reshape(-1), color='red', label='Regressão aprendida')
plt.title("Regressão Linear com PyTorch")
plt.show()

# %%
