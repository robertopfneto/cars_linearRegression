#%%
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np

#%% importando dataset
arquivo_carro = 'car_data.csv'
carros = pd.read_csv(arquivo_carro)
carros.head()
# %% Visualizar modelo
sns.scatterplot(x='Present_Price', y='Selling_Price', data=carros)
sns.regplot(x='Present_Price', y='Selling_Price', data=carros)
#%% convertendo para tensores

# converte os dados em matrizes do numpy e redimensiono
X_list = carros['Present_Price'].values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1) #

y_list = carros['Selling_Price'].values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)

# transformo as matrizes resultantes em tensores do pytorch
X = torch.from_numpy(X_np) 
y_true = torch.from_numpy(y_np)

# %% Classe do Modelo
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size): 
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_dim, output_dim)
# %% Função de Perda
loss_func = nn.MSELoss()

#%% Optimizer
LR = 0.02 #testar diferentes valores (01 ate 0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

losses, slope, bias = [],[],[]

#%% treinamento
EPOCHS = 1000
for epoch in range(EPOCHS):
    # zerando gradiente
    optimizer.zero_grad()

    #forward pass
    y_pred = model(X)

    # computa perda
    loss = loss_func(y_pred,y_true)
    loss.backward()

    # atualizando pesos
    optimizer.step()

    # pegando parametros
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'lienar.bias':
                bias.append(param.data.numpy()[0])
            
    losses.append(float(loss.data))
    if epoch % 100 == 0: # a cada 100 epocas
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.data))

#%% visualizar treinamento
sns.scatterplot(x=range(EPOCHS), y=losses) #exibe grafico de perda

#%% atualizacao do bias
sns.scatterplot(x=range(EPOCHS), y=bias) #exibe grafico de perda

# atualizacao da delividade
sns.scatterplot(x=range(EPOCHS), y=slope) #exibe grafico de perda

# chegando resultados
