#%%
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


#%% Importando dataset
arquivo_carro = 'car_data.csv'
carros = pd.read_csv(arquivo_carro)

# Visualizando dados
sns.scatterplot(x='Present_Price', y='Selling_Price', data=carros)
sns.regplot(x='Present_Price', y='Selling_Price', data=carros)

#%% Pré-processamento e normalização
X_list = carros['Present_Price'].values
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)

y_list = carros['Selling_Price'].values
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)

# transforma as matrizes resultantes em tensores do pytorch
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

# Hyperparametros
EPOCHS = 200
batch_size = 2
losses, slope, bias = [], [], []

#%% Criando Dataset e Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset=LinearRegressionDataset(X_np, y_np), batch_size=batch_size)


#%% model
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size): 
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Inicializa modelo
model = LinearRegressionTorch(1, 1)

#%% Função de perda e otimizador
loss_func = nn.MSELoss()

lr = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#%% Treinamento
for epoch in range(EPOCHS):
    for i, (X_batch, y_batch) in enumerate(train_loader):

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_func(y_pred, y_batch)
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


#%% Salvando os pesos
model.state_dict()
torch.save(model.state_dict(),'model.pth')

#%% carregar os pesos
model = LinearRegressionTorch(input_size=1,output_size=1)
model.load_state_dict(torch.load('model.pth'))

#%% Visualizações do treinamento
sns.lineplot(x=range(EPOCHS), y=losses).set(title='Perda durante o treinamento')
plt.show()

sns.lineplot(x=range(EPOCHS), y=slope).set(title='Inclinação (peso) ao longo das épocas')
plt.show()

sns.lineplot(x=range(EPOCHS), y=bias).set(title='Viés (bias) ao longo das épocas')
plt.show()

#%% Visualização final da regressão aprendida
# Reconvertendo para escala original
# Geração das previsões diretamente no espaço original
y_pred = model(X).detach().numpy().reshape(-1)

# Plotando os dados reais e a regressão aprendida
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_np.reshape(-1), y=y_np.reshape(-1), label='Dados reais')
sns.lineplot(x=X_np.reshape(-1), y=y_pred, color='red', label='Regressão aprendida')
plt.xlabel("Present_Price")
plt.ylabel("Selling_Price")
plt.title("Regressão Linear com PyTorch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
