# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numba as nb
import multiprocessing as processing
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# %%
N = 2
Sigma = 0.05
lambdat = 1.0
g = 1.0
eta = 0.5
numProcess = 12
# non linears case
phi = np.tanh 

# %%
dt = 0.01

@nb.njit
def linearIter(x_0, K):
    return x_0 + (-lambdat * x_0 + g * np.dot(K, x_0) + np.random.normal(0, np.sqrt(Sigma))) * dt

@nb.njit
def nonLinearIter(x_0, K, phi):
    return x_0 + (-lambdat * x_0 + g * np.dot(K, phi(x_0)) + np.random.normal(0, np.sqrt(Sigma))) * dt

# %%
@nb.njit
def task(numTrajacry=1000):
    eps = 1e-3
    stationaryPoints = []
    maxIteraton = 10000
    numSuccess = 0

    for t in range(numTrajacry):
        K = np.random.normal(0.0, 1 / np.sqrt(N), size=(N, N))
        K = K + eta * K.T
        x_0 = np.random.uniform(-10, 10, size=N)

        flag = False
        for i in range(maxIteraton):
            x = linearIter(x_0, K)
            if np.max(np.abs(x - x_0)) < eps:
                flag = True
                break
            x_0 = x
        
        if flag:
            stationaryPoints.append(x)
            numSuccess += 1

    return stationaryPoints, numSuccess

numTrajacry = 1000000

p = processing.Pool(processes=numProcess)
numTrajacryPreP = [numTrajacry // numProcess] * (numProcess - 1)
numTrajacryPreP.append(numTrajacry % numProcess)
ress = p.map(task, numTrajacryPreP)
p.close()
p.join()

stationaryPoints = []
numSuccess = 0
for stationaryPointsPerProcess, numSuccessPerprocess in ress:
    stationaryPoints.extend(stationaryPointsPerProcess)
    numSuccess += numSuccessPerprocess

# stationaryPoints, numSuccess = task(numTrajacry=numTrajacry)
print("Find stationary point: {}/{}".format(numSuccess, numTrajacry))

# %%
stationaryPoints = np.asarray(stationaryPoints)
plt.scatter(stationaryPoints[:, 0], stationaryPoints[:, 1], s=2.0)

# %%
print('Points ranges in [{:.2f}-{:.2f}, {:.2f}-{:.2f}]'.format(np.min(stationaryPoints[:, 0]), np.max(stationaryPoints[:, 0]), np.min(stationaryPoints[:, 1]), np.max(stationaryPoints[:, 1])))

epsl = 0.01
unit = int(1 / epsl)
x = np.arange(-20.0, 20.0, epsl)
y = np.arange(-20.0, 20.0, epsl)
pointsCounter = np.zeros(shape=(x.shape[0], y.shape[0]), dtype=np.int64)

for point in stationaryPoints:
    if point[0] >= -20 and point[0] <= 20 and point[1] >= -20 and point[1] <= 20:
        pointsCounter[np.floor(point[0] * unit).astype(np.int64) + 20 * unit, np.floor(point[1] * unit).astype(np.int64) + 20 * unit] += 1

# X, Y = np.meshgrid(x, y)
pointsCounterLog = np.log(pointsCounter + 1)
im = plt.imshow(pointsCounterLog)
plt.colorbar(im)
plt.show()
plt.close()

# %%
nn = torch.nn
mlp = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    nn.ReLU()
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
pStar = pointsCounter / numSuccess
class PointDataset(Dataset):
    def __init__(self, probability, X, Y):
        super(PointDataset).__init__()
        self.data = torch.tensor(np.stack([X, Y], axis=2).reshape(-1, 2), dtype=torch.float32).to(device)
        self.labels = torch.tensor(probability.reshape(-1)).to(device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

X, Y = np.meshgrid(x, y)
pointsDataset = PointDataset(pStar, X, Y)
pointsDataloader = DataLoader(pointsDataset, batch_size=5120, shuffle=True)

# %%
class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, inputs, targets):
        return -torch.sum(targets * torch.log(inputs + 1e-10))


# %%
learningRate = 0.001
numEpoch = 100
criterier = KLDivergence()
optimizer = optim.Adam(mlp.parameters(), lr=learningRate)

mlp = mlp.to(device)
for t in range(numEpoch):
    mlp.train()
    totalLosses = 0.0
    for i, (coordinates, targets) in enumerate(pointsDataloader):
        coordinates = coordinates
        targets = targets
        outputs = mlp(coordinates)
        loss = criterier(outputs.squeeze(), targets)

        optimizer.zero_grad()
        loss.backward()
        totalLosses += loss.item()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch: {}, Batches: {}/{}, Losses: {}'. format(t, i+1, len(pointsDataloader), totalLosses), end='\r')

    print('Epoch: {}, Losses: {}'.format(t, totalLosses))
# %%
