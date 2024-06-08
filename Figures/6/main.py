import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pypushdeer import PushDeer
import os
import sys
sys.path.append(os.getcwd())
import logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
from Replicas import SDEs

# Task
logging.getLogger().setLevel(logging.DEBUG)
pushdeer = PushDeer(pushkey="PDU1549TDgJhw6oNVKBSzvBeVGOSJkrRn8ynXfA4")
prefixPath = "./Data/results/6/a1"
betaList = torch.logspace(2, 4, 10)
g = 0.98
gamma = 0.0
eta = 0.0
nonLinearity = 'tanh'
initialValues = None
minusBetaFList = []
energyList = []
L2NormList = []
QList = []
termList = []

numUV = 60000
numX = 10000
isLoadSample = False
tryTimes = 20
if isLoadSample:
    mcSample = torch.load(prefixPath + "/mcSample.pt", map_location="cpu")
else:
    uss = [torch.normal(0, 1, size=(numUV, 1), device="cpu").repeat_interleave(numX, dim=1) for _ in range(tryTimes)]
    vss = [torch.normal(0, 1, size=(numUV, 1), device="cpu").repeat_interleave(numX, dim=1) for _ in range(tryTimes)]
    xsRaw = [torch.normal(0, 1, size=(numUV, numX), device="cpu") for _ in range(tryTimes)]
    mcSample = [uss, vss, xsRaw]
    torch.save(mcSample, prefixPath + "/mcSample.pt")

for i, beta in enumerate(betaList):
    for t in range(100):
        record, flag = SDEs(beta.item(), g, eta, gamma, nonLinearity, initialValues=initialValues, mcSample=mcSample)
        if math.isfinite(record["minusBetaF"][-1]) and flag:
            minusBetaF = record["minusBetaF"][-1]
            energy = record["energy"][-1]
            L2Norm = record["L2Norm"][-1]
            finalValues = record["Qs"][-1]
            logging.info("beta: {:.2f}, g: {:.2f}, eta: {:.2f}, minusBetaF: {:.4f}, energy: {:.4f}, L2Norm: {:.4f}, flag: {}, numIter: {}, t: {}".format(beta.item(), g, eta, minusBetaF, energy, L2Norm, flag, record["iterationIndex"][-1], t))
            
            if flag:
                minusBetaFList.append(minusBetaF)
                energyList.append(energy)
                L2NormList.append(L2Norm)
                QList.append(finalValues)

                record = {
                    "beta": betaList,
                    "currentIndex": i,
                    "minusBetaF": torch.tensor(minusBetaFList),
                    "energy": torch.tensor(energyList),
                    "L2Norm": torch.tensor(L2NormList),
                    "Qs": torch.tensor(QList).t()
                }
                torch.save(record, prefixPath + f"/beta-g{g:.2f}-gamma{gamma:.1f}-bi{betaList[0].item():.0f}-bf{betaList[-1].item():.0f}-eta{eta:.1f}-{nonLinearity}.pt")
                break

    initialValues = finalValues

pushdeer.send_text("Program done!")