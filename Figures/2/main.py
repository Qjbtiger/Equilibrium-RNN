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
prefixPath = "./Data/results/2/a1/"
pushdeer = PushDeer(pushkey="PDU1549TDgJhw6oNVKBSzvBeVGOSJkrRn8ynXfA4")
betaList = torch.logspace(2, 4, 10)
g = 1.00
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

pushdeer = PushDeer(pushkey="PDU1549TDgJhw6oNVKBSzvBeVGOSJkrRn8ynXfA4")
logging.getLogger().setLevel(logging.DEBUG)
beta = 10000.0
gList = torch.arange(0.8, 1.2, 0.02)
eta = 0.0
gammaLists = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
nonLinearity = 'tanh'

for gamma in gammaLists:
    minusBetaFList = []
    energyList = []
    L2NormList = []
    QList = []
    initialValues = None

    for i, g in enumerate(gList):
        _ = [x.append([]) for x in [minusBetaFList, energyList, L2NormList, QList]]
        for t in range(100):
            record, flag = SDEs(beta, g.item(), eta, gamma, nonLinearity, initialValues=initialValues, mcSample=mcSample)
            if math.isfinite(record["minusBetaF"][-1]):
                minusBetaF = record["minusBetaF"][-1]
                energy = record["energy"][-1]
                L2Norm = record["L2Norm"][-1]
                finalValues = record["Qs"][-1]
                logging.info("beta: {:.2f}, g: {:.2f}, eta: {:.2f}, gamma: {:.2f}, minusBetaF: {:.4f}, energy: {:.4f}, L2Norm: {:.4f}, flag: {}, numIter: {}, t: {}".format(beta, g.item(), eta, gamma, minusBetaF, energy, L2Norm, flag, record["iterationIndex"][-1], t))
                
                if flag:
                    minusBetaFList[-1].append(minusBetaF)
                    energyList[-1].append(energy)
                    L2NormList[-1].append(L2Norm)
                    QList[-1].append(finalValues)

                    initialValues = finalValues

                    if len(minusBetaFList[-1]) >= 1:
                        break
                else:
                    initialValues = None
                    
        minusBetaFList[-1], energyList[-1], L2NormList[-1]= [torch.mean(torch.tensor(x[-1]), dim=0).item() for x in [minusBetaFList, energyList, L2NormList]]
        QList[-1] = torch.mean(torch.tensor(QList[-1]), dim=0).tolist()                

        # save
        record = {
            "gList": gList.tolist(),
            "currentIndex": i,
            "minusBetaF": torch.tensor(minusBetaFList),
            "energy": torch.tensor(energyList),
            "L2Norm": torch.tensor(L2NormList),
            "Qs": torch.tensor(QList).t()
        }
        torch.save(record, prefixPath + f"/gCurve-beta{beta}-gamma{gamma}-gi{gList[0].item():.1f}-gf{gList[-1].item():.1f}-eta{eta:.1f}-{nonLinearity}.pt")

pushdeer.send_text("Program done!")