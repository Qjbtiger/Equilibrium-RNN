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

pushdeer = PushDeer(pushkey="PDU1549TDgJhw6oNVKBSzvBeVGOSJkrRn8ynXfA4")
logging.getLogger().setLevel(logging.DEBUG)

prefixPath = "./Data/results/8/a2/"
beta = 10000.0
gList = [0.8, 0.9, 0.98, 1.0, 1.02, 1.1, 1.2]
eta = 0.0
gamma = 0.0
nonLinearity = 'tanh'

numUV = 60000
numX = 10000
isLoadSample = True
tryTimes = 1
if isLoadSample:
    mcSample = torch.load(prefixPath + "/mcSample.pt", map_location="cpu")
else:
    uss = [torch.normal(0, 1, size=(numUV, 1), device="cpu").repeat_interleave(numX, dim=1) for _ in range(tryTimes)]
    vss = [torch.normal(0, 1, size=(numUV, 1), device="cpu").repeat_interleave(numX, dim=1) for _ in range(tryTimes)]
    xsRaw = [torch.normal(0, 1, size=(numUV, numX), device="cpu") for _ in range(tryTimes)]
    mcSample = [uss, vss, xsRaw]
    torch.save(mcSample, prefixPath + "/mcSample.pt")

for i, g in enumerate(gList):
    initialValues = [1e-6, 
                     1e-7, 
                     - torch.rand(1).item(),
                     torch.rand(1).item(), 0, 0, 0, 0]
    
    for t in range(100):
        record, flag = SDEs(beta, g, eta, gamma, nonLinearity, initialValues=initialValues, mcSample=mcSample)
        if math.isfinite(record["minusBetaF"][-1]):
            minusBetaF = record["minusBetaF"][-1]
            energy = record["energy"][-1]
            L2Norm = record["L2Norm"][-1]
            finalValues = record["Qs"][-1]
            logging.info("beta: {:.2f}, g: {:.2f}, eta: {:.2f}, gamma: {:.2f}, minusBetaF: {:.4f}, energy: {:.4f}, L2Norm: {:.4f}, flag: {}, numIter: {}, t: {}".format(beta, g, eta, gamma, minusBetaF, energy, L2Norm, flag, record["iterationIndex"][-1], t))
            
            if flag:
                break         

    # save
    record = {
        "g": g,
        "minusBetaF": torch.tensor(record["minusBetaF"]),
        "energy": torch.tensor(record["energy"]),
        "L2Norm": torch.tensor(record["L2Norm"]),
        "Qs": torch.tensor(record["Qs"]).t()
    }
    torch.save(record, prefixPath + f"/path-beta{beta}-gamma{gamma}-g{g}-eta{eta:.1f}-{nonLinearity}.pt")

pushdeer.send_text("Program done!")