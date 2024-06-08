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

from Replicas import SDEsZeroTemperature

if __name__ == "__main__":
    # Task
    pushdeer = PushDeer(pushkey="PDU1549TDgJhw6oNVKBSzvBeVGOSJkrRn8ynXfA4")
    prefixPath = "./Data/results/10/a8"
    logging.getLogger().setLevel(logging.DEBUG)
    gListOfGamma = {
        0.0: torch.arange(0.8, 0.95, 0.02).tolist() + torch.arange(0.95, 1.2, 0.01).tolist(),
        0.2: torch.arange(0.7, 0.82, 0.02).tolist()+ torch.arange(0.8, 1.0, 0.01).tolist(),
        0.5: torch.arange(0.5, 0.62, 0.02).tolist() + torch.arange(0.6, 0.8, 0.01).tolist(),
        -0.2: torch.arange(1.1, 1.4, 0.01).tolist()
    }
    eta = 0.0
    gammaLists = [0.2]
    nonLinearity = 'tanh'

    for repeatIndex in range(1):
        for gamma in gammaLists:
            gList = gListOfGamma[gamma]
            minusFList = []
            energyList = []
            L2NormList = []
            QList = []
            QsIteration = []
            initialValues = None

            for i, g in enumerate(gList):
                for t in range(100):
                    record, flag = SDEsZeroTemperature(g, eta, gamma, nonLinearity, initialValues=initialValues)
                    if len(record["minusF"]) >= 1 and math.isfinite(record["minusF"][-1]):
                        minusF = record["minusF"][-1]
                        energy = record["energy"][-1]
                        L2Norm = record["L2Norm"][-1]
                        finalValues = record["Qs"][-1]
                        logging.info("g: {:.2f}, eta: {:.2f}, gamma: {:.2f}, minusF: {:.4f}, energy: {:.4f}, L2Norm: {:.4f}, flag: {}, numIter: {}, t: {}".format(g, eta, gamma, minusF, energy, L2Norm, flag, record["iterationIndex"][-1], t))
                        
                        # if flag:
                        minusFList.append(minusF)
                        energyList.append(energy)
                        L2NormList.append(L2Norm)
                        QList.append(finalValues)
                        QsIteration.append(record["Qs"])

                        # initialValues = finalValues
                        break

                    initialValues = None    

                # save
                record = {
                    "gList": gList,
                    "currentIndex": i,
                    "minusF": torch.tensor(minusFList),
                    "energy": torch.tensor(energyList),
                    "L2Norm": torch.tensor(L2NormList),
                    "Qs": torch.tensor(QList).t(),
                    "QsIteration": QsIteration
                }
                torch.save(record, prefixPath + f"/gCurve-gamma{gamma}-gi{gList[0]:.1f}-gf{gList[-1]:.1f}-eta{eta:.1f}-{nonLinearity}{repeatIndex}.pt")

    pushdeer.send_text("Program done!")