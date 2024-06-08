import torch
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
import os
import sys
sys.path.append(os.getcwd())
import logging, math, Utils
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
from pypushdeer import PushDeer

def getXStar(gammas, w):
    x = torch.zeros_like(gammas, device=device)
    
    flag = False
    for _ in range(10000):
        newx = gammas + w * torch.tanh(x)
        if torch.all(torch.abs(newx - x) < 1e-5):
            flag = True
            break
        
        x = newx.clone()

    if not flag:
        logging.info("Not converged!")
    
    return x
    

def getCR(g, eta, sigma, args):
    '''
        args: numgamma, maxIteration, epsilon, damping, phi, phiPrime
    '''

    numgamma, maxIteration, epsilon,damping, phi, phiPrime = args

    C = torch.rand(size=(1, )).item()
    R = torch.rand(size=(1, )).item() * 0.1

    record = Utils.Record()
    for t in range(maxIteration):
        gamma = torch.normal(0, g * math.sqrt(C), size=(numgamma, ), device=device)

        w = (g**2) * eta * R
        xStar = getXStar(gamma, w)
        phiStar = phi(xStar)
        newC = torch.mean(phiStar**2).item()
        newR = torch.mean(
            phiPrime(gamma + w * phiStar) * (1 + w * R)
        ).item()

        record.add({
            "C": C,
            "R": R
        })
        
        logging.debug("Iteration {t}: C={C}, R={R}".format(t=t, C=C, R=R))
        
        if torch.all(torch.abs(torch.tensor([newC, newR]) - torch.tensor([C, R])) < epsilon) and t >= 40:
            break
        
        C = damping * C + (1 - damping) * newC
        R = damping * R + (1 - damping) * newR

    return record

if __name__ == "__main__":
    prefixPath = "./Data/results/11/a2"
    pushdeer = PushDeer(pushkey="PDU1549TDgJhw6oNVKBSzvBeVGOSJkrRn8ynXfA4")

    torch.set_default_dtype(torch.float64)

    etaList = [0.8]
    gListOfEta = {
        0.0: torch.arange(0.8, 1.2, 0.01).tolist(),
        0.2: torch.arange(0.65, 1.1, 0.01).tolist(),
        0.5: torch.arange(0.5, 0.85, 0.01).tolist(),
        0.8: torch.arange(0.4, 0.8, 0.01).tolist(),
    }
    for eta in  etaList:
        gList = gListOfEta[eta]
        CList = []
        RList = []
        IterationList = []
        for g in gList:
            record = getCR(g=g, eta=eta, sigma=0.0, args=(
                100000, # numgamma
                100, # maxIteration
                1e-5, # epsilon
                0.2, # damping
                torch.tanh, # phi
                lambda x: 1 - torch.tanh(x)**2 # phiPrime
            ))

            logging.info(f"g: {g:.2f}, C: {record.C[-1]}, R: {record.R[-1]}")

            CList.append(record.C[-1])
            RList.append(record.R[-1])
            IterationList.append(torch.stack([torch.tensor(list) for list in [record.C, record.R]], dim=0))

        torch.save({
            "gList": gList,
            "C": torch.tensor(CList),
            "R": torch.tensor(RList),
            "IterationList": IterationList
        }, prefixPath + "/dmfts-eta{}.pt".format(eta))

    pushdeer.send_text("Program done!")