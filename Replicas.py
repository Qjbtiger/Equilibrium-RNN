import math
import time
import numpy as np
import scipy.integrate as integrate
import random
import torch
from Utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
import logging

# compute free energy by Monte Carlo method
def getMCSample(numUV, numX):
    us = torch.normal(0, 1, size=(numUV, 1), device=device)
    vs = torch.normal(0, 1, size=(numUV, 1), device=device)
    us = us.repeat_interleave(numX, dim=1)
    vs = vs.repeat_interleave(numX, dim=1)

    xs = torch.normal(0, 1, size=(numUV, numX), device=device)

    return us, vs, xs

def isConverge(Qs, newQs, damping):
    diff = torch.abs(Qs - newQs)
    return torch.all(diff < 1e-3 * (1 - damping)) or torch.all(diff / newQs < 1e-2 * (1 - damping))


def SDEs(beta, g, eta, gamma, nonLinearity, damping=0.2, initialValues=None, mcSample=None):
    sqrtBeta = math.sqrt(beta)
    g2 = g**2
    phi = getattr(torch, nonLinearity)
    
    numUV = 60000
    numX = 20000
    tryTimes = len(mcSample[0])
    # if mcSample is None:
    #     numUV = 60000
    #     numX = 20000
    #     us = torch.normal(0, 1, size=(numUV, 1), device=device)
    #     vs = torch.normal(0, 1, size=(numUV, 1), device=device)
    #     xsRaw = torch.normal(0, 1, size=(1, numX), device=device)
    # else:
    #     us, vs = mcSample[0], mcSample[1]
    #     xsRaw = mcSample[2]

    
    if initialValues is not None:
        q = torch.tensor(initialValues[0], device=device)
        Q = torch.tensor(initialValues[1], device=device)
        qhat = torch.tensor(initialValues[2], device=device)
        Qhat = torch.tensor(initialValues[3], device=device)
        r = torch.tensor(initialValues[4], device=device)
        R = torch.tensor(initialValues[5], device=device)
        rhat = torch.tensor(initialValues[6], device=device)
        Rhat = torch.tensor(initialValues[7], device=device)
    else:
        # q = torch.rand(1, device=device)
        q = torch.tensor(1e-5, device=device)
        Q = torch.rand(1, device=device) * q
        # newq = torch.tensor(0.0)
        # newQ = torch.tensor(0.0)
        qhat = - torch.rand(1, device=device)
        Qhat = torch.rand(1, device=device)
        r = torch.rand(1, device=device)
        R = torch.rand(1, device=device) * r
        # rhat = torch.rand(1, device=device)
        # Rhat = torch.rand(1, device=device)
        # r = torch.tensor(0.0, device=device)
        # R = torch.tensor(0.0, device=device)
        rhat = torch.tensor(0.0, device=device)
        Rhat = torch.tensor(0.0, device=device)
    qhat = torch.tensor(0)
    Qhat = torch.tensor(0)
    logging.debug("Iteration: 0, " + ", ".join([f"{i}: {j.item():.4f}" for i, j in zip(
        ["q", "Q", "qhat", "Qhat", "r", "R", "rhat", "Rhat"], 
        [q, Q, qhat, Qhat, r, R, rhat, Rhat])]))

    maxIteration = 4000
    
    record = Record()
    flag = False
    if tryTimes > 1:
        uss, vss = mcSample[0], mcSample[1]
        xsRaw = mcSample[2]
    else:
        us, vs = mcSample[0][0], mcSample[1][0]
        xsRaw = mcSample[2][0]

    for t in range(maxIteration):
        sigma2 = 1 + beta*g*g*(q - Q)
        sigma = torch.sqrt(sigma2)
        k = g * beta / sigma2
        gkQ = g * k * Q

        def f(c1: float, c2: float, c4: float):
            return c1 * rhat + c2 * Rhat + c4 * gkQ * (rhat - Rhat)

        c = beta * (1/sigma2 + 2*eta)

        averageOverSquarePhi = 0
        averageOverPhiSquare = 0
        averageOverSquareX = 0
        averageOverXSquare = 0
        averageOverPhiX = 0
        averageOverPhiMultiplyX = 0
        for i in range(tryTimes):
            if tryTimes > 1:
                us = uss[i].to(device)
                vs = vss[i].to(device)
            xs = torch.sqrt(1 / c) * (xsRaw[i].to(device))
            # us = torch.normal(0, 1, size=(numUV, 1), device=device)
            # vs = torch.normal(0, 1, size=(numUV, 1), device=device)
            # xs = torch.normal(0, torch.sqrt(1 / c).item(), size=(1, numX), device=device)
            # xs = torch.sqrt(1 / c) * xsRaw
            phxs = phi(xs)

            H = (2*qhat - Qhat) * (phxs**2) / 2 \
                + (2 * sqrtBeta * (rhat - Rhat) * xs * phxs - ((rhat - Rhat) * phxs)**2) / sigma2 / 2 \
                + torch.sqrt(Qhat - (Rhat**2) / (g2 * beta * Q)) * phxs * us \
                - (g * torch.sqrt(beta * Q) * ((rhat - Rhat) * phxs - sqrtBeta * xs) / sigma2 - Rhat * phxs / (g * torch.sqrt(beta * Q))) * vs

            expTerm = torch.exp(H - H.max())

            numeritorTerm1 = torch.mean(expTerm * (phxs**2), dim=1)
            denominatorTerm = torch.mean(expTerm, dim=1)
            numeritorTerm2 = torch.mean(expTerm * phxs, dim=1)
            numeritorTerm3 = torch.mean(expTerm * (xs**2), dim=1)
            numeritorTerm4 = torch.mean(expTerm * xs, dim=1)
            numeritorTerm5 = torch.mean(expTerm * phxs * xs, dim=1)
            c4 = (denominatorTerm==0).sum().item()
            if c4 > 0:
                logging.info(c4)
                return record, False
            
            averageOverSquarePhi += torch.mean(numeritorTerm1 / denominatorTerm)
            averageOverPhiSquare += torch.mean((numeritorTerm2 / denominatorTerm)**2)
            averageOverSquareX += torch.mean(numeritorTerm3 / denominatorTerm)
            averageOverXSquare += torch.mean((numeritorTerm4 / denominatorTerm)**2)
            averageOverPhiX += torch.mean(numeritorTerm5 / denominatorTerm)
            averageOverPhiMultiplyX += torch.mean(numeritorTerm2 * numeritorTerm4 / (denominatorTerm**2))
        
        averageOverSquarePhi /= tryTimes
        averageOverPhiSquare /= tryTimes
        averageOverSquareX /= tryTimes
        averageOverXSquare /= tryTimes
        averageOverPhiX /= tryTimes
        averageOverPhiMultiplyX /= tryTimes
        
        newq = 1.0 * averageOverSquarePhi
        newQ = 1.0 * averageOverPhiSquare
        newR = - f(0, 1, -1) * averageOverSquarePhi / sigma2 \
            - f(1, -2, 1) * averageOverPhiSquare / sigma2 \
            - sqrtBeta * gkQ * averageOverPhiX / sigma2 \
            + sqrtBeta * (1 + gkQ) * averageOverPhiMultiplyX / sigma2
        newRhat = beta * g2 * gamma * R

        newqhat = -g * k / 2 \
            + g2 * (k**2) * Q / 2 \
            + (k**2) * (1 - 2*gkQ) * averageOverSquareX / 2 \
            + (k**3) * g * Q * averageOverXSquare \
            + g * k * (rhat - Rhat) * f(1, 1, -2) * averageOverSquarePhi / (2 * sigma2) \
            + g * k * (rhat - Rhat) * f(0, -1, 1) * averageOverPhiSquare / sigma2 \
            + g * k * sqrtBeta * f(0, 1, -2) * averageOverPhiMultiplyX / sigma2 \
            + g * k * sqrtBeta * f(-1, 0, 2) * averageOverPhiX / sigma2
        newQhat = g2 * (k**2) * Q \
            + (k**2) * (1 + 2 * gkQ) * averageOverXSquare \
            - 2 * g * (k**3) * Q * averageOverSquareX \
            + 2 * g * k * (rhat - Rhat) * f(0, 1, -1) * averageOverSquarePhi / sigma2 \
            + g * k * (rhat - Rhat) * f(1, -3, 2) * averageOverPhiSquare / sigma2 \
            - 2 * g * k * sqrtBeta * f(1, -2, 2) * averageOverPhiMultiplyX / sigma2 \
            + 2 * g * k * sqrtBeta * f(0, -1, 2) * averageOverPhiX / sigma2
        newr = - f(1, 0, -1) * averageOverSquarePhi / sigma2 \
            + f(0, 1, -1) * averageOverPhiSquare / sigma2 \
            + sqrtBeta * (1 - gkQ) * averageOverPhiX / sigma2 \
            + sqrtBeta * gkQ * averageOverPhiMultiplyX / sigma2
        
        newrhat = beta * g2 * gamma * r

        if damping > 0:
            newq = damping * q + (1 - damping) * newq
            newQ = damping * Q + (1 - damping) * newQ
            newr = damping * r + (1 - damping) * newr
            newR = damping * R + (1 - damping) * newR
            newRhat = damping * Rhat + (1 - damping) * newRhat
            newqhat = damping * qhat + (1 - damping) * newqhat
            newQhat = damping * Qhat + (1 - damping) * newQhat
            newrhat = damping * rhat + (1 - damping) * newrhat
        
        intOverLnI = torch.mean(torch.log(torch.sqrt(2 * torch.pi / c) * torch.exp(H).mean(dim=1)))

        term1 = newQ * newQhat / 2 - newq * newqhat + newR * newRhat - newr * newrhat
        term2 = - torch.log(sigma)
        term3 = - g2 * beta * Q / sigma2 / 2
        term4 = beta * g2 * gamma * (r**2 - R**2) / 2
        term5 = intOverLnI
        minusBetaF = term1 + term2 + term3 + term4 + term5

        energy = g2 * (q - gkQ * (q - Q)) / (2 * sigma2) - g2 * (r * r - R * R) * gamma / 2 \
        + (1 + 2 * eta * sigma2 - g * k * (q - Q) - 2 * gkQ / sigma2) * averageOverSquareX / (2 * sigma2) \
        + gkQ * averageOverXSquare / (sigma2 ** 2) \
        - (f(1, -1, 0) + f(0, 1, -3) / sigma2 + g * k * (q - Q) * f(-2, 1, 1)) * averageOverPhiX / (2 * sqrtBeta * sigma2) \
        - (f(0, -1, 3) / sigma2 + g * k * (q - Q) * f(0, 1, -1)) * averageOverPhiMultiplyX / (2 * sqrtBeta * sigma2) \
        - (Rhat**2 + g * k * gkQ * (q - Q) * ((rhat - Rhat)**2) + f(0, -1, 1) * (f(0, 1, 1) - 2 * g * k * gkQ * (q - Q) * (rhat - Rhat))) * averageOverSquarePhi / (2 * g2 * Q * (beta**2)) \
        + (Rhat**2 + f(0, -1, 1) * (f(0, 1, 1) - 2 * g * k * gkQ * (q - Q) * (rhat - Rhat))) * averageOverPhiSquare / (2 * g2 * Q * (beta**2))
        
        L2Norm = 1.0 * averageOverSquareX
        
        logging.debug(f"Iteration: {t+1}, " \
                      + ", ".join([f"{i}: {j.item():.4f}" for i, j in zip(
                        ["q", "Q", "qhat", "Qhat", "r", "R", "rhat", "Rhat"], 
                        [newq, newQ, newqhat, newQhat, newr, newR, newrhat, newRhat])]) \
                      + ", minusBetaF: {:.4f}, energy: {:.4f}, L2Norm: {:.4f}".format(minusBetaF.item(), energy.item(), L2Norm.item()))
        
        record.add({
            "iterationIndex": t,
            "Qs": [a.item() for a in [q, Q, qhat, Qhat, r, R, rhat, Rhat]],
            "minusBetaF": minusBetaF.item(),
            "energy": energy.item(),
            "L2Norm": L2Norm.item()
        })

        Qs = torch.tensor([q, Q, r, R])
        newQs = torch.tensor([newq, newQ, newr, newR])
        if isConverge(Qs, newQs, damping) and t >= 100:
            flag = True
            logging.debug("Done!")
            break

        if not (torch.isfinite(newq) and torch.isfinite(newQ) and torch.isfinite(newqhat) and torch.isfinite(newQhat) and torch.isfinite(newr) and torch.isfinite(newR) and torch.isfinite(newrhat) and torch.isfinite(newRhat)):
            break

        q = newq.clone()
        Q = newQ.clone()
        qhat = newqhat.clone()
        Qhat = newQhat.clone()
        r = newr.clone()
        R = newR.clone()
        rhat = newrhat.clone()
        Rhat = newRhat.clone()

        if Q < 0:
            logging.debug("Q should great than zero!")
            Q = torch.tensor(1e-5, device=device)
        if Qhat < 0:
            logging.debug("Qhat should great than zero!")
            Qhat = torch.tensor(1e-5, device=device)
        # if qhat > 0:
        #     qhat = - torch.tensor(10, device=device)
        # if rhat > 100.0:
        #     rhat = torch.tensor(100.0, device=device)
        # if Rhat > 100.0:
        #     Rhat = torch.tensor(100.0, device=device)
        if Rhat**2 > g2 * beta * Q * Qhat:
            logging.debug("Rhat too large!")
            Rhat = torch.sqrt(g2 * beta * Q * Qhat * 0.1)

        if t >= 1000:
            damping = 0.6
        if t >= 1500:
            damping = 0.8

    return record, flag

def goldenSection(f, a, b, maxIteration=1000, tol=1e-12):
    goldenConst = 0.61803398875
    x0 = (a + b) / 2
    flag = False
    for i in range(maxIteration):
        c = (b - a) * goldenConst
        atilde = b - c
        btilde = a + c
        fatilde = f(a)
        fbtilde = f(b)
        faLefb = fatilde <= fbtilde
        faGefb = ~faLefb
        b[faLefb] = btilde[faLefb]
        a[faGefb] = atilde[faGefb]
        x1 = (a + b) / 2

        if torch.all(torch.abs(x0 - x1) < tol):
            flag = True
            break

        x0 = x1.clone()
    
    return x1, flag

def findMins(f, shape):
    '''
        golden section search method
    '''
    a = torch.ones(size=shape, device=device, dtype=dtype) * (-10)
    b = torch.zeros(size=shape, device=device, dtype=dtype)
    xl, flag = goldenSection(f, a, b, maxIteration=1000, tol=1e-12)

    a = torch.zeros(size=shape, device=device, dtype=dtype)
    b = torch.ones(size=shape, device=device, dtype=dtype) * 10
    xr, flag = goldenSection(f, a, b, maxIteration=1000, tol=1e-12)

    fl = f(xl)
    fr = f(xr)
    flGefr = fl > fr
    # logging.debug("Amount of swich position: {}".format(torch.sum(flGefr)))
    xl[flGefr] = xr[flGefr]
    
    if not flag:
        logging.debug("Not converge!")

    return xl


def SDEsZeroTemperature(g, eta, gamma, nonLinearity, damping=0.2, initialValues=None):
    '''
        Qs: q, chi, Qhat, chihat, rtilde, xi, kappa, Gamma
    '''
    torch.use_deterministic_algorithms(True)
    g2 = g**2
    phi = getattr(torch, nonLinearity)
    numUV = 500000
    
    if initialValues is not None:
        q, chi, Qhat, chihat, rtilde, xi, kappa, Gamma = [
            torch.tensor(value, device=device, dtype=dtype) 
            for value in initialValues
        ]
    else:
        q = torch.rand(1, device=device, dtype=dtype)
        chi = torch.rand(1, device=device, dtype=dtype)
        # Qhat = torch.rand(1, device=device, dtype=dtype)
        chihat = torch.tensor(-g2, device=device, dtype=dtype)
        rtilde = torch.rand(1, device=device, dtype=dtype)
        xi = -torch.rand(1, device=device, dtype=dtype)
        # kappa = torch.rand(1, device=device, dtype=dtype) * torch.sqrt(g2 * q * Qhat)
        kappa = g2 * gamma * rtilde
        Qhat = torch.rand(1, device=device, dtype=dtype) + (kappa**2) / (g2 * q)
        Gamma = -g2 * gamma * xi

    logging.debug("Iteration: 0, " + ", ".join([f"{i}: {j.item():.4g}" for i, j in zip(
        ["q", "chi", "Qhat", "chihat", "rtilde", "xi", "kappa", "Gamma"], 
        [q, chi, Qhat, chihat, rtilde, xi, kappa, Gamma])]))

    maxIteration = 1000
    
    record = Record()
    flag = False

    for t in range(maxIteration):
        sigma2 = 1 + g2 * chi
        sigma = torch.sqrt(sigma2)

        us = torch.normal(0, 1, size=(numUV, ), device=device, dtype=dtype)
        vs = torch.normal(0, 1, size=(numUV, ), device=device, dtype=dtype)

        # H0 = lambda x: - eta * (x**2) \
        #                - chihat * (phi(x)**2) \
        #                + (torch.sqrt(Qhat - (kappa**2)/(g2 * q)) * us + kappa * vs / (g * torch.sqrt(q))) * phi(x) \
        #                - ((g*torch.sqrt(q)*vs + Gamma*phi(x) - x)**2) / (2 * sigma2)
        def H0(x):
            phix = phi(x)
            return - eta * (x**2) \
                   - chihat * (phix**2) \
                   + (torch.sqrt(Qhat - (kappa**2)/(g2 * q)) * us + kappa * vs / (g * torch.sqrt(q))) * phix \
                   - ((g*torch.sqrt(q)*vs + Gamma*phix - x)**2) / (2 * sigma2)
        
        minusH0 = lambda x: -H0(x)
        try:
            start = time.time()
            xStar = findMins(minusH0, shape=(numUV, ))
            logging.debug("Cost time of golden section: {}".format(time.time() - start))
        except Exception as e:
            logging.debug(f"Error: {e}")
            break
        
        averageOverSquarePhi = torch.mean(phi(xStar)**2)
        averageOverSquareX = torch.mean(xStar**2)
        averageOverUPhi = torch.mean(us * phi(xStar))
        averageOverVX = torch.mean(vs * xStar)
        averageOverVPhi = torch.mean(vs * phi(xStar))
        averageOverPhiX = torch.mean(xStar * phi(xStar))
        
        termSqrt = torch.sqrt(Qhat * g2 * (q**2) - q * (kappa**2))
        newq = 1.0 * averageOverSquarePhi
        newchi = g * q * averageOverUPhi / termSqrt
        newQhat = g2 / (sigma2**2) * (g2 * q \
            + (Gamma**2) * averageOverSquarePhi \
            + averageOverSquareX \
            - 2 * Gamma * averageOverPhiX \
            + 2 * g * torch.sqrt(q) * Gamma * averageOverVPhi \
            - 2 * g * torch.sqrt(q) * averageOverVX)
        newChihat = g2 / (2 * sigma2) \
            - g * averageOverVX / (2 * sigma2 * torch.sqrt(q)) \
            - (kappa**2) * averageOverUPhi / (2 * g * q * termSqrt) \
            + (g * Gamma / sigma2 + kappa / (g * q)) * averageOverVPhi / (2 * torch.sqrt(q))
        newrtilde = - g * torch.sqrt(q) * averageOverVPhi / sigma2 \
            - Gamma * averageOverSquarePhi / sigma2 \
            + averageOverPhiX / sigma2
        newxi = kappa * averageOverUPhi / (g * termSqrt) \
            - averageOverVPhi / (g * torch.sqrt(q))
        newkappa = g2 * gamma * rtilde
        newGamma = - g2 * gamma * xi

        if torch.isnan(newChihat):
            logging.debug("newChihat is nan!")

        if damping > 0:
            newq = damping * q + (1 - damping) * newq
            newchi = damping * chi + (1 - damping) * newchi
            newQhat = damping * Qhat + (1 - damping) * newQhat
            newChihat = damping * chihat + (1 - damping) * newChihat
            newrtilde = damping * rtilde + (1 - damping) * newrtilde
            newxi = damping * xi + (1 - damping) * newxi
            newkappa = damping * kappa + (1 - damping) * newkappa
            newGamma = damping * Gamma + (1 - damping) * newGamma
        
        intOverLnI = torch.mean(H0(xStar))

        term1 = -(-2 * newq * newChihat + newQhat * newchi) / 2
        - (newrtilde * newGamma - newkappa * newxi)
        term2 = - g2 * gamma * xi * rtilde
        term4 = intOverLnI
        minusF = term1 + term2 + term4
        
        L2Norm = 1.0 * averageOverSquareX
        
        logging.debug(f"\033[0;34mIteration: {t+1} \033[0m, " \
                      + ", ".join([f"{i}: {j.item():.4g}" for i, j in zip(
                        ["q", "chi", "Qhat", "chihat", "rtilde", "xi", "kappa", "Gamma"], 
                        [newq, newchi, newQhat, newChihat, newrtilde, newxi, newkappa, newGamma])]) \
                      + ", minusBetaF: {:.4f}, L2Norm: {:.4f}".format(minusF.item(), L2Norm.item()))
        logging.debug(f"averageOverSquarePhi: {averageOverSquarePhi:.4g}, SquareX: {averageOverSquareX:.4g}, UPhi: {averageOverUPhi:.4g}, VX: {averageOverVX:.4g}, VPhi: {averageOverVPhi:.4g}, PhiX: {averageOverPhiX:.4g}, termSqrt: {termSqrt.item():.4g}")
        
        record.add({
            "iterationIndex": t,
            "Qs": [a.item() for a in [q, chi, Qhat, chihat, rtilde, xi, kappa, Gamma]],
            "minusF": minusF.item(),
            "energy": 0.0,
            "L2Norm": L2Norm.item()
        })

        Qs = torch.tensor([q, chi, rtilde, xi])
        newQs = torch.tensor([newq, newchi, newrtilde, newxi])
        if isConverge(Qs, newQs, damping) and t >= 100:
            flag = True
            logging.debug("Done!")
            break

        if not (torch.isfinite(newq) and torch.isfinite(newchi) and torch.isfinite(newQhat) and torch.isfinite(newChihat) and torch.isfinite(newrtilde) and torch.isfinite(newxi) and torch.isfinite(newkappa) and torch.isfinite(newGamma)):
            break

        if newq < 0:
            logging.debug("q should great than zero!")
            newq = torch.tensor(1e-8, device=device)
        if newQhat < 0:
            logging.debug("Qhat should great than zero!")
            # Qhat = torch.tensor(1e-8, device=device)
            newQhat = Qhat.clone()
        if newchi < 0:
            logging.debug("chi should great than zero!")
            newchi = torch.tensor(1e-8, device=device)
        if newkappa**2 > g2 * newq * newQhat:
            logging.debug("kappa too large!")
            newkappa = torch.sqrt(g2 * newq * newQhat * 0.1)

        q = newq.clone()
        chi = newchi.clone()
        Qhat = newQhat.clone()
        chihat = newChihat.clone()
        rtilde = newrtilde.clone()
        xi = newxi.clone()
        kappa = newkappa.clone()
        Gamma = newGamma.clone()

        if t >= 600:
            damping = 0.6
        if t >= 800:
            damping = 0.8

    return record, flag