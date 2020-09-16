import random as rd
import math
import numpy as np
from scipy.optimize import minimize

p1 = 0.10 # prababilidade de sair cara na moeda 1
p2 = 0.70 # prababilidade de sair cara na moeda 2
n = 50 # repetições do processo
m = 400 # lançamentos de moedas por iteração
resultados = np.zeros((n, m))

for i in range(n):
    if rd.random() < 0.5:
        for j in range(m):
            if rd.random() < p1:
                resultados[i][j] = 1
    else:
        for j in range(m):
            if rd.random() < p2:
                resultados[i][j] = 1

def log_verossimilhanca_negativa(theta, resultados):
    m = len(resultados[0])
    n = len(resultados)
    soma = 0
    for i in range(n):
        parcela_M1 = 1
        parcela_M2 = 1
        for j in range(m):
            if resultados[i][j] == 1:
                parcela_M1 *= theta[0]
                parcela_M2 *= theta[1]
            else:
                parcela_M1 *= (1 - theta[0])
                parcela_M2 *= (1 - theta[1])
        parcela = (parcela_M1 + parcela_M2)/2
        soma += math.log(parcela)
    
    return -soma

def loop_EM(theta, resultados):
    results = minimize(log_verossimilhanca_negativa, theta, args = resultados, bounds = ((0.1, 0.9), (0.1, 0.9)), method = 'SLSQP')
    theta = results.x
    result_ant = results.fun
    diff = result_ant - 0
    while diff > 0.000000000001:
        results = minimize(log_verossimilhanca_negativa, theta, args = resultados, bounds = ((0.1, 0.9), (0.1, 0.9)), method = 'SLSQP')
        theta = results.x
        diff = result_ant - results.fun
        result_ant = results.fun

    return results

theta = [0.3, 0.9]
print('Theta inicial:', theta)
results = loop_EM(theta, resultados)

print('Theta final:', results.x)
print('Função log-verossimilhança negativa:', results.fun)
print()

theta = [0.5, 0.5]
print('Theta inicial:', theta)
results = loop_EM(theta, resultados)

print('Theta final:', results.x)
print('Função log-verossimilhança negativa:', results.fun)
print()

theta = [0.1, 0.7]
print('Theta inicial:', theta)
results = loop_EM(theta, resultados)

print('Theta final:', results.x)
print('Função log-verossimilhança negativa:', results.fun)
print()

print('Theta usado para gerar os dados:', [p1, p2])
print('Função log-verossimilhança negativa:', log_verossimilhanca_negativa([p1, p2], resultados))
