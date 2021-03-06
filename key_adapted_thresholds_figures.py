import pandas as pd
import matplotlib.pyplot as plt

err00 = pd.read_csv('eer_00.csv', sep=',', header=0)
far00 = pd.read_csv('far_00.csv', sep=',', header=0)
frr00 = pd.read_csv('frr_00.csv', sep=',', header=0)
err01 = pd.read_csv('eer_01.csv', sep=',', header=0)
far01 = pd.read_csv('far_01.csv', sep=',', header=0)
frr01 = pd.read_csv('frr_01.csv', sep=',', header=0)
err10 = pd.read_csv('eer_10.csv', sep=',', header=0)
far10 = pd.read_csv('far_10.csv', sep=',', header=0)
frr10 = pd.read_csv('frr_10.csv', sep=',', header=0)
err11 = pd.read_csv('eer_11.csv', sep=',', header=0)
far11 = pd.read_csv('far_11.csv', sep=',', header=0)
frr11 = pd.read_csv('frr_11.csv', sep=',', header=0)
ths = pd.read_csv('threshold_00.csv', sep=',', header=0)
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# FAR e FRR - Hocquet - Janela Deslizante
plt.figure(figsize=(8, 6))
plt.plot(x, far00.iloc[:, 1], label='FAR')
plt.plot(x, frr00.iloc[:, 1], label='FRR')
plt.xlabel('Limiar Inicial')
plt.ylabel('FAR e FRR')
plt.title('Janela Deslizante, Hocquet')
plt.legend()
plt.grid(True)
plt.savefig('farxfrr_jdh.png')
plt.close()

# FAR e FRR - Hocquet - Janela Crescente
plt.figure(figsize=(8, 6))
plt.plot(x, far01.iloc[:, 1], label='FAR')
plt.plot(x, frr01.iloc[:, 1], label='FRR')
plt.xlabel('Limiar Inicial')
plt.ylabel('FAR e FRR')
plt.title('Janela Crescente - Hocquet')
plt.legend()
plt.grid(True)
plt.savefig('farxfrr_jch.png')
plt.close()

# FAR e FRR - Magalhães - Janela Deslizante
plt.figure(figsize=(8, 6))
plt.plot(x, far10.iloc[:, 1], label='FAR')
plt.plot(x, frr10.iloc[:, 1], label='FRR')
plt.xlabel('Limiar Inicial')
plt.ylabel('FAR e FRR')
plt.title('Janela Deslizante - Magalhães')
plt.legend()
plt.grid(True)
plt.savefig('farxfrr_jdm.png')
plt.close()

# FAR e FRR - Magalhães - Janela Crescente
plt.figure(figsize=(8, 6))
plt.plot(x, far11.iloc[:, 1], label='FAR')
plt.plot(x, frr11.iloc[:, 1], label='FRR')
plt.xlabel('Limiar Inicial')
plt.ylabel('FAR e FRR')
plt.title('Janela Crescente - Magalhães')
plt.legend()
plt.grid(True)
plt.savefig('farxfrr_jcm.png')
plt.close()

# ERR
plt.figure(figsize=(8, 6))
plt.plot(far00.iloc[:, 1], frr00.iloc[:, 1], label='Deslizante, Hocquet, ERR = 0.4573')
plt.plot(far01.iloc[:, 1], frr01.iloc[:, 1], label='Crescente, Hocquet, ERR = 0.3431')
plt.plot(far10.iloc[:, 1], frr10.iloc[:, 1], label='Deslizante, Magalhães, ERR = 0.4063')
plt.plot(far11.iloc[:, 1], frr11.iloc[:, 1], label='Crescente, Magalhães, ERR = 0.2419')
plt.xlabel('FAR')
plt.xlim([0, 1])
plt.ylabel('FRR')
plt.ylim([0, 1])
plt.title('EER')
plt.legend()
plt.grid(True)
plt.savefig('eer.png')
plt.close()
