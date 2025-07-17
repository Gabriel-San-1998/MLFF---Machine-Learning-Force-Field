import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados salvos
data = np.loadtxt("/home/gabas/treino_saida_sem_forcas/Pb_I_E_vs_r_2.txt", skiprows=1)
r = data[:, 0]
delta_E = data[:, 2]

# Plotar
plt.figure(figsize=(8, 5))
plt.plot(r, delta_E, 'o-', label="ΔE (eV)", color='tab:blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel("Distância Pb–I (Å)", fontsize=12)
plt.ylabel("Energia relativa ΔE (eV)", fontsize=12)
plt.title("Curva de Energia ΔE(r) para interação Pb–I", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
