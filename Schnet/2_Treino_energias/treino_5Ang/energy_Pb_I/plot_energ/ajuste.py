import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

# === Carregar dados ===
data = np.loadtxt("/home/gabas/treino_saida_sem_forcas/Pb_I_E_vs_r_2.txt", skiprows=1)
r = data[:, 0]
delta_E = data[:, 2]

# === Modelo Gaussiano invertido + linear ===
def gaussian_linear(r, A, r0, sigma, B, C):
    return A * np.exp(-((r - r0)/sigma)**2) + B * r + C

# === Chutes iniciais para o modelo Gaussiano + Linear ===
i_min = np.argmin(delta_E)
A0 = -abs(delta_E[i_min])
r0 = r[i_min]
sigma0 = 0.3
B0 = 0.0
C0 = 0.0

# === Ajuste Gaussiano + linear ===
gauss_params, _ = curve_fit(gaussian_linear, r, delta_E, p0=[A0, r0, sigma0, B0, C0], maxfev=1300)

# === Ajustes Polinomiais ===
#poly4 = Polynomial.fit(r, delta_E, 4).convert()
poly5 = Polynomial.fit(r, delta_E, 5).convert()

# === Geração de curvas ajustadas ===
r_fit = np.linspace(min(r), max(r), 300)
E_gauss = gaussian_linear(r_fit, *gauss_params)
#E_poly4 = poly4(r_fit)
E_poly5 = poly5(r_fit)

# === Plotagem ===
plt.figure(figsize=(10, 6))
plt.plot(r, delta_E, 'ko', label="ΔE dados", markersize=4)
plt.plot(r_fit, E_gauss, 'g--', label="Gaussiano + linear")
#plt.plot(r_fit, E_poly4, 'b-.', label="Polinômio grau 4")
plt.plot(r_fit, E_poly5, 'm:', label="Polinômio grau 5")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel("Distância Pb–I (Å)", fontsize=12)
plt.ylabel("Energia relativa ΔE (eV)", fontsize=12)
plt.title("Ajustes à curva Pb–I", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Salvar parâmetros ===
with open("/home/gabas/treino_saida_sem_forcas/plot_energ/parametros_ajustes_comparativos_3.txt", "w") as f:
    f.write("# Ajustes para interação Pb–I \n\n")

    f.write("### Gaussiano + Linear ###\n")
    f.write(f"A      = {gauss_params[0]:.6f} eV\n")
    f.write(f"r0     = {gauss_params[1]:.6f} Å\n")
    f.write(f"sigma  = {gauss_params[2]:.6f} Å\n")
    f.write(f"B      = {gauss_params[3]:.6f} eV/Å\n")
    f.write(f"C      = {gauss_params[4]:.6f} eV\n\n")

 #   f.write("### Polinômio grau 4 ###\n")
 #   for i, coef in enumerate(poly4.coef[::-1]):
 #       f.write(f"a{i} = {coef:.6f}\n")
 #   f.write("\n")

    f.write("### Polinômio grau 5 ###\n")
    for i, coef in enumerate(poly5.coef[::-1]):
        f.write(f"a{i} = {coef:.6f}\n")

print("✅ Ajuste (sem Morse) concluído. Parâmetros salvos em 'parametros_ajustes_comparativos_2.txt'")
