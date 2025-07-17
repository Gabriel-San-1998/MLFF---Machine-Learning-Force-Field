import os
import numpy as np
import matplotlib.pyplot as plt

# Pasta onde estão os arquivos de energia
pasta_dos_txts = os.path.dirname(os.path.abspath(__file__))

# Lista todos os arquivos *_E_vs_r.txt na pasta
arquivos = [f for f in os.listdir(pasta_dos_txts) if f.endswith("_E_vs_r.txt")]

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta_dos_txts, nome_arquivo)

    # Carrega os dados
    dados = np.loadtxt(caminho)
    if dados.shape[1] < 3:
        print(f"[!] Arquivo {nome_arquivo} não tem 3 colunas. Pulando...")
        continue

    r = dados[:, 0]
    delta_E = dados[:, 2]

    # Extrai nomes dos elementos (ex: Pb_I)
    par = nome_arquivo.replace("_E_vs_r.txt", "").replace("_", "–")

    # Cria figura
    plt.figure(figsize=(10, 6))
    plt.plot(r, delta_E, marker='o', label='ΔE (eV)')

    # Eixos e título
    plt.xlabel(f"Distância {par} (Å)", fontsize=12)
    plt.ylabel("Energia relativa ΔE (eV)", fontsize=12)
    plt.title(f"Curva de Energia ΔE(r) para interação {par}", fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True)

    # Salva o gráfico
    nome_imagem = nome_arquivo.replace(".txt", ".png")
    caminho_imagem = os.path.join(pasta_dos_txts, nome_imagem)
    plt.savefig(caminho_imagem, dpi=300)
    plt.close()

    print(f"✅ Gráfico salvo como {nome_imagem}")
