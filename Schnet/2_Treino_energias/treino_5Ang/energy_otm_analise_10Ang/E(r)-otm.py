import numpy as np
import os
from ase.io import read
from ase.neighborlist import neighbor_list
from tqdm import tqdm
import torch
from schnetpack.interfaces import AtomsConverter
from schnetpack.task import AtomisticTask
from schnetpack.transform import ASENeighborList, CastTo32

# === Configurações ===
cif_path = "/home/gabas/Downloads/flama-MA_FA_LAMMPS/salrodgom-flama-6ec3fef/tests/test_FAPbBr2I/struc/cubic_angle_all_83.cif"
model_path = "/home/gabas/treino_saida_sem_forcas_5Ang/lightning_logs/version_0/checkpoints/epoch=125-step=1512.ckpt"
output_dir = "/home/gabas/treino_saida_sem_forcas_5Ang/energy_otm_analise_10Ang"
cutoff = 10.0
scaling_factors = np.linspace(0.7, 1.4, 50)
pares_desejados = [('Br', 'Br'), ('Br', 'C'), ('Br', 'H'), ('Br', 'I'), ('Br', 'N'), ('Br', 'Pb'),
('C', 'C'), ('C', 'H'), ('C', 'I'), ('C', 'N'), ('C', 'Pb'),
('H', 'H'), ('H', 'I'), ('H', 'N'), ('H', 'Pb'),
('I', 'I'), ('I', 'N'), ('I', 'Pb'),
('N', 'N'), ('N', 'Pb')]  
# === Preparação ===
os.makedirs(output_dir, exist_ok=True)
structure = read(cif_path)
symbols = structure.get_chemical_symbols()
positions = structure.get_positions()

# Carregar modelo treinado
task = AtomisticTask.load_from_checkpoint(model_path)
model = task.model
model.eval()
converter = AtomsConverter(neighbor_list=ASENeighborList(cutoff=cutoff), transforms=[CastTo32()])

# Energia original da estrutura
inputs_orig = converter(structure)
with torch.no_grad():
    energy_orig = model(inputs_orig)["energy"].item()

print(f"\nEnergia original da estrutura: {energy_orig:.6f} eV\n")

# Encontrar todos os pares dentro do cutoff
idx_i, idx_j, dists = neighbor_list("ijd", structure, cutoff)

# === Loop para cada par específico ===
for par in tqdm(pares_desejados, desc="Pares sendo processados"):
    a1, a2 = par
    candidatos = []

    for i, j, d in zip(idx_i, idx_j, dists):
        s1, s2 = symbols[i], symbols[j]
        if sorted([s1, s2]) == sorted([a1, a2]):
            candidatos.append((i, j, d))

    if not candidatos:
        print(f"[!] Nenhum par encontrado para {a1}-{a2} dentro do cutoff.")
        continue

    # Pegar o mais próximo
    candidatos.sort(key=lambda x: x[2])
    i1, i2, dist = candidatos[0]
    vec = positions[i2] - positions[i1]
    unit_vec = vec / np.linalg.norm(vec)

    # Variação da distância
    energies, delta_energies, distances = [], [], []

    for factor in scaling_factors:
        new_dist = dist * factor
        delta_vec = unit_vec * (new_dist - dist)

        mod_structure = structure.copy()
        mod_pos = mod_structure.get_positions()
        mod_pos[i2] += delta_vec
        mod_structure.set_positions(mod_pos)

        inputs = converter(mod_structure)
        with torch.no_grad():
            energy = model(inputs)["energy"].item()

        delta_E = energy - energy_orig
        distances.append(new_dist)
        energies.append(energy)
        delta_energies.append(delta_E)

    # Salvar resultados
    fname = f"{a1}_{a2}_E_vs_r.txt".replace("/", "-")
    np.savetxt(os.path.join(output_dir, fname),
               np.column_stack([distances, energies, delta_energies]),
               header="r (Angstrom)    E (eV)    ΔE (eV)")

    print(f"✅ Par {a1}-{a2} processado e salvo em {fname}")
