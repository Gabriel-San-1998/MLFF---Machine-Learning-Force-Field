import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
import torch
from schnetpack.interfaces import AtomsConverter
from pytorch_lightning import LightningModule
from schnetpack.task import AtomisticTask
from schnetpack.transform import ASENeighborList, CastTo32

# === Caminhos ===
cif_path = "/home/gabas/Downloads/flama-MA_FA_LAMMPS/salrodgom-flama-6ec3fef/tests/test_FAPbBr2I/struc/cubic_angle_all_83.cif"
model_path = "/home/gabas/treino_saida_sem_forcas/lightning_logs/version_0/checkpoints/epoch=125-step=1512.ckpt"

# === Carregar estrutura original ===
structure = read(cif_path)
original_structure = structure.copy()

# === Encontrar par Pb–I mais próximo ===
symbols = structure.get_chemical_symbols()
positions = structure.get_positions()
cutoff = 5.0  # corte típico para Pb–I

i_indices = [i for i, s in enumerate(symbols) if s == "I"]
pb_indices = [i for i, s in enumerate(symbols) if s == "Pb"]

# Usar vizinhança real
idx_i, idx_j, dists = neighbor_list("ijd", structure, cutoff)

# Filtrar para pares Pb–I
pb_i_pairs = [(i, j, d) for i, j, d in zip(idx_i, idx_j, dists)
              if (symbols[i] == "Pb" and symbols[j] == "I") or (symbols[i] == "I" and symbols[j] == "Pb")]

# Ordenar por distância
pb_i_pairs.sort(key=lambda x: x[2])
pb_index, i_index, dist_original = pb_i_pairs[0]

print(f"Usando o par Pb (índice {pb_index}) e I (índice {i_index})")

# === Vetor de ligação ===
vec = positions[i_index] - positions[pb_index]
unit_vec = vec / np.linalg.norm(vec)

# === Carregar modelo ===
task = AtomisticTask.load_from_checkpoint(model_path)
model = task.model
model.eval()

converter = AtomsConverter(
    neighbor_list=ASENeighborList(cutoff=5.0),
    transforms=[CastTo32()]
)

# === Energia original ===
inputs_orig = converter(original_structure)
with torch.no_grad():
    energy_orig = model(inputs_orig)["energy"].item()
print(f"\nEnergia original da estrutura: {energy_orig:.6f} eV\n")

# === Geração de geometrias modificadas ===
scaling_factors = np.linspace(0.7, 1.4, 50)
energies, delta_energies, distances = [], [], []

for factor in scaling_factors:
    new_dist = dist_original * factor
    delta_vec = unit_vec * (new_dist - dist_original)

    mod_structure = original_structure.copy()
    mod_pos = mod_structure.get_positions()
    mod_pos[i_index] += delta_vec
    mod_structure.set_positions(mod_pos)

    inputs = converter(mod_structure)
    with torch.no_grad():
        energy = model(inputs)["energy"].item()

    delta_E = energy - energy_orig
    distances.append(new_dist)
    energies.append(energy)
    delta_energies.append(delta_E)

    print(f"Dist: {new_dist:.3f} Å | E: {energy:.6f} eV | ΔE: {delta_E:.6f} eV")

# === Salvar para ajuste posterior ===
np.savetxt("/home/gabas/treino_saida_sem_forcas/Pb_I_E_vs_r_2.txt",
           np.column_stack([distances, energies, delta_energies]),
           header="r (Angstrom)    E (eV)    ΔE (eV)")

