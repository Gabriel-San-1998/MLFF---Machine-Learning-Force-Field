import numpy as np
from ase.io import read
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

# === Identificar o primeiro par Pb–I ===
positions = structure.get_positions()
symbols = structure.get_chemical_symbols()
pb_index = next(i for i, s in enumerate(symbols) if s == "Pb")
i_index = next(i for i, s in enumerate(symbols) if s == "I")
print(f"Usando o par Pb (índice {pb_index}) e I (índice {i_index})")

# === Vetor de distância original ===
vec = positions[i_index] - positions[pb_index]
dist_original = np.linalg.norm(vec)
unit_vec = vec / dist_original

# === Carrega modelo SchNet treinado ===
task = AtomisticTask.load_from_checkpoint(model_path)
model = task.model
model.eval()

# === Converter ASE -> SchNetPack ===
converter = AtomsConverter(
    neighbor_list=ASENeighborList(cutoff=5.0),
    transforms=[CastTo32()]
)

# === Prever energia da estrutura original ===
inputs_orig = converter(original_structure)
with torch.no_grad():
    energy_orig = model(inputs_orig)["energy"].item()
print(f"\nEnergia original da estrutura: {energy_orig:.6f} eV\n")

# === Geração de geometrias variadas ===
scaling_factors = np.linspace(0.70, 1.4, 50)
energies = []
delta_energies = []
distances = []

for factor in scaling_factors:
    new_dist = dist_original * factor
    delta_vec = unit_vec * (new_dist - dist_original)

    mod_structure = original_structure.copy()
    mod_positions = mod_structure.get_positions()
    mod_positions[i_index] += delta_vec
    mod_structure.set_positions(mod_positions)

    # Previsão da nova energia
    inputs = converter(mod_structure)
    with torch.no_grad():
        prediction = model(inputs)
        energy = prediction["energy"].item()

    delta_E = energy - energy_orig

    distances.append(new_dist)
    energies.append(energy)
    delta_energies.append(delta_E)

    print(f"Dist: {new_dist:.3f} Å | E: {energy:.6f} eV | ΔE: {delta_E:.6f} eV")

# === Salvar para ajuste posterior ===
np.savetxt("/home/gabas/treino_saida_sem_forcas/Pb_I_E_vs_r.txt", np.column_stack([distances, energies, delta_energies]),
           header="r (Angstrom)    E (eV)    ΔE (eV)")
