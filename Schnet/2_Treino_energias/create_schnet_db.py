from schnetpack.data import ASEAtomsData
import os
import numpy as np
import ase.io

# Caminhos
list_file = "Downloads/flama-MA_FA_LAMMPS/salrodgom-flama-6ec3fef/tests/test_FAPbBr2I/struc/ReferenceDatabase/list_MAPbBr2I.dat"
root_dir = "Downloads/flama-MA_FA_LAMMPS/salrodgom-flama-6ec3fef/tests/test_FAPbBr2I"
db_path = "Db_estr_en_v2.db"

# Apaga banco anterior
if os.path.exists(db_path):
    os.remove(db_path)

# Cria o banco
ASEAtomsData.create(
    db_path,
    property_unit_dict={"energy": "eV"},
    distance_unit="Ang"
)

dataset = ASEAtomsData(db_path)

# Leitura da lista
atoms_list = []
properties_list = []
n_added = 0
n_failed = 0

with open(list_file, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines, start=1):
    tokens = line.strip().split()

    if len(tokens) < 2:
        print(f"[Linha {idx}] Ignorada: formato inválido -> {line.strip()}")
        n_failed += 1
        continue

    path_rel = tokens[0]
    try:
        energy = float(tokens[1])
    except ValueError:
        print(f"[Linha {idx}] Ignorada: energia inválida -> {tokens[1]}")
        n_failed += 1
        continue

    structure_path = os.path.join(root_dir, path_rel)

    if not os.path.exists(structure_path):
        print(f"[Linha {idx}] Ignorada: arquivo não encontrado -> {structure_path}")
        n_failed += 1
        continue

    try:
        atoms = ase.io.read(structure_path)
        atoms_list.append(atoms)
        properties_list.append({"energy": np.array([energy], dtype=np.float32)})
        n_added += 1
    except Exception as e:
        print(f"[Linha {idx}] Erro ao ler {structure_path}: {e}")
        n_failed += 1

# Adiciona tudo ao banco
dataset.add_systems(properties_list, atoms_list)

print(f"\n✅ Banco de dados criado com sucesso: {db_path}")
print(f"📦 Estruturas incluídas: {n_added}")
print(f"❌ Estruturas ignoradas: {n_failed}")
