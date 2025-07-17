from schnetpack.data import ASEAtomsData, AtomsDataModule
from schnetpack import transform as trn
import os

# Caminho do banco
db_path = "Downloads/flama-MA_FA_LAMMPS/salrodgom-flama-6ec3fef/tests/test_FAPbBr2I/struc/Db_estr_en_v2.db"

# Caminho explícito do split
split_file = os.path.join(os.path.dirname(db_path), "split.npz")

# Define DataModule com split_file explícito
data_module = AtomsDataModule(
    datapath=db_path,
    split_file=split_file,  # 🔥 ESSA LINHA GARANTE QUE O SPLIT FICA JUNTO DO DB
    batch_size=8,
    num_train=90,
    num_val=12,
    num_test=12,
    transforms=[
        trn.ASENeighborList(cutoff=15.0),
        trn.RemoveOffsets("energy", remove_mean=True),
        trn.CastTo32(),
    ],
    distance_unit="Ang",
    property_units={"energy": "eV"},
    num_workers=0,
)

# Remove o split antigo se quiser forçar recriação
if os.path.exists(split_file):
    os.remove(split_file)

# Prepara
data_module.prepare_data()
data_module.setup()
print("✅ DataModule carregado com sucesso!")
