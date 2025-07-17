import os
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from schnetpack import nn, representation, atomistic, model, task
from schnetpack.train import ModelCheckpoint

# Caminho do banco e saída
db_path = "Downloads/flama-MA_FA_LAMMPS/salrodgom-flama-6ec3fef/tests/test_FAPbBr2I/struc/Db_estr_en_v2.db"
workdir = "treino_saida_sem_forcas"
os.makedirs(workdir, exist_ok=True)

# 🔧 Caminho para o arquivo de split correto
split_file = "Downloads/flama-MA_FA_LAMMPS/salrodgom-flama-6ec3fef/tests/test_FAPbBr2I/struc/split.npz"

# ⚙️ DataModule com split correto
from schnetpack.data import AtomsDataModule
from schnetpack import transform as trn

data_module = AtomsDataModule(
    datapath=db_path,
    split_file=split_file,  # <- split fixado aqui
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
data_module.prepare_data()
data_module.setup()

# 🧠 Modelo SchNet
cutoff = 15.0
n_atom_basis = 64

representation_model = representation.SchNet(
    n_atom_basis=n_atom_basis,
    n_interactions=3,
    radial_basis=nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
    cutoff_fn=nn.CosineCutoff(cutoff)
)

# Previsão de energia apenas
energy_head = atomistic.Atomwise(n_in=n_atom_basis, output_key="energy")

model_schnet = model.NeuralNetworkPotential(
    representation=representation_model,
    input_modules=[atomistic.PairwiseDistances()],
    output_modules=[energy_head]
)

# 📦 Tarefa com perda só de energia
outputs = [
    task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={"mae": torchmetrics.MeanAbsoluteError()}
    )
]

atomistic_task = task.AtomisticTask(
    model=model_schnet,
    outputs=outputs,
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
)

# Logger e checkpoint
logger = pl.loggers.TensorBoardLogger(save_dir=workdir)
checkpoint = ModelCheckpoint(
    model_path=os.path.join(workdir, "best_model"),
    save_top_k=1,
    monitor="val_loss"
)

# ⚡ Treinador sem o earling stop
#trainer = pl.Trainer(
#    default_root_dir=workdir,
#    max_epochs=150,
#    logger=logger,
#    callbacks=[checkpoint],
#    accelerator="cpu"
#)

# ⏹️ Early stopping: para evitar overfitting
early_stop_callback = EarlyStopping(
    monitor="val_loss",   # Métrica monitorada
    patience=30,          # Nº de epochs sem melhora antes de parar
    mode="min"            # Porque queremos minimizar o val_loss
)

# ⚡ Treinador atualizado
trainer = pl.Trainer(
    default_root_dir=workdir,
    max_epochs=300,
    logger=logger,
    callbacks=[checkpoint, early_stop_callback],
    accelerator="cpu"
)

# 🚀 Treinar!
trainer.fit(atomistic_task, datamodule=data_module)
