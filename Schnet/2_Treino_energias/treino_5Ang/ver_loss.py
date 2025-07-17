
import torch

# Caminho para o best_model
ckpt_path = "/home/gabas/treino_saida_sem_forcas/lightning_logs/version_0/checkpoints/epoch=125-step=1512.ckpt"

# Carrega o dicionário do checkpoint
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

# Mostra as métricas salvas no checkpoint
print("Métricas salvas no best_model:")
for k, v in checkpoint['callbacks'].items():
    print(f"\nCallback: {k}")
    print(v)
