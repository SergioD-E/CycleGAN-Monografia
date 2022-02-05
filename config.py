import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "DATA/TRAIN"    # Determina el directorio con las imágenes de entrenamiento de la IA
VAL_DIR = "DATA/VAL"
BATCH_SIZE = 1
LEARNING_RATE = 5e-7
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 3
NUM_EPOCHS = 100
LOAD_MODEL = True    # Determina si se quiere cargar el progreso del entrenamiento de la IA. Este será almacenado por defecto en los archivos ".pth.tar".
SAVE_MODEL = False   # Determina si se quiere guardar el progreso en el entrenamiento de la IA.
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)