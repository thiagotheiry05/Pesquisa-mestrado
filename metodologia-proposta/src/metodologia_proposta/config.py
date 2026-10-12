from pathlib import Path


# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Diretórios principais
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
UNSEEN_DATA_DIR = DATA_DIR / "unseen"

UNSEEN_IMAGE_PATH = (
    UNSEEN_DATA_DIR
    / "tgs-grc2001_line01a-12_fin-mgr_tapeXfile15.npy"
)

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# Configurações do experimento
RANDOM_SEED = 42
PATCH_SIZE = 64

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

COLOR_TO_CLASS = {
    (0, 175, 239): "agua",
    (255, 242, 18): "bacia sedimentar",
    (132, 113, 107): "crosta continental",
    (245, 134, 52): "crosta oceanica",
    (136, 160, 172): "manto",
}

CLASS_NAMES = [
    "agua",
    "bacia sedimentar",
    "crosta continental",
    "crosta oceanica",
    "manto",
]

AUGMENTATION_TARGET_PER_CLASS = 4000

D_MODEL = 128
NUM_HEADS = 4
MLP_DIM = 256
NUM_TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 50

EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "val_loss"

# TRAIN_MODEL = False
TRAIN_MODEL = True

RECONSTRUCTION_BATCH_SIZE = 256