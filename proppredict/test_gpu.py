from chemprop.train import run_training
from chemprop.args import TrainArgs
from chemprop.data.utils import get_data
import torch

print("🚦 CUDA available:", torch.cuda.is_available())
print("🚦 CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("🚦 Current CUDA device:", torch.cuda.current_device())
    print("🚦 CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

args = TrainArgs().parse_args([
    '--data_path', 'mini_dataset.csv',
    '--dataset_type', 'classification',
    '--smiles_column', 'SMILES',
    '--target_columns', 'ACTIVITY',
    '--save_dir', 'gpu_test_model',
    '--epochs', '2',
    '--gpu', '0'
])

data = get_data(path=args.data_path, args=args)
run_training(args, data)
