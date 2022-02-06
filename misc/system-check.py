import sys
import torch

print(f"Python version: {sys.version}")
print(f"Pytorch version: {torch.__version__}")

print(f"cuda version: {torch.version.cuda}")
print(f"cuda supported architectures {torch.cuda.get_arch_list()}")
print(f"Cuda devices: {torch.cuda.device_count()}")

for devNo in range(torch.cuda.device_count()):
    print(f"Cuda 0 device name: {torch.cuda.get_device_name(devNo)}")
    print(f"Cuda 0 memory: {torch.cuda.get_device_properties(devNo).total_memory / 1024 / 1024 / 1000}")
    print(torch.cuda.get_device_properties(devNo))

