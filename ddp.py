import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.net2(self.relu(self.net1))
    
def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"start running basic ddp example on rank {rank}")

    device_id = rank%torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad() #清除grad—
    outputs = ddp_model(torch.randn(20,10))

    print(device_id)
    print(outputs.shape)
    labels = torch.randn(20,5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.set()
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()

# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=127.0.0.1:29991 ddp.py