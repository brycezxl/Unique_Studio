import torch
a = torch.tensor([1, 2])
b = torch.tensor([1, 3])
c = [a ,b]
print(torch.add(d for d in c))