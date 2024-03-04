import torch

t = torch.tensor([[
    [[1,2,3],
     [4,5,6]], # ch1

    [[3,2,1],
     [8,0,2]], # ch2

    [[1,2,3],
     [4,5,6]], # ch3
]])

print(torch.argmax(t[0, [0,1]], dim=0))
print(t[0, [0,1]].shape, t.shape)
