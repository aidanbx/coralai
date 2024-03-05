import torch


t = torch.tensor([4,8,4], dtype=torch.float32)
# t_probs = torch.softmax(t, dim=0)
t_probs = t / t.sum()
selected_index = torch.multinomial(t_probs, 10, replacement=True)
random_coord_x = torch.randint(0, 300, (1,))
random_coord_y = torch.randint(0, 300, (1,))