import torch


class RandomPerturbation:


    def __init__(self):
        super(RandomPerturbation, self).__init__()

    def solve(self, x, radius):
        torch.manual_seed(42)
        delta = torch.randn_like(x)
        delta = (delta / delta.norm()) * radius
        return x + delta

if __name__ == "__main__":
    x = torch.tensor([4.0, 5.0, 6.0])
    opt = RandomPerturbation();
    x_t = opt.solve(x, 1)
    print(x_t)
