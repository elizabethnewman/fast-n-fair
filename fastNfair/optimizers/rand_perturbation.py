import torch

class RandomPerturbation:


    def __init__(self, per_sample = False):
        super(RandomPerturbation, self).__init__()
        self.per_sample = per_sample

    def solve(self, x, radius):
        delta = torch.randn_like(x)
        if self.per_sample:
            norms = torch.sqrt(torch.sum(delta**2, dim=1))
            delta = delta / torch.reshape(norms, (-1, 1)) * radius
        else:
            delta = (delta / delta.norm()) * radius
        return x + delta

if __name__ == "__main__":
    x = torch.tensor([[4.0, 5.0, 6.0], [2.0, 1.0, 8.0]])
    opt1 = RandomPerturbation();
    x_t1 = opt1.solve(x, 1.5)
    print(x_t1)
    opt2 = RandomPerturbation(per_sample = True)
    x_t2 = opt2.solve(x, 1.5)
    print(x_t2)



