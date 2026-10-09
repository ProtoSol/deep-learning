import os

import torch
from torch import nn

def corrupt(x, noise_amount):
    noise = torch.randn_like(x) * noise_amount.view(-1, 1, 1, 1)
    return x + noise


class BasicUNet(nn.Module):
    """A tiny U-Net: three down convolutions, three up convolutions, skip adds."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            if i < 2:
                h.append(x)
                x = self.downscale(x)
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                skip = h.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[-2:])
                x = x + skip
            x = self.act(layer(x))
        return x


def check_shapes():
    net = BasicUNet()
    x = torch.rand(8, 1, 28, 28)
    y = net(x)
    assert tuple(y.shape) == (8, 1, 28, 28)
    amount = torch.linspace(0, 1, x.shape[0])
    noised = corrupt(x, amount)
    assert noised.shape == x.shape
    print("demo_mnist_diffusion ok", int(sum(p.numel() for p in net.parameters())))


def train():
    from matplotlib import pyplot as plt
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torchvision.datasets.MNIST(
        root="data/", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    net = BasicUNet().to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    n_epochs = 3
    for epoch in range(n_epochs):
        for x, _y in train_dataloader:
            x = x.to(device)
            noise_amount = torch.rand(x.shape[0], device=device)
            noisy_x = corrupt(x, noise_amount)
            pred = net(noisy_x)
            loss = loss_fn(pred, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        print(f"epoch {epoch} avg loss {avg_loss:.5f}")
    os.makedirs("output", exist_ok=True)
    plt.plot(losses)
    plt.ylim(0, 0.1)
    plt.savefig("output/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    check_shapes()
    # ponytail: the 3-epoch MNIST loop downloads the dataset; set TRAIN_MNIST=1 to run it
    if os.environ.get("TRAIN_MNIST") == "1":
        train()
