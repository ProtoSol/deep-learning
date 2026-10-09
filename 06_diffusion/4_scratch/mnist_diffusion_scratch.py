"""Forward noise and a tiny UNet denoiser, no pretrained weights."""

import torch


def corrupt(clean, amount):
    noise = torch.randn_like(clean)
    amount = amount.view(-1, 1, 1, 1)
    return clean * (1 - amount) + noise * amount, noise


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.down2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.up1 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.up2 = torch.nn.Conv2d(8, 1, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)
        self.grow = torch.nn.Upsample(scale_factor=2)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        h1 = self.act(self.down1(x))
        h2 = self.act(self.down2(self.pool(h1)))
        skip = self.grow(h2)
        if skip.shape[-2:] != h1.shape[-2:]:
            skip = torch.nn.functional.interpolate(skip, size=h1.shape[-2:])
        return self.up2(self.act(self.up1(skip + h1)))


def check_claims():
    clean = torch.zeros(4, 1, 8, 8)
    clean[:, :, 2:6, 2:6] = 1.0
    amount = torch.linspace(0, 1, 4)
    noisy, noise = corrupt(clean, amount)
    assert noisy.shape == clean.shape
    assert torch.allclose(noisy[0], clean[0], atol=1e-6)  # amount 0 is the image
    assert not torch.allclose(noisy[-1], clean[-1])  # amount 1 is noise

    net = TinyUNet()
    pred = net(noisy)
    assert pred.shape == clean.shape
    loss = torch.nn.functional.mse_loss(pred, clean)
    loss.backward()
    assert net.up2.weight.grad is not None


if __name__ == "__main__":
    check_claims()
    print("mnist_diffusion_scratch ok")
