"""Toy promptable segmenter: image encoder + point prompt + mask decoder."""

import torch


SIZE = 16
BLOB = (4, 10)  # inclusive slice of the square object


def make_example(batch=4):
    image = torch.zeros(batch, 1, SIZE, SIZE)
    mask = torch.zeros(batch, 1, SIZE, SIZE)
    lo, hi = BLOB
    image[:, :, lo:hi, lo:hi] = 1.0
    mask[:, :, lo:hi, lo:hi] = 1.0
    # one positive click at the blob centre
    centre = torch.full((batch, 2), (lo + hi - 1) / 2 / (SIZE - 1))
    return image, centre, mask


def prompt_map(points, size=SIZE):
    """Put a Gaussian blob on a grid at each (row, col) in [0, 1]."""
    grid = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    yy = yy.unsqueeze(0)
    xx = xx.unsqueeze(0)
    row = points[:, 0].view(-1, 1, 1)
    col = points[:, 1].view(-1, 1, 1)
    dist2 = (yy - row) ** 2 + (xx - col) ** 2
    return torch.exp(-dist2 / (2 * 0.04 ** 2)).unsqueeze(1)


class PromptSegment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Conv2d(16, 1, 1)

    def forward(self, image, points):
        x = torch.cat([image, prompt_map(points)], dim=1)
        return self.decoder(self.encoder(x))


def check_claims():
    image, points, mask = make_example(3)
    assert image.shape == (3, 1, SIZE, SIZE)
    assert mask[0, 0, 6, 6] == 1
    assert mask[0, 0, 0, 0] == 0

    model = PromptSegment()
    logits = model(image, points)
    assert logits.shape == mask.shape

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
    loss.backward()
    assert model.decoder.weight.grad is not None

    with torch.no_grad():
        pred = (model(image, points).sigmoid() > 0.5).float()
    # after one untrained step the shape still matches; overlap is not guaranteed
    assert pred.shape == mask.shape


if __name__ == "__main__":
    check_claims()
    print("prompt_segment_scratch ok")
