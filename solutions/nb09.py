"""Reference implementations for notebook 09 -- PyTorch and CNN architectures."""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------ architecture algebra
def conv_params(in_ch, out_ch, k, bias=True):
    """Parameter count of a Conv2d. Note: independent of H and W."""
    n = out_ch * in_ch * k * k
    return n + (out_ch if bias else 0)


def conv_flops(in_ch, out_ch, k, out_h, out_w):
    """Multiply-accumulates for one Conv2d forward pass on one image.

    Every output position costs in_ch*k*k MACs, and there are
    out_ch*out_h*out_w output positions. Unlike the parameter count, this DOES
    scale with spatial size -- which is why early high-resolution layers
    dominate compute while late layers dominate parameters.
    """
    return out_ch * out_h * out_w * in_ch * k * k


def receptive_field(specs):
    """Receptive field (in input pixels) of one unit after a stack of layers.

    specs: list of (kernel_size, stride), in order from the input.

    r_out = r_in + (k - 1) * jump ;  jump *= stride
    """
    r = 1
    jump = 1
    for k, s in specs:
        r = r + (k - 1) * jump
        jump = jump * s
    return r


def count_parameters(module, trainable_only=True):
    """Total number of (trainable) scalars in a torch module."""
    ps = module.parameters()
    return int(sum(p.numel() for p in ps
                   if (p.requires_grad or not trainable_only)))


# ------------------------------------------------------------- architectures
class ResidualBlock(nn.Module):
    """The basic ResNet block: out = ReLU(F(x) + x).

    The skip connection is the entire idea. Its gradient path is the identity,
    so d(out)/d(x) contains a +1 term that no amount of depth can attenuate --
    which is what makes 152 layers trainable.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # bias=False because the BatchNorm immediately after subtracts the mean
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # the shortcut must match shape before it can be added
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity          # add BEFORE the final ReLU
        return F.relu(out)


class SmallResNet(nn.Module):
    """A CIFAR-sized ResNet: stem, three stages, global average pool."""

    def __init__(self, num_classes=10, width=32, blocks_per_stage=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))

        stages = []
        in_ch = width
        for i, out_ch in enumerate([width, width * 2, width * 4]):
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and i > 0) else 1
                stages.append(ResidualBlock(in_ch, out_ch, stride))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # global average pooling instead of a huge flatten -> Linear
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class PlainNet(nn.Module):
    """SmallResNet with the skip connections removed -- the control experiment."""

    def __init__(self, num_classes=10, width=32, blocks_per_stage=1):
        super().__init__()
        layers = [nn.Conv2d(3, width, 3, padding=1, bias=False),
                  nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        in_ch = width
        for i, out_ch in enumerate([width, width * 2, width * 4]):
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and i > 0) else 1
                layers += [
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                              bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
                in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(self.pool(x).flatten(1))


def make_lenet(num_classes=10):
    """LeNet-5, 1998, on 32x32 input. ~62k parameters."""
    return nn.Sequential(
        nn.Conv2d(3, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
        nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
        nn.Linear(120, 84), nn.Tanh(),
        nn.Linear(84, num_classes))


def make_vgg_block(in_ch, out_ch, n_convs):
    """The VGG idea: n stacked 3x3 convs, then halve the spatial size."""
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                   nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class InceptionBlock(nn.Module):
    """GoogLeNet's parallel-branch module, with the 1x1 bottlenecks."""

    def __init__(self, in_ch, c1, c3r, c3, c5r, c5, pp):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, c1, 1), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, c3r, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(c3r, c3, 3, padding=1), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, c5r, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(c5r, c5, 5, padding=2), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                nn.Conv2d(in_ch, pp, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


# ---------------------------------------------------------------- training
def train_torch(model, Xtr, ytr, Xva, yva, epochs=10, batch_size=128, lr=1e-3,
                weight_decay=0.0, augment=False, seed=0, verbose=True):
    """Compact CPU training loop. X are float32 NCHW tensors."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    N = Xtr.shape[0]
    hist = {'loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        running = 0.0
        nb = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            if augment:
                if torch.rand(1).item() < 0.5:
                    xb = torch.flip(xb, dims=[3])
                pad = F.pad(xb, (4, 4, 4, 4), mode='reflect')
                dx, dy = torch.randint(0, 9, (2,))
                xb = pad[:, :, dy:dy + 32, dx:dx + 32]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        hist['loss'].append(running / max(nb, 1))
        hist['train_acc'].append(evaluate(model, Xtr[:2000], ytr[:2000]))
        hist['val_acc'].append(evaluate(model, Xva, yva))
        if verbose:
            print("  epoch %2d/%d  loss %.4f  train %.3f  val %.3f"
                  % (ep + 1, epochs, hist['loss'][-1], hist['train_acc'][-1],
                     hist['val_acc'][-1]))
    return hist


@torch.no_grad()
def evaluate(model, X, y, batch_size=512):
    """Accuracy. Note model.eval() -- BatchNorm must use its running stats."""
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], batch_size):
        pred = model(X[i:i + batch_size]).argmax(dim=1)
        correct += (pred == y[i:i + batch_size]).sum().item()
    return correct / float(X.shape[0])


# ---------------------------------------------------------------- test cases
CASES = {
    "conv_params": lambda: [(3, 64, 3, True), (64, 128, 3, False),
                            (3, 96, 11, True)],
    "conv_flops": lambda: [(3, 64, 3, 32, 32), (256, 512, 3, 14, 14)],
    "receptive_field": lambda: [
        ([(3, 1), (3, 1), (3, 1)],),
        ([(3, 1), (2, 2), (3, 1), (2, 2), (3, 1)],),
        ([(7, 2), (3, 2), (3, 1)],),
        ([(5, 1)],),
    ],
    "count_parameters": lambda: [
        (nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(4, 5)),),
        (nn.Linear(10, 3),),
    ],
}
