from pathlib import Path

from ResNet18 import ResNet
from create_dataset import make_total_dataset
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import time
import json
import math
import random
from PIL import ImageFilter


parser = argparse.ArgumentParser(description="Barlow Twins Training")
parser.add_argument(
    "--backbone_weights",
    default=None,
    type=Path,
    metavar="DIR",
    help="backbone weight filename (under best results)",
)
parser.add_argument(
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="num workers",
)

parser.add_argument(
    "--pretext-num-classes",
    default=8,
    type=int,
    metavar="N",
    help="Pose estimation num classes",
)
parser.add_argument(
    "--epochs",
    default=1000,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--batch-size", default=2048, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument(
    "--learning-rate-weights",
    default=0.2,
    type=float,
    metavar="LR",
    help="base learning rate for weights",
)
parser.add_argument(
    "--learning-rate-biases",
    default=0.0048,
    type=float,
    metavar="LR",
    help="base learning rate for biases and batch norm parameters",
)
parser.add_argument(
    "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
)
parser.add_argument(
    "--lambd",
    default=0.0051,
    type=float,
    metavar="L",
    help="weight on off-diagonal terms",
)
parser.add_argument(
    "--projector",
    default="8192-8192-8192",
    type=str,
    metavar="MLP",
    help="projector MLP",
)
parser.add_argument(
    "--print-freq", default=100, type=int, metavar="N", help="print frequency"
)
parser.add_argument(
    "--checkpoint-dir",
    default="./checkpoint/",
    type=Path,
    metavar="DIR",
    help="path to checkpoint directory",
)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    # assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = self.args.batch_size
        self.backbone = ResNet(in_channels=1, num_classes=args.pretext_num_classes)
        self.backbone = nn.DataParallel(self.backbone)

        if args.backbone_weights is not None:
            backbone_weigths = torch.load("./best_results" / args.backbone_weights)
            self.backbone.load_state_dict(backbone_weigths)
        self.backbone.linear = nn.Identity()

        # projector
        projector_dims = list(map(int, args.projector.split("-")))
        sizes = [256] + projector_dims
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=False,
        lars_adaptation_filter=False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if not g["weight_decay_filter"] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if not g["lars_adaptation_filter"] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def main(gpu, name_lookup_table):
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    torch.cuda.device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{"params": param_weights}, {"params": param_biases}]
    model = torch.nn.DataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        parameters,
        lr=0,
        weight_decay=args.weight_decay,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.checkpoint_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    dataset = make_total_dataset(
        8,
        name_lookup_table=name_lookup_table,
        is_train=True,
        use_gen_data=True,
        is_barlow_twins=True,
        transform=Transform(),
    )
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # assert args.batch_size % args.world_size == 0
    # per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        for step, ((y1, y2, *_)) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    lr_weights=optimizer.param_groups[0]["lr"],
                    lr_biases=optimizer.param_groups[1]["lr"],
                    loss=loss.item(),
                    time=int(time.time() - start_time),
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
        # save checkpoint
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(state, args.checkpoint_dir / "checkpoint.pth")
        # save final model
        torch.save(
            model.module.backbone.state_dict(), args.checkpoint_dir / "resnet18.pth"
        )


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
    optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.RandomResizedCrop(
                #     8,
                #     scale=(0.25, 1),
                #     ratio=(1.0, 1.0),
                #     interpolation=Resampling.NEAREST,
                # ),
                # transforms.RandomApply(
                # [GaussianBlur(p=0.7)],
                # ),
                transforms.ToTensor()
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.RandomResizedCrop(
                #     8,
                #     scale=(0.25, 1),
                #     ratio=(1.0, 1.0),
                #     interpolation=Resampling.NEAREST,
                # ),
                # transforms.RandomApply(
                # [GaussianBlur(p=0.7)],
                # ),
                transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == "__main__":
    main("cuda", {"001": 0})
