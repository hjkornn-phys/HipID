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
from torch.utils import data
import numpy as np


parser = argparse.ArgumentParser(description="Barlow Twins Training")
parser.add_argument(
    "--backbone-num-blocks",
    default="2-2-2-2",
    type=str,
    metavar="NUM_BLOCKS",
    help="num_blocks of ResNet",
)

parser.add_argument(
    "--backbone_weights",
    default="resnet18-pred_pos-9classes-9.pt",
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
    default=9,
    type=int,
    metavar="N",
    help="Pose estimation num classes",
)
parser.add_argument(
    "--epochs",
    default=110,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--batch-size", default=4, type=int, metavar="N", help="mini-batch size"
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
    default="64-64",
    type=str,
    metavar="MLP",
    help="projector MLP",
)
parser.add_argument(
    "--print-freq", default=5, type=int, metavar="N", help="print frequency"
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
        self.backbone = ResNet(
            num_blocks=list(map(int, args.backbone_num_blocks.split("-"))),
            in_channels=1,
            num_classes=args.pretext_num_classes,
            channel_depth=4,
        )

        if args.backbone_weights is not None:
            backbone_weights = torch.load("./backbone_results" / args.backbone_weights)
            model_weights = {}
            for k, v in backbone_weights["model"].items():
                name = k[7:]  # remove `module.`
                model_weights[name] = v
            self.backbone.load_state_dict(model_weights)
        self.backbone.linear = nn.Identity()

        # self.backbone = nn.DataParallel(self.backbone)
        # projector
        projector_dims = list(map(int, args.projector.split("-")))
        sizes = [16] + projector_dims
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
        lr=0.001,
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

    total_ds = make_total_dataset(
        8,
        name_lookup_table=name_lookup_table,
        is_train=True,
        use_gen_data=False,
        is_barlow_twins=True,
        transform=Transform(),
    )
    train_ds, val_ds = data.random_split(
        total_ds,
        [int(len(total_ds) * 0.8), len(total_ds) - int(len(total_ds) * 0.8)],
        generator=torch.Generator().manual_seed(42),
    )

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        for step, ((y1, y2, *_)) in enumerate(loader):
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
    torch.save(model.module.backbone.state_dict(), args.checkpoint_dir / "resnet18.pth")


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
        # 0~1023 -> -1~1 sigma()
        # 0~255 sigma(0.1, 2.0)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(8, (0.25, 1), ratio=(1, 1)),
                # transforms.RandomApply(
                #     torch.nn.ModuleList(
                #         torch.rot90(self.x, 2, [1,2]),
                #     ),
                # p = 0.5),
                transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            transforms.GaussianBlur(3, sigma=(0.001, 0.005)),
                            # transforms.RandomRotation(180),
                        ]
                    ),
                    p=0.7,
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(8, (0.25, 1), ratio=(1, 1)),
                # transforms.RandomApply(
                #     torch.nn.ModuleList(
                #         torch.rot90(self.x, 2, [1,2]),
                #     ),
                # p = 0.5),
                transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                            # transforms.RandomRotation(180),
                        ]
                    ),
                    p=0.7,
                ),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class EmbExtractor(nn.Module):
    def __init__(self, gpu, extract_emb, num_classes=None):
        super().__init__()
        self.extract_emb = extract_emb
        args = parser.parse_args()
        torch.cuda.device(gpu)
        torch.backends.cudnn.benchmark = True
        self.model = BarlowTwins(args)
        self.model = torch.nn.DataParallel(self.model, device_ids=[gpu])  # module error
        bt_weights = torch.load(args.checkpoint_dir / "checkpoint.pth")["model"]
        self.model.load_state_dict(bt_weights)
        self.model.eval()
        # if extract_emb is False:
        #     self.fin_linear = nn.Linear(args.projector.split("-")[-1], num_classes)

    def forward(self, x):
        x = self.model.module.backbone(x)
        if self.extract_emb:
            return self.model.module.projector(x)
        return x


def emb_match(name_lookup_table, extract_emb, use_gen_data=False):
    PATH = str(Path.cwd())
    Path.mkdir(Path(f"{PATH}/embeddings"), exist_ok=True)
    bool_dict = {True: "train", False: "validation"}
    emb_dict = {True: "embedding", False: "RepVec"}
    emb_model = EmbExtractor("cuda", extract_emb=extract_emb)
    dataset_dict = {True: "Trainset", False: "Validationset"}
    total_ds = make_total_dataset(
        8,
        name_lookup_table=name_lookup_table,
        is_train=True,
        transform=None,
        is_barlow_twins=False,
        use_gen_data=use_gen_data,
    )
    splitted_ds = data.random_split(
        total_ds,
        [int(len(total_ds) * 0.8), len(total_ds) - int(len(total_ds) * 0.8)],
        generator=torch.Generator().manual_seed(42),
    )
    splitted_ds[0], splitted_ds[1] = splitted_ds[1], splitted_ds[0]
    test_ds = make_total_dataset(
        8,
        name_lookup_table=name_lookup_table,
        is_train=False,
        transform=None,
        is_barlow_twins=False,
        use_gen_data=False,
    )
    # Train, validation dataset(Po1~7, 9)의 representation 생성
    for train_flag in [True, False]:
        use_train = not train_flag
        concat_data = None
        concat_labels = None
        dl = data.DataLoader(splitted_ds[use_train], batch_size=128, shuffle=False)

        for batch_idx, batch in enumerate(dl):
            if use_gen_data:
                img, name_label = batch
            else:
                img, name_label, pos_label = batch
            input_img = img.to("cuda", dtype=torch.float)
            batch_embs = emb_model(input_img)
            batch_embs = batch_embs.detach().cpu().numpy()
            if batch_idx == 0:
                concat_data = batch_embs
                concat_labels = name_label
                continue
            concat_data = np.concatenate((concat_data, batch_embs), axis=0)
            concat_labels = np.concatenate((concat_labels, name_label), axis=0)
        np.savetxt(
            f"{PATH}/embeddings/{emb_dict[extract_emb]}-{len(splitted_ds[use_train])}samples-{bool_dict[use_train]}.csv",
            concat_data,
            delimiter=",",
        )
        np.savetxt(
            f"{PATH}/embeddings/{emb_dict[extract_emb]}-label-{len(splitted_ds[use_train])}samples-{bool_dict[use_train]}.csv",
            concat_labels,
            delimiter=",",
        )
    # Test dataset(Po8)의 representation 생성
    test_dl = data.DataLoader(test_ds, batch_size=128, shuffle=False)
    for batch_idx, batch in enumerate(test_dl):
        if use_gen_data:
            img, name_label = batch
        else:
            img, name_label, pos_label = batch
        input_img = img.to("cuda", dtype=torch.float)
        batch_embs = emb_model(input_img)
        batch_embs = batch_embs.detach().cpu().numpy()
        if batch_idx == 0:
            concat_data = batch_embs
            concat_labels = name_label
            continue
        concat_data = np.concatenate((concat_data, batch_embs), axis=0)
        concat_labels = np.concatenate((concat_labels, name_label), axis=0)
    np.savetxt(
        f"{PATH}/embeddings/{emb_dict[extract_emb]}-{len(test_ds)}samples-test.csv",
        concat_data,
        delimiter=",",
    )
    np.savetxt(
        f"{PATH}/embeddings/{emb_dict[extract_emb]}-label-{len(test_ds)}samples-test.csv",
        concat_labels,
        delimiter=",",
    )
    return True


if __name__ == "__main__":
    names = {
        "002": 0,
        "003": 1,
        "004": 2,
        "005": 3,
        "006": 4,
        "007": 5,
        "008": 6,
        "009": 7,
        "010": 8,
        # "011": 9,
        # "012": 10,
        # "013": 11,
        # "014": 12,
        # "015": 13,
        # "016": 14,
        # "017": 15,
        # "018": 16,
        # "019": 17,
        # "020": 18,
        # "021": 19,
        # "022": 20,
        # "023": 21,
        # "024": 22,
        # "025": 23,
        # "026": 24,
        # "027": 25,
        # "028": 26,
        # "029": 27,
        # "030": 28,
        # "031": 29,
    }
    main("cuda", names)
    embds = emb_match(name_lookup_table=names, extract_emb=False)
    # embds = emb_match(name_lookup_table=names, extract_emb = False, use_trainset=False)
    # np.savetxt('embds_representation.csv',embds[0],delimiter=",")
    # np.savetxt('embds_label.csv',embds[1],delimiter=",")
