import numpy as np
from pathlib import Path
from torch.utils import data
from torchvision import transforms
import torch


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        name_lookup_table,
        exts=["csv"],
        use_name_as_label=False,
        is_gen_data=False,
        transform=None,
        is_barlow_twins=False,
    ):
        super().__init__()
        self.folder = folder  # ./8x8_Data/name/images
        self.name = folder.split("\\")[-2]
        self.is_gen_data = is_gen_data
        self.transform = transform
        self.is_barlow_twins = is_barlow_twins
        if use_name_as_label:
            self.name_lookup_table = {name: name for name in name_lookup_table}
        else:
            self.name_lookup_table = name_lookup_table
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]
        if self.is_gen_data is False:
            self.pos_labels = torch.as_tensor(
                np.array([(int(str(p).split("\\")[-1][2]) - 1) for p in self.paths]),
                dtype=torch.long,
            )
        if self.transform is None:  # 훈련
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(3, sigma=(0.1, 0.2))], p=0.9
                    ),
                    transforms.RandomErasing(
                        p=0.8, scale=(9 / 64, 0.25), ratio=(1, 1), value=-1
                    ),
                ]
            )

    def __len__(self):
        return len(self.paths)

    # import copy
    def __getitem__(self, index):
        path = self.paths[index]
        img = np.genfromtxt(path, dtype=np.int16, delimiter=",")  # 0~1023
        img = np.divide(img, 1023).astype(np.float32)  # 0~1
        img = normalize_to_neg_one_to_one(img)
        if self.is_barlow_twins:
            y1, y2 = self.transform(img)
            p, q = torch.rand(2)
            if p >= 0.5:
                y1 = torch.rot90(y1, 2, (1, 2))
            if q >= 0.5:
                y2 = torch.rot90(y2, 2, (1, 2))
            return (
                y1,
                y2,
                self.name_lookup_table[self.name],
            )  # 0~1 로 normalize
        else:
            img = self.transform(img)
            p = torch.rand(1)
            if p >= 0.5:
                img = torch.rot90(img, 2, (1, 2))

            if self.is_gen_data:
                return (
                    img,
                    self.name_lookup_table[self.name],
                    torch.tensor(9),  # pos 예측에 사용불가
                )  # 0~1 로 normalize
            else:
                pos_label = self.pos_labels[index]
                return (
                    img,
                    self.name_lookup_table[self.name],
                    pos_label,
                )  # 0~1 로 normalize


def make_total_dataset(
    image_size,
    name_lookup_table,
    folder=None,
    exts=["csv"],
    use_name_as_label=False,
    is_train=True,
    use_gen_data=False,
    transform=None,
    is_barlow_twins=False,
):
    PATH = str(Path.cwd())
    Bool_dict = {True: "images", False: "test_images"}
    if folder is None:
        folder = PATH
    if is_train is False:  # 테스트
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    names = name_lookup_table.keys()
    total_dataset = data.ConcatDataset(
        [
            Dataset(
                f"{folder}\8x8_Data\{name}\{Bool_dict[is_train]}",
                image_size=image_size,
                name_lookup_table=name_lookup_table,
                use_name_as_label=use_name_as_label,
                exts=exts,
                transform=transform,
                is_barlow_twins=is_barlow_twins,
            )
            for name in names
        ]
    )
    if use_gen_data and is_train:
        gen_dataset = data.ConcatDataset(
            [
                Dataset(
                    f"{folder}\8x8_Data\{name}\gen_images",
                    image_size=image_size,
                    name_lookup_table=name_lookup_table,
                    use_name_as_label=use_name_as_label,
                    exts=exts,
                    is_gen_data=True,
                    transform=transform,
                    is_barlow_twins=is_barlow_twins,
                )
                for name in names
            ]
        )
        total_dataset = data.ConcatDataset([total_dataset, gen_dataset])

    return total_dataset


if __name__ == "__main__":
    name_dict = {
        "002": 0,
        "003": 1,
        "004": 2,
        "005": 3,
        "006": 4,
        "007": 5,
        "008": 6,
        "009": 7,
        "010": 8,
        "011": 9,
        "012": 10,
        "013": 11,
        "014": 12,
        "015": 13,
        "016": 14,
        "017": 15,
        "018": 16,
        "019": 17,
        "020": 18,
        "021": 19,
        "022": 20,
    }
    total_dataset = make_total_dataset(
        8,
        name_dict,
        folder=None,
        exts=["csv"],
        use_name_as_label=False,
        is_train=True,
        use_gen_data=True,
        is_barlow_twins=False,
    )
    print(total_dataset[200], len(total_dataset))
