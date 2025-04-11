import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils import tokenize_prompt


class MakeSquareTransform:
    """
    Custom transformation class to make an image square with padding.
    """

    def __init__(self, background_color=(0, 0, 0)):
        """
        Initializes the transformation.

        :param background_color: The color for the padding. Default is white.
        """
        self.background_color = background_color

    def __call__(self, img, bg_color=None):
        """
        Apply the transformation.

        :param img: PIL image to be transformed.
        :return: Transformed image.
        """
        # Check if the image is already square
        if img.width == img.height:
            return img

        if bg_color is None:
            bg_color = self.background_color

        # Calculate dimensions for a square image
        square_size = max(img.width, img.height)

        # Create a new image with the desired dimensions and background color
        new_img = Image.new("RGB", (square_size, square_size), bg_color)

        # Calculate coordinates to paste the original image onto the new image
        paste_x = (square_size - img.width) // 2
        paste_y = (square_size - img.height) // 2

        # Paste the original image onto the new image
        new_img.paste(img, (paste_x, paste_y))

        return new_img


def get_ndata(rootdir, filename):
    count = 0
    filename = os.path.join(rootdir, filename)

    with open(filename, "r") as f:
        while True:
            lines = f.readline()
            if not lines:
                break

            lines = lines.strip()
            split_lines = lines.split(" ")
            path_tmp = split_lines[0]
            label_tmp = split_lines[1:]
            count += 1

    return count


def load_paths_and_labels(rootdir, filename):
    image_paths = []
    image_labels = []

    filename = os.path.join(rootdir, filename)

    with open(filename, "r") as f:
        while True:
            lines = f.readline()
            if not lines:
                break

            lines = lines.strip()
            split_lines = lines.split(" ")
            path_tmp = split_lines[0]
            label_tmp = split_lines[1:]
            is_onehot = len(label_tmp) != 1
            if not is_onehot:
                label_tmp = label_tmp[0]
            image_paths.append(path_tmp)
            image_labels.append(label_tmp)

    image_paths = np.array(image_paths)
    image_labels = np.array(image_labels, dtype=np.float32)

    return image_paths, image_labels, is_onehot


class ImageDataset(Dataset):

    def __init__(
        self,
        rootdir,
        filename="train.txt",
        path_prefix="",
        transform=None,
        target_transform=None,
    ):
        super().__init__()

        self.rootdir = rootdir
        self.filename = filename
        self.path_prefix = path_prefix

        self.image_paths, self.image_labels, self.is_onehot = load_paths_and_labels(
            self.rootdir, self.filename
        )

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.image_paths[index], self.image_labels[index]
        target = torch.tensor(target)

        img = Image.open(f"{self.path_prefix}{path}").convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.image_paths)


class DreamCreatureDataset(ImageDataset):

    def __init__(
        self,
        rootdir,
        filename="train.txt",
        path_prefix="",
        code_filename="train_caps.txt",
        num_parts=8,
        num_k_per_part=256,
        repeat=1,
        use_gt_label=False,
        bg_code=7,
        transform=None,
        target_transform=None,
        extra_data_dir=None,
        shape_attn_loss=False,
    ):
        super().__init__(rootdir, filename, path_prefix, transform, target_transform)

        if extra_data_dir is not None:
            self.extra_paths, self.extra_labels, _ = load_paths_and_labels(
                extra_data_dir, filename
            )
        else:
            self.extra_paths, self.extra_labels = [], []

        self.image_codes = np.array(open(rootdir + "/" + code_filename).readlines())
        self.num_parts = num_parts
        self.num_k_per_part = num_k_per_part
        self.repeat = repeat
        self.use_gt_label = use_gt_label
        self.bg_code = bg_code

        self.indices = np.tile(np.arange(len(self.image_paths)), self.repeat)
        if extra_data_dir is not None:
            self.indices = np.concatenate(
                [self.indices, np.arange(len(self.extra_paths))]
            )

        self.shape_attn_loss = shape_attn_loss

    def filter_by_class(self, targets):
        targets = targets.split(",")

        target_mask = np.zeros(len(self.image_labels), dtype=bool)
        for target in targets:
            target = int(target)
            target_mask |= self.image_labels == target

        self.image_paths = self.image_paths[target_mask]
        self.image_codes = self.image_codes[target_mask]
        self.image_labels = self.image_labels[target_mask]

    def set_max_samples(self, n, seed):
        np.random.seed(seed)
        rand_idx = np.arange(len(self.image_paths))
        np.random.shuffle(rand_idx)

        self.image_paths = self.image_paths[rand_idx[:n]]
        self.image_codes = self.image_codes[rand_idx[:n]]
        self.image_labels = self.image_labels[rand_idx[:n]]

    def __len__(self):
        return len(self.image_paths) * self.repeat + len(self.extra_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = self.indices[index]

        is_training_image = index < len(self.image_paths)

        if is_training_image:
            path, target = self.image_paths[index], self.image_labels[index]
            target = torch.tensor(target)
        else:
            path, _ = (
                self.extra_paths[index - len(self.image_paths)],
                self.extra_labels[index - len(self.image_paths)],
            )
            target = torch.tensor(
                self.num_k_per_part
            )  # all extra represents not exists

        img = Image.open(f"{self.path_prefix}{path}").convert("RGB")

        if is_training_image:
            cap = self.image_codes[index].strip()
        else:
            cap = " ".join([f"{i}:{target.item()}" for i in range(self.num_parts)])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        appeared = []

        code = torch.ones(self.num_parts) * self.num_k_per_part  # represents not exists
        splits = cap.strip().replace(".", "").split(" ")
        for c in splits:
            idx, intval = c.split(":")
            appeared.append(int(idx))
            if (self.use_gt_label and self.bg_code != int(idx)) or self.shape_attn_loss:
                code[int(idx)] = target
            else:
                code[int(idx)] = int(intval)

        example = {
            "pixel_values": img,
            "captions": cap,
            "codes": code,
            "labels": target,
            "appeared": appeared,
            "extra": torch.tensor(int(not is_training_image)),
        }

        return example


class DreamCreatureReflowDataset(ImageDataset):

    def __init__(
        self,
        rootdir,
        filename="train.txt",
        path_prefix="",
        code_filename="train_caps.txt",
        num_parts=8,
        num_k_per_part=256,
        repeat=1,
        use_gt_label=False,
        bg_code=7,
        transform=None,
        target_transform=None,
        extra_data_dir=None,
        shape_attn_loss=False,
    ):
        super().__init__(rootdir, filename, path_prefix, transform, target_transform)

        if extra_data_dir is not None:
            self.extra_paths, self.extra_labels, _ = load_paths_and_labels(
                extra_data_dir, filename
            )
        else:
            self.extra_paths, self.extra_labels = [], []

        self.image_codes = np.array(open(rootdir + "/" + code_filename).readlines())
        self.num_parts = num_parts
        self.num_k_per_part = num_k_per_part
        self.repeat = repeat
        self.use_gt_label = use_gt_label
        self.bg_code = bg_code

        self.indices = np.tile(np.arange(len(self.image_paths)), self.repeat)
        if extra_data_dir is not None:
            self.indices = np.concatenate(
                [self.indices, np.arange(len(self.extra_paths))]
            )

        self.shape_attn_loss = shape_attn_loss

    def filter_by_class(self, targets):
        targets = targets.split(",")

        target_mask = np.zeros(len(self.image_labels), dtype=bool)
        for target in targets:
            target = int(target)
            target_mask |= self.image_labels == target

        self.image_paths = self.image_paths[target_mask]
        self.image_codes = self.image_codes[target_mask]
        self.image_labels = self.image_labels[target_mask]

    def set_max_samples(self, n, seed):
        np.random.seed(seed)
        rand_idx = np.arange(len(self.image_paths))
        np.random.shuffle(rand_idx)

        self.image_paths = self.image_paths[rand_idx[:n]]
        self.image_codes = self.image_codes[rand_idx[:n]]
        self.image_labels = self.image_labels[rand_idx[:n]]

    def __len__(self):
        return len(self.image_paths) * self.repeat + len(self.extra_paths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = self.indices[index]

        is_training_image = index < len(self.image_paths)

        if is_training_image:
            path, target = self.image_paths[index], self.image_labels[index]
            target = torch.tensor(target)
        else:
            path, _ = (
                self.extra_paths[index - len(self.image_paths)],
                self.extra_labels[index - len(self.image_paths)],
            )
            target = torch.tensor(
                self.num_k_per_part
            )  # all extra represents not exists

        mv_paths = path.split(";;;")

        if is_training_image:
            cap = self.image_codes[index].strip()
        else:
            cap = " ".join([f"{i}:{target.item()}" for i in range(self.num_parts)])

        imgs = []
        for mv_path in mv_paths:
            img = Image.open(f"{self.path_prefix}{mv_path}").convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        appeared = []

        code = torch.ones(self.num_parts) * self.num_k_per_part  # represents not exists
        splits = cap.strip().replace(".", "").split(" ")
        for c in splits:
            idx, intval = c.split(":")

            appeared.append(int(idx))
            if (self.use_gt_label and self.bg_code != int(idx)) or self.shape_attn_loss:
                code[int(idx)] = target
            else:
                code[int(idx)] = int(intval)

        example = {
            "pixel_values": imgs,
            "captions": cap,
            "codes": code,
            "labels": target,
            "appeared": appeared,
            "extra": torch.tensor(int(not is_training_image)),
        }

        return example


class PartImageNet(ImageDataset):
    PARTIMAGENET_PART_CATEGORIES = [
        "Quadruped-Head",
        "Quadruped-Torso",
        "Quadruped-Foot",
        "Quadruped-Tail",
        "Biped-Head",
        "Biped-Torso",
        "Biped-Head",
        "Biped-Foot",
        "Biped-Tail",
        "Fish-Head",
        "Fish-Torso",
        "Fish-Fin",
        "Fish-Tail",
        "Bird-Head",
        "Bird-Torso",
        "Bird-Wing",
        "Bird-Foot",
        "Bird-Tail",
        "Snake-Head",
        "Snake-Torso",
        "Reptile-Head",
        "Reptile-Torso",
        "Reptile-Foot",
        "Reptile-Tail",
        "Car-Body",
        "Car-Tire",
        "Car-Side_Mirror",
        "Bicycle-Head",
        "Bicycle-Body",
        "Bicycle-Seat",
        "Bicycle-Tire",
        "Boat-Body",
        "Boat-Sail",
        "Aeroplane-Head",
        "Aeroplane-Body",
        "Aeroplane-Wing",
        "Aeroplane-Enging",
        "Aeroplane-Tail",
        "Bottle-Body",
        "Bottle-Mouth",
        "Background",
    ]

    def __init__(
        self,
        rootdir,
        filename="train.txt",
        path_prefix="",
        mask_filename="mask.txt",
        repeat=1,
        bg_code=0,
        transform=None,
        target_transform=None,
        ignore_background=False,
        filter_parts=None,
        quads_only=False,
    ):
        super().__init__(rootdir, filename, path_prefix, transform, target_transform)

        if quads_only:
            self.num_parts = 5
            self.num_k_per_part = 46 + 1
        else:
            self.num_parts = 14
            self.num_k_per_part = 159

        self.bg_code = bg_code
        self.repeat = repeat
        self.ignore_background = ignore_background
        self.quads_only = quads_only

        self.filter_parts = filter_parts

        if filter_parts is not None:
            filter_parts = list(map(int, filter_parts.split(",")))
            self.filter_parts = filter_parts
            self.num_parts = len(filter_parts)

        self.image_mask_paths = np.array(
            open(rootdir + "/" + mask_filename).readlines()
        )
        self.to_pil = transforms.ToPILImage()

    def get_part_name(self, idx):
        return self.PARTIMAGENET_PART_CATEGORIES[idx]

    def get_part_names(self):
        return self.PARTIMAGENET_PART_CATEGORIES

    def filter_by_class(self, targets):
        targets = targets.split(",")

        target_mask = np.zeros(len(self.image_labels), dtype=bool)
        for target in targets:
            target = int(target)
            target_mask |= self.image_labels == target

        self.image_paths = self.image_paths[target_mask]
        self.image_mask_paths = self.image_mask_paths[target_mask]
        self.image_labels = self.image_labels[target_mask]

    def set_max_samples(self, n, seed):
        np.random.seed(seed)
        rand_idx = np.arange(len(self.image_paths))
        np.random.shuffle(rand_idx)

        self.image_paths = self.image_paths[rand_idx[:n]]
        self.image_mask_paths = self.image_mask_paths[rand_idx[:n]]
        self.image_labels = self.image_labels[rand_idx[:n]]

    def __len__(self):
        return len(self.image_paths) * self.repeat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.image_paths)
        path, target = self.image_paths[index].strip(), self.image_labels[index]
        target = int(target)
        mask_path = self.image_mask_paths[index].strip()

        img = Image.open(f"{self.path_prefix}{path}").convert("RGB")
        mask = torch.load(f"{self.path_prefix}{mask_path}", map_location="cpu").float()

        if self.filter_parts is not None:
            mask = mask[self.filter_parts]

        appeared = torch.arange(len(mask))[mask.sum(dim=(1, 2)) > 0].tolist()

        cap = ""
        for a in appeared:
            cap += f"{a}:{target} "
        cap = cap.strip()

        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        code = torch.ones(self.num_parts) * self.num_k_per_part  # represents not exists
        splits = cap.strip().replace(".", "").split(" ")
        for c in splits:
            idx, intval = c.split(":")
            if self.ignore_background and int(idx) == self.bg_code:
                intval = 0
            code[int(idx)] = int(intval)

        example = {
            "pixel_values": img,
            "masks": [self.to_pil(mask[i]) for i in range(len(mask))],
            "captions": cap,
            "codes": code,
            "labels": target,
            "appeared": appeared,
        }

        return example


class DreamCreatureVLPartDataset(ImageDataset):

    def __init__(
        self,
        tokenizer,
        rootdir,
        filename="train.txt",
        path_prefix="",
        mask_filename="train_caps.txt",
        repeat=1,
        bg_code=5,
        transform=None,
        target_transform=None,
        use_text_encoder=False,
        skip_part_name=False,
    ):
        super().__init__(rootdir, filename, path_prefix, transform, target_transform)

        self.tokenizer = tokenizer
        self.image_masks = np.array(open(rootdir + "/" + mask_filename).readlines())
        self.repeat = repeat
        self.bg_code = bg_code
        self.skip_part_name = skip_part_name
        self.partname2id = {
            "foot": 0,
            "head": 1,
            "tail": 2,
            "body": 3,
            "wing": 4,
            "background": 5,
        }
        self.partid2name = {v: k for k, v in self.partname2id.items()}
        self.animals = [
            "Cat",
            "Dog",
            "Rabbit",
            "Hamster",
            "Goat",
            "Sheep",
            "Pig",
            "Cow",
            "Horse",
            "Donkey",
            "Deer",
            "Fox",
            "Bear",
            "Wolf",
            "Lion",
            "Tiger",
            "Leopard",
            "Cheetah",
            "Elephant",
            "Giraffe",
            "Zebra",
            "Hippo",
            "Rhinoceros",
            "Sparrow",
            "Pigeon",
            "Swan",
            "Eagle",
            "Owl",
            "Parrot",
            "Flamingo",
        ]
        self.use_text_encoder = use_text_encoder

    def filter_by_class(self, targets):
        targets = targets.split(",")

        target_mask = np.zeros(len(self.image_labels), dtype=bool)
        for target in targets:
            target = int(target)
            target_mask |= self.image_labels == target

        self.image_paths = self.image_paths[target_mask]
        self.image_masks = self.image_masks[target_mask]
        self.image_labels = self.image_labels[target_mask]

    def set_max_samples(self, n, seed):
        np.random.seed(seed)
        rand_idx = np.arange(len(self.image_paths))
        np.random.shuffle(rand_idx)

        self.image_paths = self.image_paths[rand_idx[:n]]
        self.image_masks = self.image_masks[rand_idx[:n]]
        self.image_labels = self.image_labels[rand_idx[:n]]

    def __len__(self):
        return len(self.image_paths) * self.repeat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.image_paths)
        path, target = self.image_paths[index].strip(), int(self.image_labels[index])
        name = self.animals[target]
        mask_path = self.image_masks[index].strip()

        img = Image.open(f"{self.path_prefix}{path}").convert("RGB")
        mask = torch.load(f"{self.path_prefix}{mask_path}")
        appeared = torch.arange(len(self.partname2id))[
            mask.sum(dim=(1, 2)) > 0
        ].tolist()

        if self.use_text_encoder:
            code = torch.zeros(len(self.partname2id), 77)

            for ai, a in enumerate(appeared):
                part_name = self.partid2name[a]
                if self.skip_part_name:
                    token = tokenize_prompt(self.tokenizer, self.animals[target])[0]
                else:
                    token = tokenize_prompt(
                        self.tokenizer, self.animals[target] + f" {part_name}"
                    )[0]
                # token = self.tokenizer(self.animals[target] + f' {part_name}',
                #                        replace_token=False,
                #                        )['input_ids']
                code[a] = token
        else:
            code = torch.zeros(len(self.partname2id))
            token = self.tokenizer(self.animals[target], replace_token=False)[
                "input_ids"
            ][1:-1][0]

            for ai, a in enumerate(appeared):
                code[a] = token

        # todo: load mask, then finetune
        # after this, part-aware 3d + texture optimization

        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        example = {
            "pixel_values": img,
            "labels": target,
            "seg_masks": mask,
            "appeared": appeared,
            "codes": code,
            "name": name,
            "index": index,
        }

        return example


class QuadsDataset(ImageDataset):

    def __init__(
        self,
        tokenizer,
        rootdir,
        filename="train.txt",
        path_prefix="",
        mask_filename="train_caps.txt",
        repeat=1,
        bg_code=0,  # foreground = 1
        transform=None,
        target_transform=None,
    ):
        super().__init__(rootdir, filename, path_prefix, transform, target_transform)

        self.tokenizer = tokenizer
        self.image_masks = np.array(open(rootdir + "/" + mask_filename).readlines())
        self.repeat = repeat
        self.bg_code = bg_code
        self.animals = ["cow", "giraffe", "horse", "zebra"]

    def filter_by_class(self, targets):
        targets = targets.split(",")

        target_mask = np.zeros(len(self.image_labels), dtype=bool)
        for target in targets:
            target = int(target)
            target_mask |= self.image_labels == target

        self.image_paths = self.image_paths[target_mask]
        self.image_masks = self.image_masks[target_mask]
        self.image_labels = self.image_labels[target_mask]

    def set_max_samples(self, n, seed):
        np.random.seed(seed)
        rand_idx = np.arange(len(self.image_paths))
        np.random.shuffle(rand_idx)

        self.image_paths = self.image_paths[rand_idx[:n]]
        self.image_masks = self.image_masks[rand_idx[:n]]
        self.image_labels = self.image_labels[rand_idx[:n]]

    def __len__(self):
        return len(self.image_paths) * self.repeat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.image_paths)
        path, target = self.image_paths[index].strip(), int(self.image_labels[index])
        name = self.animals[target]
        mask_path = self.image_masks[index].strip()

        img = Image.open(f"{self.path_prefix}{path}").convert("RGB")
        mask = Image.open(f"{self.path_prefix}{mask_path}").convert("RGB")
        appeared = [0, 1]  # only foreground and background

        code = torch.zeros(2, 77)

        for ai, a in enumerate(appeared):
            token = tokenize_prompt(self.tokenizer, self.animals[target])[0]
            code[a] = token

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        target = torch.tensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        example = {
            "pixel_values": img,
            "labels": target,
            "seg_masks": mask,
            "appeared": appeared,
            "codes": code,
            "name": name,
            "index": index,
        }

        return example
