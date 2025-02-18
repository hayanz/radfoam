import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize

from .ray_utils import *

from functools import cached_property
from typing import List, Dict, Tuple


class Synthetic(Dataset):
    def __init__(
        self,
        datadir: str, split: str = "train", is_stack: bool = False, N_vis: int = -1,
        img_size: List[int] = [800, 800], resize: float = 0.5, white_bkgd: bool = True,
    ):
        assert split in ["train", "val", "test"]

        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack

        with open(f"{self.root_dir}/transforms_{self.split}.json", 'r') as f:
            self.info = json.load(f)
        if resize is not None:
            self.resize = [int(i * resize) for i in img_size]
        else:
            self.resize = img_size
        self.H, self.W = img_size
        self.F = 0.5 * self.W / np.tan(self.info["camera_angle_x"] / 2)

        self.intrinsics = torch.tensor([
            [self.F, 0, self.W / 2],
            [0, self.F, self.H / 2],
            [0, 0, 1]
        ])

        data = self.preprocess()
        if white_bkgd:
            data["rgbs"] = data["rgbs"][..., :-1] * data["rgbs"][..., -1:] + (1. - data["rgbs"][..., -1:])
        else:
            data["rgbs"] = data["rgbs"][..., :-1] * data["rgbs"][..., -1:]

        self.all_rgbs = data["rgbs"]
        self.all_poses = data["poses"]
        self.all_rays = data["rays"]
        self.cameras = data["cameras"]

        if not self.is_stack:
            self.all_rays = self.all_rays.reshape(-1, *self.all_rays.shape[-2:])
            self.all_rgbs = self.all_rgbs.reshape(-1, *self.all_rgbs.shape[-2:])

        # XXX implement from here

    def preprocess(self) -> Dict[str, torch.Tensor]:
        data = {"imgs": [], "poses": [], "rays": [], "cameras": []}
        mode = "eval" if self.split == "val" else self.split
        pbar = tqdm(self.info["frames"], desc=f"[{mode}] Loading data...")

        for frame in pbar:
            fname = os.path.join(self.root_dir, frame["file_path"] + ".png")
            img = self.load_image(fname)
            pose = np.array(frame["transform_matrix"]).astype(np.float32)
            pose = torch.from_numpy(pose)
            cam, rays = self.convert_rays(pose)
            data["imgs"].append(img)
            data["poses"].append(pose)
            data["rays"].append(rays)
            data["cameras"].append(cam.view(-1, 3)[0])

        data = {k: torch.stack(v) for k, v in data.items()}

        return data

    def __getitem__(self, idx):
        if self.split == "train":   # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
            }
        else:                       # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            sample = {"rays": rays, "rgbs": img}

        return sample

    @cached_property
    def transform(self):
        if self.resize is None:
            transform = Compose([ToTensor()])
        else:
            transform = Compose([Resize(tuple(self.resize)), ToTensor()])

        return transform

    def load_image(self, fname: str) -> torch.Tensor:
        img = self.transform(Image.open(fname))                     # [C, H, W]
        img = img.permute(1, 2, 0)                                  # [H, W, C]

        return img

    def convert_rays(self, pose: torch.FloatTensor) -> Tuple[torch.Tensor]:
        w_range, h_range = torch.arange(self.W), torch.arange(self.H)
        xs, ys = torch.meshgrid(w_range, h_range, indexing="xy")

        # blender projection
        dirs = torch.stack([(xs - self.W / 2) / self.F, -(ys - self.H / 2) / self.F, -torch.ones_like(xs)], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)   # [H, W, 3]
        rays_o = pose[:3, -1].expand(rays_d.shape)                  # [H, W, 3]
        rays = torch.concat([rays_o, rays_d], dim=-1)               # [H, W, 6]

        return rays_o, rays
