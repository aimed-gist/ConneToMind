import os
import re
import glob
import random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torch import nn
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import nibabel as nib

import utils.utils as utils
    
class hug_TrainDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, transform):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor
       

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

class hug_TestDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, low_image_path, transform, use_low_image):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.low_image_path = low_image_path
        self.transform = transform # PIL.Image -> tensor
        self.use_low_image = use_low_image

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        low_image_path = os.path.join(self.low_image_path, self.cocoid[idx])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.use_low_image:
            low_image = Image.open(low_image_path).convert('RGB')
            if self.transform:
                low_image = self.transform(low_image)
        else:
            low_image = []

        return fmri_vol, image, low_image, self.cocoid[idx]
    
class FuncSpatial_TrainDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, transform):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.transform = transform # PIL.Image -> tensor

        self.seq_len = self.fmri.shape[1]
       

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

class FuncSpatial_TestDataset(Dataset): # ses단위로 실행
    def __init__(self, fmri_path, image_path, low_image_path, transform, use_low_image):
        self.data = np.load(fmri_path, mmap_mode='r', allow_pickle=True) # 포인터만 받아와서 메모리에 올라온 것은 아님
        self.fmri = self.data['X']
        self.cocoid = self.data['Y']
        self.image_path = image_path
        self.low_image_path = low_image_path
        self.transform = transform # PIL.Image -> tensor
        self.use_low_image = use_low_image

        self.seq_len = self.fmri.shape[1]

    def __len__(self):
        return len(self.cocoid)

    def __getitem__(self, idx): 
        # fMRI 데이터 로딩
        fmri_vol = torch.tensor(self.fmri[idx], dtype=torch.float32)

        # 이미지 로딩
        image_path = os.path.join(self.image_path, self.cocoid[idx])
        low_image_path = os.path.join(self.low_image_path, self.cocoid[idx])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.use_low_image:
            low_image = Image.open(low_image_path).convert('RGB')
            if self.transform:
                low_image = self.transform(low_image)
        else:
            low_image = []

        return fmri_vol, image, low_image, self.cocoid[idx]
    

def sub1_train_dataset_hug(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    transform = transforms.ToTensor()
    
    fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/fmri_with_labels.npz"
    image_path = f"{root_dir}/{image_dir}"
 
    train_dataset = hug_TrainDataset(fmri_path, image_path, transform)
    
    return train_dataset

def sub1_test_dataset_hug(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    code_dir = args.code_dir
    output_dir= args.output_dir
    transform = transforms.ToTensor()
    use_low_image = args.use_low_image
    
    fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/fmri_with_labels_test.npz"
    image_path = f"{root_dir}/{image_dir}"
    low_image_path = f"{root_dir}/{code_dir}/{output_dir}/low_recons"
 
    test_dataset = hug_TestDataset(fmri_path, image_path, low_image_path, transform, use_low_image)
    
    return test_dataset

def sub1_train_dataset_FuncSpatial(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    transform = transforms.ToTensor()
    
    fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/beta_hf_dk_train.npz"
    image_path = f"{root_dir}/{image_dir}"
 
    train_dataset = FuncSpatial_TrainDataset(fmri_path, image_path, transform)
    
    return train_dataset

def sub1_test_dataset_FuncSpatial(args):
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    code_dir = args.code_dir
    output_dir= args.output_dir
    transform = transforms.ToTensor()
    use_low_image = args.use_low_image
    
    fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/beta_hf_dk_test.npz"
    image_path = f"{root_dir}/{image_dir}"
    low_image_path = f"{root_dir}/{code_dir}/{output_dir}/low_recons"
 
    test_dataset = FuncSpatial_TestDataset(fmri_path, image_path, low_image_path, transform, use_low_image)
    
    return test_dataset

def train_dataset_hug2(args, subj_names):
    """
    subj_names: ['subj01', 'subj02', ...] 
    """
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir

    transform = transforms.ToTensor()

    datasets = {}
    for subj in subj_names:
        fmri_path = os.path.join(root_dir, fmri_dir, fmri_detail_dir, subj, f"{subj}_fmri_with_labels_train.npz")
        image_path = os.path.join(root_dir, image_dir)

        dataset = hug2_TrainDataset(fmri_path, image_path, transform)
        datasets[subj] = dataset

    return datasets  # {'subj01': Dataset, 'subj02': Dataset, ...}

def test_dataset_hug2(args, subj_names):
    """
    subj_names: ['subj01', 'subj02', ...] 
    """
    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir

    transform = transforms.ToTensor()

    datasets = {}
    for subj in subj_names:
        fmri_path = os.path.join(root_dir, fmri_dir, fmri_detail_dir, subj, f"{subj}_fmri_with_labels_test.npz")
        image_path = os.path.join(root_dir, image_dir)

        dataset = hug2_TestDataset(fmri_path, image_path, transform)
        datasets[subj] = dataset

    return datasets  # {'subj01': Dataset, 'subj02': Dataset, ...}

def get_dataloader_hug2(args):
    train_loaders={}
    inference_loaders={}

    if args.mode == 'train':
        train_datasets = train_dataset_hug2(args)

        for subj in train_datasets.keys():
            train_loader = DataLoader(train_datasets[subj], batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True)
            train_loaders[subj] = train_loader

        return train_loaders
    
    if args.mode == 'inference':
        inference_datasets = test_dataset_hug2(args)

        for subj in inference_datasets.keys():
            inference_loaders = DataLoader(inference_datasets[subj], batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True)
            inference_loaders[subj] = inference_loaders

        return inference_loaders

def get_dataloader(args):

    # 제거할 index 집합
    # drop_idx = {0, 5, 8, 10, 15, 18} # low
    # drop_idx = {1,4,11,14} # high
 
    if args.mode == 'train':
        train_dataset = sub1_train_dataset_FuncSpatial(args)
        # keep_idx = [i for i in range(train_dataset.seq_len) if i not in drop_idx]
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn_factory_train(keep_idx))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=True)
        return train_loader
    
    if args.mode == 'inference':
        test_dataset = sub1_test_dataset_FuncSpatial(args)
        # keep_idx = [i for i in range(test_dataset.seq_len) if i not in drop_idx]
        # test_loader = DataLoader(test_dataset, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=args.is_shuffle, worker_init_fn=worker_init_fn, collate_fn=collate_fn_factory_test(keep_idx))
        test_loader = DataLoader(test_dataset, batch_size=args.inference_batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, persistent_workers=False, pin_memory=True, shuffle=args.is_shuffle)
        return test_loader
    
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32  # worker별 고유 seed 생성
    np.random.seed(seed)
    random.seed(seed)    

def collate_fn_factory_train(keep_idx):
    """
    keep_idx 리스트를 클로저로 잡는 collate_fn 생성기
    """
    def collate_fn(batch):
        # batch: list of tuples [(fmri, label), ...]
        fmri_batch, label_batch = zip(*batch)
        fmri_batch = torch.stack(fmri_batch, dim=0)        # [B, 20, 2056]
        fmri_batch = fmri_batch[:, keep_idx, :]            # [B, len(keep_idx), 2056]
        label_batch = torch.stack(label_batch, dim=0)
        return fmri_batch, label_batch
    return collate_fn

def collate_fn_factory_test(keep_idx):
    """
    keep_idx 리스트를 클로저로 잡는 collate_fn 생성기
    """
    def collate_fn(batch):
        # batch: list of tuples [(fmri_vol, image, low_image, image_id), ...]
        fmri_list, image_list, low_list, id_list = zip(*batch)

        # 1) fmri: [B, seq_len, feats]
        fmri_batch = torch.stack(fmri_list, dim=0)
        # 필요한 ROI만 남기기 (keep_idx 는 미리 정의된 리스트)
        fmri_batch = fmri_batch[:, keep_idx, :]

        # 2) image: [B, C, H, W] (transform이 Tensor 변환까지 했을 경우)
        image_batch = torch.stack(image_list, dim=0)

        # 3) low_image: use_low_image 여부에 따라 텐서 혹은 빈 리스트
        if isinstance(low_list[0], torch.Tensor):
            low_batch = torch.stack(low_list, dim=0)
        else:
            # low_image가 [] 로 들어오는 경우, 그냥 빈 리스트 묶음으로 전달
            low_batch = list(low_list)

        # 4) image_id: 문자열 ID 리스트
        id_batch = list(id_list)

        # 최종 반환: fmri, image, low_image, id
        return fmri_batch, image_batch, low_batch, id_batch
    return collate_fn

    