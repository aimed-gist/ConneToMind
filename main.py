import os
import gc
import atexit
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import wandb

from args import parse_args
from data import get_dataloader, sub1_train_dataset, sub1_train_dataset_hug, sub1_train_dataset_FuncSpatial
from mindeye1 import get_model_highlevel, get_model_lowlevel, get_model_highlevel_FuncSpatial
from optimizers import get_optimizer_highlevel, get_optimizer_lowlevel
from schedulers import get_scheduler
from metrics import get_metric
from trainer import train, inference, evaluate, retrieval_evaluate
from all_trainer import high_train_inference_evaluate, low_train_inference_evaluate
from utils import seed_everything, get_unique_path, save_gt_vs_recon_images


def main_high_all():

    args = parse_args()
    seed_everything(args.seed)

    #load data
    train_data = get_dataloader(args)
    setattr(args, 'mode', 'inference')
    test_data = get_dataloader(args)

    # model 정의
    models = get_model_highlevel_FuncSpatial(args) 
    model_bundle = {
        "clip": models["clip"].to(args.device),
        "diffusion_prior": models["diffusion_prior"].to(args.device),
        #inference only
        "unet": models["unet"].to(args.device),
        "vae": models["vae"].to(args.device),
        "noise_scheduler": models["noise_scheduler"],
    }

    # optimizer 정의
    optimizer = get_optimizer_highlevel(args, model_bundle["diffusion_prior"])

    # scheduler 정의(train만 함)
    train_dataset = sub1_train_dataset_hug(args)
    num_train = len(train_dataset) 
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # metric 정의
    metrics = get_metric(args)
    metric_bundle = {
        "pixcorr": metrics["pixcorr"],
        "ssim": metrics["ssim"],
        "alexnet2": {
            "model": metrics["alexnet2"]["model"].to(args.device),
            "preprocess": metrics["alexnet2"]["preprocess"],
            "layer": metrics["alexnet2"]["layer"],
            "metric_fn": metrics["alexnet2"]["metric_fn"],
        },
        "alexnet5": {
            "model": metrics["alexnet5"]["model"].to(args.device),
            "preprocess": metrics["alexnet5"]["preprocess"],
            "layer": metrics["alexnet5"]["layer"],
            "metric_fn": metrics["alexnet5"]["metric_fn"],
        },
        "clip": {
            "model": metrics["clip"]["model"].to(args.device),
            "preprocess": metrics["clip"]["preprocess"],
            "metric_fn": metrics["clip"]["metric_fn"],
        },
        "inception": {
            "model": metrics["inception"]["model"].to(args.device),
            "preprocess": metrics["inception"]["preprocess"],
            "metric_fn": metrics["inception"]["metric_fn"],
        },
        "efficientnet": {
            "model": metrics["efficientnet"]["model"].to(args.device),
            "preprocess": metrics["efficientnet"]["preprocess"],
            "metric_fn": metrics["efficientnet"]["metric_fn"],
        },
        "swav": {
            "model": metrics["swav"]["model"].to(args.device),
            "preprocess": metrics["swav"]["preprocess"],
            "metric_fn": metrics["swav"]["metric_fn"],
        },
    }

    # wandb 적용
    wandb.login() # login
    wandb.init(project="neural_decoding_highlevel", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    high_train_inference_evaluate(args, train_data, test_data, model_bundle, optimizer, lr_scheduler, metric_bundle)

def main_low_all():

    args = parse_args()
    seed_everything(args.seed, cudnn_deterministic=False)

    train_data = get_dataloader(args)
    setattr(args, 'mode', 'inference')
    test_data = get_dataloader(args)

    models = get_model_lowlevel(args) 
    model_bundle = {
        "voxel2sd": models["voxel2sd"].to(args.device),
        "cnx": models["cnx"].to(args.device),
        "vae": models["vae"].to(args.device),
        "noise_scheduler": models["noise_scheduler"],
    }

    # optimizer 정의
    optimizer = get_optimizer_lowlevel(args, model_bundle["voxel2sd"])

    # scheduler 정의(train만 함)
    train_dataset = sub1_train_dataset_hug(args)
    num_train = len(train_dataset) 
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # metric 정의
    metrics = get_metric(args)
    metric_bundle = {
        "pixcorr": metrics["pixcorr"],
        "ssim": metrics["ssim"],
        "alexnet2": {
            "model": metrics["alexnet2"]["model"].to(args.device),
            "preprocess": metrics["alexnet2"]["preprocess"],
            "layer": metrics["alexnet2"]["layer"],
            "metric_fn": metrics["alexnet2"]["metric_fn"],
        },
        "alexnet5": {
            "model": metrics["alexnet5"]["model"].to(args.device),
            "preprocess": metrics["alexnet5"]["preprocess"],
            "layer": metrics["alexnet5"]["layer"],
            "metric_fn": metrics["alexnet5"]["metric_fn"],
        },
        "clip": {
            "model": metrics["clip"]["model"].to(args.device),
            "preprocess": metrics["clip"]["preprocess"],
            "metric_fn": metrics["clip"]["metric_fn"],
        },
        "inception": {
            "model": metrics["inception"]["model"].to(args.device),
            "preprocess": metrics["inception"]["preprocess"],
            "metric_fn": metrics["inception"]["metric_fn"],
        },
        "efficientnet": {
            "model": metrics["efficientnet"]["model"].to(args.device),
            "preprocess": metrics["efficientnet"]["preprocess"],
            "metric_fn": metrics["efficientnet"]["metric_fn"],
        },
        "swav": {
            "model": metrics["swav"]["model"].to(args.device),
            "preprocess": metrics["swav"]["preprocess"],
            "metric_fn": metrics["swav"]["metric_fn"],
        },
    }

    # wandb 적용
    wandb.login() # login
    wandb.init(project="neural_decoding_lowlevel", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    low_train_inference_evaluate(args, train_data, test_data, model_bundle, optimizer, lr_scheduler, metric_bundle)

def main_high_all_FuncSpatial():
    args = parse_args()

    # data loader
    seed_everything(args.seed) # 시드 고정
    train_data = get_dataloader(args)
    setattr(args, 'mode', 'inference')
    test_data = get_dataloader(args)

    # model 정의
    models = get_model_highlevel_FuncSpatial(args) 
    model_bundle = {
        "clip": models["clip"].to(args.device),
        "diffusion_prior": models["diffusion_prior"].to(args.device),
        #inference only
        "unet": models["unet"].to(args.device),
        "vae": models["vae"].to(args.device),
        "noise_scheduler": models["noise_scheduler"],
    }

    # optimizer 정의
    optimizer = get_optimizer_highlevel(args, model_bundle["diffusion_prior"])

    # scheduler 정의(train only)
    #train_dataset = sub1_train_dataset_FuncSpatial(args)
    #num_train = len(train_dataset)
    num_train = len(train_data.dataset)
    lr_scheduler = get_scheduler(args, optimizer, num_train)

    # metric 정의
    metrics = get_metric(args)
    metric_bundle = {
        "pixcorr": metrics["pixcorr"],
        "ssim": metrics["ssim"],
        "alexnet2": {
            "model": metrics["alexnet2"]["model"].to(args.device),
            "preprocess": metrics["alexnet2"]["preprocess"],
            "layer": metrics["alexnet2"]["layer"],
            "metric_fn": metrics["alexnet2"]["metric_fn"],
        },
        "alexnet5": {
            "model": metrics["alexnet5"]["model"].to(args.device),
            "preprocess": metrics["alexnet5"]["preprocess"],
            "layer": metrics["alexnet5"]["layer"],
            "metric_fn": metrics["alexnet5"]["metric_fn"],
        },
        "clip": {
            "model": metrics["clip"]["model"].to(args.device),
            "preprocess": metrics["clip"]["preprocess"],
            "metric_fn": metrics["clip"]["metric_fn"],
        },
        "inception": {
            "model": metrics["inception"]["model"].to(args.device),
            "preprocess": metrics["inception"]["preprocess"],
            "metric_fn": metrics["inception"]["metric_fn"],
        },
        "efficientnet": {
            "model": metrics["efficientnet"]["model"].to(args.device),
            "preprocess": metrics["efficientnet"]["preprocess"],
            "metric_fn": metrics["efficientnet"]["metric_fn"],
        },
        "swav": {
            "model": metrics["swav"]["model"].to(args.device),
            "preprocess": metrics["swav"]["preprocess"],
            "metric_fn": metrics["swav"]["metric_fn"],
        },
    }

    # wandb 적용
    wandb.login() # login
    wandb.init(project="neural_decoding_highlevel", name=f"run-{wandb.util.generate_id()}", config=vars(args)) # init

    high_train_inference_evaluate(args, train_data, test_data, model_bundle, optimizer, lr_scheduler, metric_bundle)

def retrieval():
    args = parse_args()
    setattr(args, 'mode', 'inference')
    fwds, bwds = [], []
    
    for i in range(30):
    
        test_data = get_dataloader(args)

        # model 정의
        # models = get_model_highlevel(args)
        models = get_model_highlevel_FuncSpatial(args) 
        model_bundle = {
            "clip": models["clip"].to(args.device),
            "diffusion_prior": models["diffusion_prior"].to(args.device),
        }
        output_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, 'recon_metric', 'mindeye1_220_fc(0.7)_learnable_layer1.pt')

        fwd, bwd = retrieval_evaluate(args, test_data, model_bundle, output_path)

        fwds=np.append(fwds, fwd)
        bwds=np.append(bwds, bwd)

    percent_fwd = np.mean(fwds)
    percent_bwd = np.mean(bwds)

    print(f"fwd percent_correct: {percent_fwd:.4f}")
    print(f"bwd percent_correct: {percent_bwd:.4f}")
    
    result_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, f"mindeye1_retrieval_metrics_{args.experiment_name}.txt")
    result_path = get_unique_path(result_path)
    with open(result_path, "w") as f:
        f.write(f"Forward Retrieval Accuracy: {percent_fwd:.4f}\n")
        f.write(f"Backward Retrieval Accuracy: {percent_bwd:.4f}\n")


if __name__ == "__main__":
    # main_high_all()
    # main_low_all()
    main_high_all_FuncSpatial()
    # retrieval()