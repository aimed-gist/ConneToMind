import os
import gc
import time
from tqdm import tqdm
import wandb

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.utils import img_augment_high, img_augment_low, mixup, mixco_nce_loss, cosine_anneal, soft_clip_loss, topk, batchwise_cosine_similarity, log_gradient_norms, check_nan_and_log, reconstruction, plot_best_vs_gt_images, save_gt_vs_recon_images, get_unique_path, soft_cont_loss

def high_train_inference_evaluate(args, train_data, test_data, models, optimizer, lr_scheduler, metrics):
    
    # train argument
    device = args.device
    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    clip_size = args.clip_size
    prior_loss_coefficient = args.prior_loss_coefficient
    only_reconstruction = args.only_reconstruction
    experiment_name = args.experiment_name

    # test argument
    seed = args.seed
    recons_per_sample = args.recons_per_sample
    num_inference_steps = args.num_inference_steps

    scaler = GradScaler() # autocast scaler 인스턴스 생성
    
    # model 정의
    clip_extractor = models["clip"]
    diffusion_prior = models["diffusion_prior"]
    unet = models["unet"]
    vae = models["vae"]
    noise_scheduler = models["noise_scheduler"]
    optimizer = optimizer
    lr_scheduler = lr_scheduler

    # log list
    losses, lrs = [], []
    global_step=0

    progress_bar = tqdm(range(0, num_epochs), ncols=1200)
    for epoch in progress_bar:
        
        # 기본 log
        loss_nce_sum = 0.0 # Negative Contrastive Estimation loss의 누적합 -> 평균 구할 때 쓰임
        loss_prior_sum = 0.0 # prior loss의 누적합 -> 평균 구할 때 쓰임

        #### train ####
        diffusion_prior.train()
        for index, (fmri_vol, image) in enumerate(train_data): # enumerate: index와 값을 같이 반환
            # global step 계산
            global_step = epoch * len(train_data) + index
            
            # gradient 초기화
            optimizer.zero_grad() 

            # data gpu 올리기 - amp 사용
            fmri_vol = fmri_vol.to(device, non_blocking=True)   # fMRI -> GPU
            image = image.to(device, non_blocking=True)         # Image -> GPU
            
            # image augmentation
            image = img_augment_high(image)

            # epoch의 1/3 지점 까지만 mixup 사용
            if epoch < int(mixup_pct * num_epochs):
                fmri_vol, perm, betas, select = mixup(fmri_vol)
            
            # forward과정만 넣기(loss까지)
            with autocast():
                #### forward 계산 + loss 계산 ####
                # target 정의
                clip_target = clip_extractor.embed_image(image).float()
                # forward(MLP backbone) -> prior에 들어갈 embedding생성
                clip_voxels, clip_voxels_proj = diffusion_prior.voxel2clip(fmri_vol)
                clip_voxels = clip_voxels.view(len(fmri_vol),-1,clip_size) # [B, (257 * 768)] -> [B, 257, 768]

                
                # forward(Diffusion prior) -> prior loss
                prior_loss, clip_voxels_prediction = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target) # text(prediction) = fMRI, image(label) = image
                if only_reconstruction:
                    loss = prior_loss_coefficient * prior_loss
                else:
                    clip_voxels_prediction = clip_voxels_prediction / diffusion_prior.image_embed_scale # scale한 것을 원상태로 대돌려 놓음
                    # forward(MLP projector) -> contrstive loss(mixco_nce_loss + soft_clip_loss)
                    clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1) # cosine simility를 위해 미리 nomalization
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1) # cosine simility를 위해 미리 nomalization
                    # mixco_nce_loss(1/3) + soft_loss_temps(2/3)
                    if epoch < int(mixup_pct * num_epochs):
                        nce_loss = mixco_nce_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006, 
                            perm=perm, betas=betas, select=select)
                    else:
                        soft_loss_temps = cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
                        epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                        nce_loss = soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=epoch_temp)

                    # 최종 loss 정의 
                    loss = nce_loss + (prior_loss_coefficient * prior_loss)

                # NaN 체크 + 넘김
                if check_nan_and_log(global_step=global_step, fmri_vol=fmri_vol, clip_voxels=clip_voxels, loss=loss):
                    continue

            #### backward 계산 + update ####
            # gradient 계산 - amp사용
            scaler.scale(loss).backward() # amp사용

            # # gradient clipping 처리
            # log_gradient_norms(diffusion_prior, global_step)
            # scaler.unscale_(optimizer) # scaler.step(optimizer)하면 알아서 다시 scale됨
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer update - amp사용
            scaler.step(optimizer) # amp사용
            scaler.update() # amp사용

            # learning rate schedule update
            lr_scheduler.step()

            #### log ####
            # loss, lr 담아두기
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            # loss 누적 합
            # loss_nce_sum += nce_loss.item() 
            loss_prior_sum += prior_loss.item()

            logs = {
                "train/num_steps": index + 1,
                "train/lr": lrs[-1],
                "train/global_step": global_step,
                "train/epoch": epoch,
                "train/loss": losses[-1],
                "train/loss_nce": nce_loss,
                "train/loss_prior": prior_loss,

                "debug/fmri_nan": float(torch.isnan(fmri_vol).any().item()),
                "debug/fmri_min": fmri_vol.min().item(),
                "debug/fmri_max": fmri_vol.max().item(),
                "debug/voxel2clip_nan": float(torch.isnan(clip_voxels).any().item()),
                "debug/voxel2clip_min": clip_voxels.min().item(),
                "debug/voxel2clip_max": clip_voxels.max().item(),
                "debug/loss_nan": float(torch.isnan(loss).item()),

            }
            progress_bar.set_postfix(**logs) # cli에 시각화
            wandb.log(logs, step=global_step) # wandb에 시각화

        torch.cuda.empty_cache() # gpu 메모리 cache삭제
        gc.collect() # # gpu 메모리 안 쓰는거 삭제

        # if epoch % 10 == 0:
        if epoch >= 200 and epoch % 10 == 0:
            
            #### inference ####
            all_recons = []
            all_targets = []
            save_recons = {}
            best_clip_score = 0.0  

            diffusion_prior.eval()
            progress_bar = tqdm(enumerate(test_data), total=len(test_data), ncols=120)
            for index, (fmri_vol, image, low_image, image_id) in progress_bar: # enumerate: index와 값을 같이 반환
                with torch.inference_mode():
                    # noise 난수 고정
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed)

                    # data + gpu 올리기
                    fmri_vol = fmri_vol.to(device)   # fMRI -> GPU
                    image = image.to(device)         # Image -> GPU
                    low_image = low_image.to(device) 
                    
                    #### forward inference ####
                    # forward(MLP backbone) 
                    clip_voxels, clip_voxels_proj = diffusion_prior.voxel2clip(fmri_vol)
                    clip_voxels = clip_voxels.view(len(fmri_vol),-1,clip_size) # [B, (257 * 768)] -> [B, 257, 768]

                    proj_embeddings = clip_voxels_proj.cpu() # pick 할 때만 사용되어서 cpu로 내림

                    # forward(diffusion prior) 
                    clip_voxels = clip_voxels.repeat_interleave(recons_per_sample, dim=0) # [B×(recons_per_sample), 257, 768] -> ex) 1번,1번,1번,2번,2번,2번,...
                    # diffusion prior 결과
                    brain_clip_embeddings = diffusion_prior.p_sample_loop(clip_voxels.shape, 
                                            text_cond = dict(text_embed = clip_voxels), 
                                            cond_scale = 1., generator=generator) 

                    # forward(versatile diffusion) 
                    _, _, _, best_img = reconstruction(
                        brain_clip_embeddings, proj_embeddings, image,
                        clip_extractor, unet, vae, noise_scheduler,
                        seed,
                        device,
                        num_inference_steps,
                        recons_per_sample, # mindeye에서는 16개
                        inference_batch_size=fmri_vol.shape[0], # batch 중에서 몇 개만 저장할지 -> inference batch와 같이 줄 것
                        img_lowlevel = low_image,
                        guidance_scale = 3.5,
                        img2img_strength = .85,
                        plotting=False,
                    )

                    # image를 실제로 저장하기 위해 dictionary에 담아둠
                    image = image.cpu()
                    for i in range(len(image_id)):
                        img_id = image_id[i] 
                        recon_img = best_img[i]
                        gt_img = image[i]
                        save_recons[img_id] = (recon_img, gt_img)

                    # metric을 위해 저장
                    all_recons.append(best_img)       # 이미 CPU 상태
                    all_targets.append(image.cpu())        # image를 GPU에서 내려야 함 (image는 아직 GPU)

            all_recons = torch.cat(all_recons, dim=0)  # [N, 3, H, W]
            all_targets = torch.cat(all_targets, dim=0)

            torch.cuda.empty_cache() # gpu 메모리 cache삭제
            gc.collect() # gpu 메모리 안 쓰는거 삭제

            #### evaluate ####
            results = {}

            # PixCorr / SSIM
            results["PixCorr"] = metrics["pixcorr"](all_recons, all_targets)
            results["SSIM"] = metrics["ssim"](all_recons, all_targets)

            # AlexNet
            results["AlexNet_2"] = metrics["alexnet2"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["alexnet2"]["model"],
                metrics["alexnet2"]["preprocess"],
                metrics["alexnet2"]["layer"]
            )

            results["AlexNet_5"] = metrics["alexnet5"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["alexnet5"]["model"],
                metrics["alexnet5"]["preprocess"],
                metrics["alexnet5"]["layer"]
            )

            # CLIP / Inception / EfficientNet / SwAV
            results["CLIP"] = metrics["clip"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["clip"]["model"],
                metrics["clip"]["preprocess"]
            )

            results["Inception"] = metrics["inception"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["inception"]["model"],
                metrics["inception"]["preprocess"]
            )

            results["EfficientNet_B1"] = metrics["efficientnet"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["efficientnet"]["model"],
                metrics["efficientnet"]["preprocess"]
            )

            results["SwAV"] = metrics["swav"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["swav"]["model"],
                metrics["swav"]["preprocess"]
            )
            
            for name, score in results.items():
                print(f"{name:12}: {score:.4f}")
            wandb.log({f"eval/epoch{epoch}_{k}": v for k, v in results.items()}, step=global_step)



            # CLIP_2이 0.9 이상이면 모델 저장
            current_score = results.get("CLIP", 0.0)
            if current_score > best_clip_score and current_score > 0.9:
                best_clip_score = current_score  # 최대값 업데이트

                save_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, f"mindeye1_{epoch}_{experiment_name}.pt")
                save_path = get_unique_path(save_path)
                torch.save(diffusion_prior.state_dict(), save_path)
                print(f"Final model saved to {save_path} (metric > 0.9)")

                # save_recons 저장
                recons_dir = os.path.join(args.root_dir, args.code_dir, args.output_dir, "recons")
                save_gt_vs_recon_images(save_recons, recons_dir)

                # 결과를 텍스트 파일로 저장
                result_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, f"mindeye1_metrics_{epoch}_{experiment_name}.txt")
                result_path = get_unique_path(result_path)
                with open(result_path, "w") as f:
                    for name, score in results.items():
                        f.write(f"{name}: {score:.4f}\n")

            torch.cuda.empty_cache() # gpu 메모리 cache삭제
            gc.collect() # gpu 메모리 안 쓰는거 삭제
            
def low_train_inference_evaluate(args, train_data, test_data, models, optimizer, lr_scheduler, metrics):

    # train argument
    device = args.device
    experiment_name = args.experiment_name

    # test argument
    seed = args.seed

    scaler = GradScaler() # autocast scaler 인스턴스 생성

    # model 정의
    voxel2sd = models["voxel2sd"]
    cnx = models["cnx"]
    vae = models["vae"]
    noise_scheduler = models["noise_scheduler"]
    optimizer = optimizer
    lr_scheduler = lr_scheduler

    # log list
    losses = []
    lrs = []

    progress_bar = tqdm(range(0, 130), ncols=1200)
    for epoch in progress_bar:

        loss_mse_sum = 0
        loss_cont_sum = 0

        #### train ####
        voxel2sd.train()
        for index, (fmri_vol, image) in enumerate(train_data): # enumerate: index와 값을 같이 반환
            # global step 계산
            global_step = epoch * len(train_data) + index
            
            # gradient 초기화
            optimizer.zero_grad() 

            # data gpu 올리기 - amp 사용
            fmri_vol = fmri_vol.to(device, non_blocking=True)   # fMRI -> GPU
            image = image.to(device, non_blocking=True)         # Image -> GPU
            image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)

            with autocast():    
                #### forward 계산 + loss 계산 ####
                # target 정의
                image_enc_target = vae.encode(2*image-1).latent_dist.mode() * 0.18215

                # forward(voxel2sd)
                image_enc_pred, transformer_feats = voxel2sd(fmri_vol, return_transformer_feats=True)

                # contrastive loss
                mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1) # imagenet의 mean
                std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1) # imagenet의 std
                image_norm = (image - mean)/std
                image_aug = (img_augment_low(image) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)
                cont_loss = soft_cont_loss(
                    F.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    F.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    F.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.075
                )

                # mse_loss 
                mse_loss = F.l1_loss(image_enc_pred, image_enc_target)

                # 최종 loss 정의 
                loss = mse_loss/0.18215 + 0.1*cont_loss 

            # gradient 계산 - amp사용
            scaler.scale(loss).backward()

            # optimizer update - amp사용
            scaler.step(optimizer)
            scaler.update()

            # learning rate schedule update
            lr_scheduler.step()

            #### log ####
            # loss, lr 담아두기
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            # loss 누적 합
            loss_mse_sum += mse_loss.item()
            loss_cont_sum += cont_loss.item()
            
            logs = {
                "train/num_steps": index + 1,
                "train/global_step": global_step,
                "train/epoch": epoch,
                "train/lr": lrs[-1],
                
                # Loss logging
                "train/loss": losses[-1],
                "train/loss_mse": mse_loss.item(),
                "train/loss_cont": cont_loss.item(),

                # Debug: fMRI 입력
                "debug/fmri_nan": float(torch.isnan(fmri_vol).any().item()),
                "debug/fmri_min": fmri_vol.min().item(),
                "debug/fmri_max": fmri_vol.max().item(),

                # Debug: latent prediction
                "debug/image_enc_pred_nan": float(torch.isnan(image_enc_pred).any().item()),
                "debug/image_enc_pred_min": image_enc_pred.min().item(),
                "debug/image_enc_pred_max": image_enc_pred.max().item(),

                # Debug: total loss
                "debug/loss_nan": float(torch.isnan(loss).item()),

                # AMP scaling 상태 추적 (선택)
                "debug/amp_scale": scaler.get_scale(),
            }
            progress_bar.set_postfix(**logs) # cli에 시각화
            wandb.log(logs, step=global_step) # wandb에 시각화

        torch.cuda.empty_cache() # gpu 메모리 cache삭제
        gc.collect() # # gpu 메모리 안 쓰는거 삭제


        if epoch >= 50 and epoch % 10 == 0:
            #### inference ####
            all_recons = []
            all_targets = []
            save_recons = {}
            save_gts = {}
            best_alexnet2_score = 0.0  

            voxel2sd.eval()
            progress_bar = tqdm(enumerate(test_data), total=len(test_data), ncols=120)
            for index, (fmri_vol, image, _, image_id) in progress_bar: # enumerate: index와 값을 같이 반환
                with torch.inference_mode():
                    #### forward inference ####
                    # noise 난수 고정
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed)

                    # data + gpu 올리기
                    # data gpu 올리기 - amp 사용
                    fmri_vol = fmri_vol.to(device, non_blocking=True)   # fMRI -> GPU
                    image = image.to(device, non_blocking=True)         # Image -> GPU
                    image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)

                    # forward(voxel2sd)
                    image_enc_pred = voxel2sd(fmri_vol)

                    # forward(vae decoder) 
                    best_img = vae.decode(image_enc_pred.detach()/0.18215).sample / 2 + 0.5

                    # image를 실제로 저장하기 위해 dictionary에 담아둠
                    for i in range(len(image_id)):
                        img_id = image_id[i] 
                        recon_img = best_img[i]
                        save_recons[img_id] = recon_img


                    # metric을 위해 저장
                    all_recons.append(best_img)       # 이미 CPU 상태
                    all_targets.append(image.cpu())        # image를 GPU에서 내려야 함 (image는 아직 GPU)

            all_recons = torch.cat(all_recons, dim=0)  # [N, 3, H, W]
            all_targets = torch.cat(all_targets, dim=0)

            torch.cuda.empty_cache() # gpu 메모리 cache삭제
            gc.collect() # gpu 메모리 안 쓰는거 삭제

            #### evaluate ####
            results = {}

            # PixCorr / SSIM
            results["PixCorr"] = metrics["pixcorr"](all_recons, all_targets)
            results["SSIM"] = metrics["ssim"](all_recons, all_targets)

            # CLIP / AlexNet / Inception 
            results["CLIP"] = metrics["clip"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["clip"]["model"],
                metrics["clip"]["preprocess"]
            )

            results["AlexNet_2"] = metrics["alexnet2"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["alexnet2"]["model"],
                metrics["alexnet2"]["preprocess"],
                metrics["alexnet2"]["layer"]
            )

            results["AlexNet_5"] = metrics["alexnet5"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["alexnet5"]["model"],
                metrics["alexnet5"]["preprocess"],
                metrics["alexnet5"]["layer"]
            )

            results["Inception"] = metrics["inception"]["metric_fn"](
                args, all_recons, all_targets,
                metrics["inception"]["model"],
                metrics["inception"]["preprocess"]
            )
            
            for name, score in results.items():
                print(f"{name:12}: {score:.4f}")
            # wandb.log({f"eval/epoch{epoch}_{k}": v for k, v in results.items()}, step=global_step)


            # AlexNet_2이 0.65 이상이면 모델 저장
            current_score = results.get("CLIP", 0.0)
            if current_score > best_alexnet2_score and current_score > 0.7:
                best_alexnet2_score = current_score  # 최대값 업데이트

                save_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, f"mindeye1_low_{epoch}_{experiment_name}.pt")
                save_path = get_unique_path(save_path)
                torch.save(voxel2sd.state_dict(), save_path)
                print(f"Final model saved to {save_path} (metric > 0.7)")

                # save_recons 저장
                for img_id, img_tensor in save_recons.items():
                    img_tensor = img_tensor.clamp(0, 1)  # 이미지 값이 [0,1] 범위로 제한되어야 함
                    save_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, "low_recons", f"{img_id}")  # 파일 경로 설정
                    save_image(img_tensor, save_path)  # 이미지 저장

                # 결과를 텍스트 파일로 저장
                result_path = os.path.join(args.root_dir, args.code_dir, args.output_dir, f"mindeye1_low_metrics_{epoch}_{experiment_name}.txt")
                result_path = get_unique_path(result_path)
                with open(result_path, "w") as f:
                    for name, score in results.items():
                        f.write(f"{name}: {score:.4f}\n")

            torch.cuda.empty_cache() # gpu 메모리 cache삭제
            gc.collect() # gpu 메모리 안 쓰는거 삭제

