import argparse
import sys
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")

    ###### Frequently changing settings ######
    parser.add_argument(
        '--mode', type=str, choices=['train', 'inference', 'evaluate'], default='train',
        help="train, inference, evaluate 구분"
    )
    parser.add_argument(
        "--num_epochs",type=int,default=250, choices=[3,240],
        help="epoch 개수",
    )
    parser.add_argument(
        "--device",type=str,default="cuda:3",
        help='device',
    )
    parser.add_argument(
        "--batch_size", type=int, default=160,
        help="Batch size(H100:160, L40:90), if benchmark L40:30",
    )
    parser.add_argument(
        "--inference_batch_size",type=int,default=25,
        help="versatile inference batch size(H100:25, L40:10)",
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=10, choices=[2,4,5,6,8,10],
        help="한 프로세스에서 몇 개 처리할지(H100:10, L40:5)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=30, choices=[4,8,10,12,16,20,30],
        help="multi-processing in dataloader-메모리와 cpu개수에 맞게 조정(H100:30, L40:10)",
    )
    parser.add_argument(
        "--only_reconstruction",action=argparse.BooleanOptionalAction,default=False,
        help="contrastive loss 사용 안하면 true",
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, choices=[1,2,4,6,8],
    )
    parser.add_argument(
        "--is_fc",action=argparse.BooleanOptionalAction,default=True,
        help="cosine matrix 사용유무",
    )
    parser.add_argument(
        "--is_position",action=argparse.BooleanOptionalAction,default=True,
        help="cosine matrix 사용유무",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="fc(1)_learnable_layer1",
        help="experiment_name 새부이름"
    )
    parser.add_argument(
        "--fc_matrix_path", type=str, default="/nas/research/03-Neural_decoding/3-bids/derivatives/raw_rest/sub-01/fc_matrix_wo_high_mean.npy",
        help="fc matrix 경로"
    )

    ###### data.py ######
    parser.add_argument(
        "--root_dir", type=str, default="/nas/research/03-Neural_decoding",
        help="Path to the BIDS root."
    )
    parser.add_argument(
        "--fmri_dir", type=str, default="3-bids/derivatives",
        help="Path to the BIDS fmri."
    )
    parser.add_argument(
        "--fmri_detail_dir", type=str, default="beta_hf_dk",
        choices=["b4_roi_zscore","beta_huggingface","beta_hf_dk"],
        help="Path to the BIDS fmri_detail."
    )
    parser.add_argument(
        "--image_dir", type=str, default="4-image/beta",
        help="Path to the BIDS image."
    )
    parser.add_argument(
        "--seed",type=int,default=42,
    )
    parser.add_argument(
        "--is_shuffle",action=argparse.BooleanOptionalAction,default=False,
        help="is shuffle",
    )
    parser.add_argument(
        "--use_low_image",action=argparse.BooleanOptionalAction,default=True,
        help="embedding 사용 유무",
    )
    parser.add_argument(
        "--world_size",type=int,default=1,
        help="is shuffle",
    )
    parser.add_argument(
        "--rank",type=int,default=0,
        help="is shuffle",
    )
    ####################

    ###### mindeye1 ######
    parser.add_argument(
        "--subj",type=int, default=1, choices=[1,2,5,7],
    )
    parser.add_argument(
        "--clip_size",type=int,default=768,
        help='clip embedding 크기',
    )
    parser.add_argument(
        "--token_size",type=int,default=257,
        help='vit patch 개수 + 1',
    )
    parser.add_argument(
        "--clip_variant",type=str,default="ViT-L/14",
        choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
        help='OpenAI clip variant',
    )
    parser.add_argument(
        "--norm_embs",action=argparse.BooleanOptionalAction,default=True,
        help="embedding 사용 유무",
    )
    parser.add_argument(
        "--hidden",action=argparse.BooleanOptionalAction,default=True,
        help="if True, CLIP embeddings will come from last hidden layer (e.g., 257x768 - Versatile Diffusion), rather than final layer",
    )
    parser.add_argument(
        "--cache_dir", type=str, default='/nas/research/03-Neural_decoding/5-mindeye_code/pretrained_cache',
        help="Where is cached Diffusion model; if not cached will download to this path",
    )
    ####################

    ###### FuncSpatial-Backbone ######
    # parser.add_argument(
    #     "--num_layers", type=int, default=2, choices=[4,6,8],
    # )
    # parser.add_argument(
    #     "--is_cosine",action=argparse.BooleanOptionalAction,default=True,
    #     help="cosine matrix 사용유무",
    # )


    ###### optimizer ######
    parser.add_argument(
        "--optimizer",type=str,default='adamw',
    )
    ####################

    ###### scheduler ######
    parser.add_argument(
        "--max_lr",type=float,default=3e-4,
    )
    parser.add_argument(
        "--scheduler_type",type=str,default='cycle',
        choices=['cycle','linear'],
    )
    ####################

    ###### trainer ######
    parser.add_argument(
        "--mixup_pct",type=float,default=0.33,
        help="BiMixCo에서 SoftCLIP로 넘어가는 epoch",
    )
    parser.add_argument(
        "--prior_loss_coefficient",type=float,default=0.3,
        help="prior loss 계수",
    )
    parser.add_argument(
        "--code_dir", type=str, default="5-mindeye_code",
        help="Path to the code."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Path to the output."
    )
    parser.add_argument(
        "--model_name", type=str, default="mindeye1",
        help="모델 이름"
    )
    parser.add_argument(
        "--recons_per_sample", type=int, default=1,
        help= "한 frmi로 몇 개 sampling할 지"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20,
        help= "versatile inference step"
    )
    parser.add_argument(
        "--recon_name", type=str, default="mindeye1_recon",
        help="recon 캐시 이름"
    )
    parser.add_argument(
        "--metrics_name", type=str, default="mindeye1_metric",
        help="metric 결과 이름"
    )
    ####################

    # Jupyter 환경에서는 빈 리스트를 전달해야 실행이 됨
    if any("ipykernel_launcher" in arg for arg in sys.argv):
        args = parser.parse_args([])  
    else:
        args = parser.parse_args()

    return args

