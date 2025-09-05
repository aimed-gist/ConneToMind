import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import PIL
from functools import partial
import glob
from scipy.stats import pearsonr


# for clip
import clip # OpenAI CLIP (RN50, ViT-L/14 등)
import open_clip # OpenAI CLIP (RN50, ViT-L/14 등) + LAION-5B로 학습한 모델
from transformers import CLIPVisionModelWithProjection

# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists, RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward
from tqdm.auto import tqdm
import random
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig

# for low-level
from diffusers import DiffusionPipeline
from models.convnext import ConvnextXL
from diffusers.models.autoencoder_kl import Decoder

# get model
import utils.utils as utils
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
from trainers.schedulers import get_scheduler

class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False, hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        self.device= device
        self.hidden_state = hidden_state
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)

        # clip preporcess를 custom으로 사용함
        self.preprocess = None # object를 변수로 저장
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073]) # OpenAI CLIP이 학습한 이미지 데이터 셋의 평균
        self.std = np.array([0.26862954, 0.26130258, 0.27577711]) # OpenAI CLIP이 학습한 이미지 데이터 셋의 표준편차
        self.normalize = transforms.Normalize(self.mean, self.std) # versatile low image vae 사용하기 전 normalize
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        preprocess = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.preprocess = preprocess # object를 변수로 저장

        # embedding preprocess 변수
        self.clamp_embs = clamp_embs # embdding 후처리 유무 ex) -1.5 ~ 1.5 범위로 제한
        self.norm_embs = norm_embs # embdding normalization 유무
        
        # "RN50", "ViT-L/14", "ViT-B/32", "RN50x64" 중에 모델이 없으면 오류메세지 출력
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64" # assert문은 조건을 만족하지 않을 때 출력
        print(clip_variant, device)

        # 1번 clip 모델 load(high level)
        if clip_variant=="ViT-L/14" and hidden_state:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval() # pre-train(transformers 라이브러리의 CLIP Vision 모델)하여 frozen 함
            image_encoder = image_encoder.to(device)
            # frozen
            for param in image_encoder.parameters():
                param.requires_grad = False 
            self.image_encoder = image_encoder
        elif hidden_state:
            raise Exception("ViT-L/14에서만 hidden_state 처리 가능")
        
        # 2번 clip 모델 load
        clip_model, _ = clip.load(clip_variant, device=device)
        clip_model.eval() 
        # frozen
        for param in clip_model.parameters():
            param.requires_grad = False 
        self.clip = clip_model

    def embed_image(self, image):
        '''
        clip_emb(hidden_state 없을 때): 최종 image embedding - shape: [batch, 768]
        clip_emb(hidden_state 있을 때): 최종 image embedding - shape: [batch, 257, 768]
            Tip: sequence_length는 vit의 sequence의 patch 개수
        '''
        if self.hidden_state:
            clip_emb = self.preprocess((image).to(self.device)) # image preprocess ex) image값 -1 ~ 1
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self._versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device)) # image preprocess
            clip_emb = self.clip.encode_image(clip_emb)

        # embedding processes: clamp_embs + norm_embs
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5) # embedding 값 -1.5 ~ 1.5 범위로 제한
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
    
        return clip_emb

    def _versatile_normalize_embeddings(self, encoder_output):
        '''
        embeds(hidden_state 있을 때): image embedding normalization - shape: [batch, 257, 768]
            Tip: sequence_length는 vit의 sequence의 patch 개수
        '''
        embeds = encoder_output.last_hidden_state
        embeds = self.image_encoder.vision_model.post_layernorm(embeds)
        embeds = self.image_encoder.visual_projection(embeds)
        return embeds 
    
class BrainTransformerNetwork(nn.Module):
    def __init__(self, input_dim=2056, embed_dim=768, output_dim=257, seq_len=20, nhead=8, num_layers=8, is_position=False, is_fc=False, fc_matrix_path=""):
        super().__init__()

        # self.linear1 = nn.Sequential(
        #     nn.Linear(input_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.is_position = is_position

        # roi별로 다른 weight로 linear transformation
        self.linear1_weight = nn.Parameter(torch.empty(seq_len, input_dim, embed_dim))
        for t in range(seq_len):
            init.xavier_uniform_(self.linear1_weight[t]) # xavier_uniform 초기화
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.5)

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        #use fc prior
        if is_fc:
            encoder_layer = CustomTransformerEncoderLayer(fc_matrix_path=fc_matrix_path, d_model=embed_dim, nhead=nhead)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear2 = nn.Linear(seq_len, output_dim, bias=True)
            
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim)
        )

    def forward(self, x):
        '''
        x(FuncSpatial Backbone): fmri -> tr - shape: [batch, (257*768)]
        x(FuncSpatial projector): fmri -> tr -> mlp - shape: ([batch, 768], [batch, 257, 768])
        '''
        # linear projection
        # x = self.linear1(x)  # [B, 20, 768]
        x = torch.einsum("btd,tdh->bth", x, self.linear1_weight)
        x = self.layernorm1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        # positional embedding
        if self.is_position:
            x = x + self.pos_embedding

        # Connectome-Transformer with fc prior
        x = self.transformer_encoder(x)  # [B, 20, 768]

        x = x.permute(0, 2, 1)  # [B, 768, 20]
        x = self.linear2(x)  # [B, 768, 257]
        x = x.permute(0, 2, 1)  # [B, 257, 768]

        # MLP backborn, MLP projection(contrastive learning)
        return x, self.projector(x.reshape(len(x), -1, self.embed_dim))
    
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, fc_matrix_path, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.fc_matrix_path = fc_matrix_path
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = F.relu 

    def forward(self, x, src_mask=None, is_causal=False, src_key_padding_mask=None):
        # Self-attention block
        residual = x
        x = self.self_attn(x, self.fc_matrix_path)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        # Feedforward block
        residual = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        x = self.norm2(x)

        return x

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, fc_matrix_path):
        B, T, E = x.shape
        
        # q, k, v 한 번에 계산하고 쪼갬
        qkv = self.qkv_proj(x)  # (B, T, 3E)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)


        # FC 사용
        fc_matrix = np.load(fc_matrix_path)        # shape (T, T)
        fc_matrix = torch.from_numpy(fc_matrix).float().to(x.device)
        fc_matrix = fc_matrix.unsqueeze(0).unsqueeze(0)
        fc_matrix = fc_matrix.expand(B, 1, T, T)
        attn_scores = attn_scores + fc_matrix * 0.7

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)

        out = self.out_proj(attn_output)

        return out
    
class BrainNetwork(nn.Module):
    def __init__(self, in_dim=15724, out_dim=257*768, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False):
        super().__init__()
        
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h) # batch norm과 layer norm에 인자(h)를 미리 고정
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU # batch norm이면 ReLU사용하도록 고정
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.clip_size = clip_size
        self.n_blocks = n_blocks

        # MLP back born 할 때 사용
        self.lin0 = nn.Sequential( # 15724 -> 4096
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm], # [nn.BatchNorm1d(4096), nn.ReLU(inplace=True)] -> unpacking
            nn.Dropout(0.5),
        ) 
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm], # [nn.BatchNorm1d(4096), nn.ReLU(inplace=True)] -> unpacking
                nn.Dropout(0.15)
            ) for _ in range(n_blocks) # 4개의 block 사용 -> sequential 4번 사용
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True) # 4096 -> (257 * 768)
        
        # contrastive learning 할 때 사용
        # clip_size -> 고차원 공간(2048) -> clip_size
        self.projector = nn.Sequential(
            nn.LayerNorm(clip_size),
            nn.GELU(),
            nn.Linear(clip_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, clip_size)
        )
        
    def forward(self, x):
        '''
        x(MLP backbone): fmri -> mlp - shape: [batch, (257*768)]
        x(MLP projector): fmri -> mlp -> mlp - shape: ([batch, 768], [batch, 257, 768])
        '''
        # fMRI volume 그대로 들어올 때
        if x.ndim == 4:
            # assert x.shape[1] == 81 and x.shape[2] == 104 and x.shape[3] == 83, "fMRI data shape 안 맞음" # [N, 81, 104, 83]은 nsd genaral roi이다
            assert x.shape[1] == 120 and x.shape[2] == 120 and x.shape[3] == 84, "fMRI data shape 안 맞음" # [N, 120, 120, 84]은 nsd raw data이다.
            x = x.reshape(x.shape[0], -1) # [N, 1209600]
        
        # MLP back born
        x = self.lin0(x) # bs, h

        residual = x
        for res_block in range(self.n_blocks): # block 개수 4개
            x = self.mlp[res_block](x)
            x += residual                                                                         
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)

        # MLP backborn, MLP projection(contrastive learning)
        return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        
# BrainNetwork과 versatileDiffusionPriorNetwork가 인자로 들어감
class BrainDiffusionPrior(DiffusionPrior):
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None) # 부모(DiffusionPrior)는 voxel2clip을 모르기 때문에 빼놓음
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    def forward(self, text_embed = None, image_embed = None, *args, **kwargs):
        '''
        loss: loss(prediction x_0, target x_0) - shape: Scala
        pred: x_t -> x_t-1 denoise한 결과 - shape: [batch, 257, 768]
        '''
        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # ex) ex: 1.0 ~ 2.5 -> prediction 후 self.image_embed_scale을 나눠줘야 함
        image_embed *= self.image_embed_scale

        # calculate forward loss
        loss, pred = self.p_losses(image_embed=image_embed, times=times, text_embed=text_embed, *args, **kwargs)
        
        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred

    def p_losses(self, image_embed, times, text_embed=None, noise = None):
        '''
        loss: loss(prediction x_0, target x_0) - shape: Scala
        pred: x_t -> x_t-1 denoise한 결과 - shape: [batch, 257, 768]
        '''
        # noise 정의
        noise = default(noise, lambda: torch.randn_like(image_embed)) 

        # random한 t가 들어옴 -> x_0에서 한 번에 t까지의 noise를 씌움 
        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        # prediction 값(x_0)
        pred = self.net(image_embed_noisy, times, text_embed=text_embed) # cond_drop_prob: 텍스트 & 이미지 드롭아웃 확률 

        # prediction 값(x_0) normalization
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        # x_0를 비교하는 방식
        target = image_embed

        loss = nn.functional.mse_loss(pred, target) # diffusion prior에서는 ε가 아닌 x_0를 예측함
        return loss, pred # train 할 때 보통 loss와 prediction 같이 반환

    @torch.no_grad()
    def p_sample_loop(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device
        x_start = None 

        # image_embed = x_t(가우시안 노이즈 최종버젼)에서 시작
        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)

        # image_embedding(x_t) nomalization
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        # t_t -> x_0으로 denoise 시작 -> self.noise_scheduler.num_timesteps: 1000
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, cond_scale = cond_scale, generator=generator)
        
        # image embedding(x_0) nomalization
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    # 한 스텝 denoise 적용
    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1., generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        
        # generator를 사용하면 분포에서 고정된 값을 뽑음 -> 매번 같은 이미지 생성
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)

        # t-1에 denoise 
        # 생성
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) # x_0이면 noise를 사용하지 않음
        # 적용
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

# dalle2의 diffusionprior에서 x0(보통은 입실론)을 뽑음 only 1 step (이름이 versatile인 이유는 모르겠음) -> DiffusionPrior class에서 net의 인자로 사용
# 참고로 입실론을 구하면 x0는 한 번에 구할 수 있음
class VersatileDiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = None, # time step 수
        num_time_embeds = 1, # time embedding 개수
        num_tokens = 257,
        causal = True,
        learned_query_mode = 'none', 
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps) # time embedding값이 continuous or discrete
        self.learned_query_mode = learned_query_mode # learnable positional embedding

        # time embedding으로 변환
        self.to_time_embeds = nn.Sequential( 
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # token
        if self.learned_query_mode == 'pos_emb': # image embedding + positional embedding
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)

        # transformer 저장
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal=causal, **kwargs)

        # tokens.shape = [batch_size, total_token, dim]
        # total_token = [brain_embed(257개), time_embed(1개), image_embed(257개)] -> 이 중에서 맞춰야 할 embed
        self.num_tokens = num_tokens # 예측해야할 vector 차원
        

    def forward(
        self,
        image_embed, # pure noise 
        diffusion_timesteps,
        *,
        text_embed=None,
    ):  
        # brain embed = text embed  
        if text_embed is not None:
            brain_embed = text_embed 
        
        # shape 변경
        image_embed = image_embed.view(len(image_embed),-1,768) # image_embed.shape -> [B, 257, 768]
        brain_embed = brain_embed.view(len(brain_embed),-1,768) # brain_embed.shape -> [B, 257, 768]
        
        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        # time_embedding
        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        # token 정의
        if self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch) # repeat: '257 768' shape -> 'b 257 768' shape로 변경
            image_embed = image_embed + pos_embs # pure noise로 만들어짐 -(forward)-> prediction 값

            tokens = torch.cat((
                brain_embed,  # [b, 257, 768] 
                time_embed,  # [b, 1, 768]
                image_embed,  # [b, 257, 768]
            ), dim = -2) # [b, 257 + 1 + 257 + 257, 768] = [b, 515, 768]

        # transformer
        tokens = self.causal_transformer(tokens) # output: [b, 515, 768]

        pred_image_embed = tokens[..., -self.num_tokens:, :] # output: [b, 257, 768] - image_embed를 prediction값으로 사용

        return pred_image_embed

class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth, # transformer의 layer 수 -> versatile의 한 step의 layer
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb), # tokens에서 key, query, value로 나눠짐
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x # attention + residual
            x = ff(x) + x # feedfoward + residual

        out = self.norm(x)
        return self.project_out(out)
    
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=True, ups_mode='4x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])

        # up ampling
        self.ups_mode = ups_mode 
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False) # 16384 = 64 * 16 * 16
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

    def forward(self, x, return_transformer_feats=False):
        '''
        loss 계산에 upsampler와 maps_projector 둘 다 필요하다

        self.upsampler(4x,8x): mlp -> up sampling - shape: [batch_size, 4, 64, 64] 
        self.maps_projector(x).flatten(2).permute(0,2,1): mlp -> up sampling -> projection - shape: 4x-[batch_size, 256, 512], 8x-[batch_size, 64, 256]
        '''

        # MLP = lin0 + lin1
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # (b,4096) -> (b,16384)

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)


def get_model_lowlevel(args):

    num_voxels = 15724

    #### voxel2autoencoder ####
    # voxel2autoencoder 정의
    voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)

    # convnext + mlp 정의(loss에서만 사용)
    cnx_path = os.path.join(args.cache_dir, "convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx = ConvnextXL(cnx_path)
    cnx.requires_grad_(False)
    cnx.eval()

    #### encoder & decoder 정의 ####
    try:
        # 전체 pipeline 로컬에서 로드
        sd_model_dir = os.path.join(args.cache_dir, "models--lambdalabs--sd-image-variations-diffusers", "snapshots")
        snapshot_name = os.listdir(sd_model_dir)  # 첫 snapshot
        snapshot_path = os.path.join(sd_model_dir, snapshot_name[0])

        sd_pipe = DiffusionPipeline.from_pretrained(snapshot_path)
    except Exception as e:
        print(f"[!] 로컬 snapshot 로딩 실패, 온라인에서 전체 모델 로드: {e}")
        sd_pipe = DiffusionPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", cache_dir=args.cache_dir)

    # 학습 비활성화
    sd_pipe.vae.eval()
    sd_pipe.vae.requires_grad_(False)
    
    vae = sd_pipe.vae
    noise_scheduler = sd_pipe.scheduler

    models = {
        "voxel2sd": voxel2sd,
        "cnx": cnx,
        "vae": vae,
        "noise_scheduler": noise_scheduler
    }

    return models

def get_model_highlevel(args):

    num_voxels = 15724
    
    #clip (img to clip embedding)
    clip_extractor = Clipper(clip_variant=args.clip_variant, norm_embs=args.norm_embs, hidden_state=args.hidden, device=args.device)
    
    # brain network (fmri to clip embedding)
    out_dim = args.token_size * args.clip_size
    voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim, clip_size=args.clip_size)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)

    # Connectome-Transformer
    out_dim = args.clip_size
    depth = 6 # transformer의 layer 수
    dim_head = 64 # head당 dim
    heads = args.clip_size//dim_head # attention head 수
    timesteps = 100 # difusion step 수 
    
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = 257,
        learned_query_mode="pos_emb"
    ).to(args.device)

    # VersatileDiffusionPriorNetwork + BrainNetwork가 인자로 들어감
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network, # VersatileDiffusionPriorNetwork(필수 모델) -> nn.Module이라 학습 됨
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip, # BrainNetwork(추가 모델) -> nn.Module이라 학습 됨
    ).to(args.device)

    #### versatile difussion 정의(unet, vae, noise scheduler) ####
    try:
        # vd_model_dir = os.path.join(args.cache_dir, "models--shi-labs--versatile-diffusion")
        vd_model_dir = os.path.join(args.cache_dir, "models--shi-labs--versatile-diffusion", "snapshots")
        snapshot_name = os.listdir(vd_model_dir)  # 첫 snapshot
        snapshot_path = os.path.join(vd_model_dir, snapshot_name[0])
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_model_dir)
    except: # 처음에는 모델 불러와야 함
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion", cache_dir = args.cache_dir)
    
    # versatile difussion의 unet 정의
    vd_pipe.image_unet.eval()
    vd_pipe.image_unet.requires_grad_(False)
    # DualTransformer2DModel 이름 추출에서 image부분만 사용한다고 명시
    for name, module in vd_pipe.image_unet.named_modules(): # class vd_pipe.image_unet를 찍은 object들 
        if isinstance(module, DualTransformer2DModel): # object들 중 DualTransformer2DModel 이름 추출
            module.mix_ratio = 0.0 # versatile에서 image condition만 사용
            # text contex를 사용하지 않더라도 shape는 맞춰야 함
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    # versatile difussion의 vae 정의 -> inference의 decoder로 사용
    vd_pipe.vae.eval()
    vd_pipe.vae.requires_grad_(False)

    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    models = {
        "clip": clip_extractor,
        "diffusion_prior": diffusion_prior,
        "unet": unet, # inference에서만 사용
        "vae": vae, # inference에서만 사용
        "noise_scheduler": noise_scheduler, # inference에서만 사용
    }

    return models

def get_model_highlevel_FuncSpatial(args):
    
    # clip (img to clip embedding)
    clip_extractor = Clipper(clip_variant=args.clip_variant, norm_embs=args.norm_embs, hidden_state=args.hidden, device=args.device)
    
    # Brain network (region-level embedding + Connectome-Transformer)
    out_dim = args.token_size * args.clip_size
    voxel2clip_kwargs = dict(input_dim=2056, embed_dim=768, output_dim=257, seq_len=16, nhead=8, num_layers=args.num_layers, is_position=args.is_position, is_fc=args.is_fc, fc_matrix_path=args.fc_matrix_path)
    voxel2clip = BrainTransformerNetwork(**voxel2clip_kwargs)

    # diffusion prior
    out_dim = args.clip_size
    depth = 6
    dim_head = 64
    heads = args.clip_size//dim_head
    timesteps = 100  
    
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = 257,
        learned_query_mode="pos_emb"
    ).to(args.device)

    # VersatileDiffusionPriorNetwork + BrainNetwork가 인자로 들어감
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network, # VersatileDiffusionPriorNetwork(필수 모델) -> nn.Module이라 학습 됨
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip, # BrainNetwork(추가 모델) -> nn.Module이라 학습 됨
    ).to(args.device)

    #### versatile difussion 정의(unet, vae, noise scheduler) ####
    try:
        # vd_model_dir = os.path.join(args.cache_dir, "models--shi-labs--versatile-diffusion")
        vd_model_dir = os.path.join(args.cache_dir, "models--shi-labs--versatile-diffusion", "snapshots")
        snapshot_name = os.listdir(vd_model_dir)  # 첫 snapshot
        snapshot_path = os.path.join(vd_model_dir, snapshot_name[0])
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_model_dir)
    except: # 처음에는 모델 불러와야 함
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion", cache_dir = args.cache_dir)
    
    # versatile difussion의 unet 정의
    vd_pipe.image_unet.eval()
    vd_pipe.image_unet.requires_grad_(False)
    # DualTransformer2DModel 이름 추출에서 image부분만 사용한다고 명시
    for name, module in vd_pipe.image_unet.named_modules(): # class vd_pipe.image_unet를 찍은 object들 
        if isinstance(module, DualTransformer2DModel): # object들 중 DualTransformer2DModel 이름 추출
            module.mix_ratio = 0.0 # versatile에서 image condition만 사용
            # text contex를 사용하지 않더라도 shape는 맞춰야 함
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    # versatile difussion의 vae 정의 -> inference의 decoder로 사용
    vd_pipe.vae.eval()
    vd_pipe.vae.requires_grad_(False)

    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    models = {
        "clip": clip_extractor,
        "diffusion_prior": diffusion_prior,
        "unet": unet, # inference에서만 사용
        "vae": vae, # inference에서만 사용
        "noise_scheduler": noise_scheduler, # inference에서만 사용
    }

    return models
