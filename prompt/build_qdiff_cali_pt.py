#!/usr/bin/env python3
# build_qdiff_cali_pt_steps.py
# ----------------------------------------------------------
# 產生 { 'xs': list[tensor], 'ts': list[tensor],
#        'cs': tensor, 'ucs': tensor }  (float32 全精度)
# ----------------------------------------------------------
import os, argparse
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# ------------------------ 參數 ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img-dir',  default='cali_images', help='1024 張實圖資料夾')
parser.add_argument('--ckpt',     default='models/ldm/stable-diffusion-v1/model.ckpt')
parser.add_argument('--cfg',      default='configs/stable-diffusion/v1-inference.yaml')
parser.add_argument('--steps',    type=int, default=50,     help='從 0~999 均勻抽多少步')
parser.add_argument('--enc-batch',type=int, default=16,     help='encode_first_stage batch size')
parser.add_argument('--out',      default='sd_cali_steps.pt')
args = parser.parse_args()

TIMESTEPS = list(range(0,1000, 1000//args.steps))   # e.g. 0,20,40,…,980
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------- 載入 UNet -----------------------
print('→ load UNet fp32 …')
unet_cfg = OmegaConf.load(args.cfg).model
unet = instantiate_from_config(unet_cfg)
sd = torch.load(args.ckpt, map_location='cpu')['state_dict']
unet.load_state_dict(sd, strict=False)
unet = unet.to(device).eval()        # **fp32**

# ----------------------- 影像前處理 -----------------------
img_paths = sorted([str(p) for p in Path(args.img_dir).glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg')])
assert img_paths, '⚠ cali_images 資料夾沒有圖！'

pre = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ------------------- encode_first_stage -------------------
print('→ encode_first_stage …')
z_list = []
with torch.no_grad():
    for i in tqdm(range(0,len(img_paths), args.enc_batch)):
        batch_imgs = [pre(Image.open(p).convert('RGB')) for p in img_paths[i:i+args.enc_batch]]
        batch = torch.stack(batch_imgs).to(device)          # (B,3,512,512)
        z = unet.get_first_stage_encoding(unet.encode_first_stage(batch))  # (B,4,64,64)
        z_list.append(z.cpu())                              # keep on CPU
z0 = torch.cat(z_list)          # (N,4,64,64)  float32
N   = z0.size(0)
print(f'latent batch shape: {z0.shape}')

# ---------------------- cond / ucond ----------------------
with torch.no_grad():
    cond = unet.get_learned_conditioning(['']*N).cpu()   # (N,77,768)
ucs  = cond.clone()

# ------------------- 每個 timestep 做 forward noise -------------------
Xs, Ts = [], []
for t_int in tqdm(TIMESTEPS, desc='make x_t'):
    t = torch.full((N,), t_int, dtype=torch.long)          # (N,)  on CPU
    # 1. 取出 scalar → 搬到 CPU → 轉成 float32
    a_bar = unet.alphas_cumprod[t_int].cpu().float()       # ()  scalar
    # 2. 建立一次就好，靠 broadcast
    sqrt_ab   = torch.sqrt(a_bar)
    sqrt_1mab = torch.sqrt(1.0 - a_bar)

    noise = torch.randn_like(z0)                          # CPU
    xt    = sqrt_ab * z0 + sqrt_1mab * noise              # (N,4,64,64)

    Xs.append(xt)        # still float32 / CPU
    Ts.append(t)         # (N,)


# --------------------------- 儲存 ---------------------------
torch.save(
    {'xs': Xs, 'ts': Ts, 'cs': cond, 'ucs': ucs},
    args.out
)
print(f'✅ saved → {args.out}')
