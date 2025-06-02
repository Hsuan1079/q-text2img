#!/usr/bin/env python3
# build_qdiff_cali_pt_steps.py  (with prompt file)
# ----------------------------------------------------------
# 產生 { 'xs': list[tensor], 'ts': list[tensor],
#        'cs': tensor, 'ucs': tensor }  (float16，節省容量)
# ----------------------------------------------------------
import os, argparse
from pathlib import Path
from tqdm import tqdm
import torch, json
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# ------------------------ 參數 ------------------------
p = argparse.ArgumentParser()
p.add_argument('--img-dir',    default='cali_images')               # 1024 張圖
p.add_argument('--prompt-file',default='cali_images/selected_prompts.txt')      # ★ 對應 prompt
p.add_argument('--ckpt',       default='models/ldm/stable-diffusion-v1/model.ckpt')
p.add_argument('--cfg',        default='configs/stable-diffusion/v1-inference.yaml')
p.add_argument('--steps',      type=int, default=50)
p.add_argument('--enc-batch',  type=int, default=16)
p.add_argument('--out',        default='sd_cali_steps.pt')
args = p.parse_args()

TIMESTEPS = list(range(0, 1000, 1000 // args.steps))   # 0,20,…,980
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------- 載入 UNet -----------------------
print('→ load UNet …')
cfg  = OmegaConf.load(args.cfg).model
unet = instantiate_from_config(cfg)
unet.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'],
                     strict=False)
unet = unet.to(device).eval()

# ----------------------- 影像 & prompt -------------------
img_paths = sorted([str(p) for p in Path(args.img_dir).glob('*')
                    if p.suffix.lower() in ('.png','.jpg','.jpeg')])
assert img_paths, '⚠ cali_images 資料夾沒有圖！'

with open(args.prompt_file, 'r', encoding='utf-8') as f:            # ★
    prompts = [l.strip() for l in f.readlines()]
assert len(prompts) == len(img_paths), '圖數與 prompt 行數不一致！'

pre = transforms.Compose([
    transforms.Resize(512), transforms.CenterCrop(512),
    transforms.ToTensor(),  transforms.Normalize([0.5]*3, [0.5]*3),
])

# ------------------- encode_first_stage -----------------
print('→ encode_first_stage …')
z_list = []
for i in tqdm(range(0, len(img_paths), args.enc_batch), desc='encode'):
    imgs  = [pre(Image.open(p).convert('RGB')) for p in img_paths[i:i+args.enc_batch]]
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        z = unet.get_first_stage_encoding(unet.encode_first_stage(batch))
    z_list.append(z.cpu())
z0 = torch.cat(z_list)           # (N,4,64,64) fp32
N  = z0.size(0)

# ------------------- cond / ucond -----------------------
print('→ encode text prompts …')
with torch.no_grad():
    cs  = unet.get_learned_conditioning(prompts).cpu()   # ★ 有條件
ucs = unet.get_learned_conditioning(['']*N).cpu()        #   無條件

# ★ 使 cond / ucond 與 timestep 數量對齊 -------------------
repeat_factor = len(TIMESTEPS)
cs  = cs.repeat(repeat_factor, 1, 1)     # (N*steps, 77, 768)
ucs = ucs.repeat(repeat_factor, 1, 1)

# ------------------- make x_t ---------------------------
Xs, Ts = [], []
for t_int in tqdm(TIMESTEPS, desc='make x_t'):
    t     = torch.full((N,), t_int, dtype=torch.long)            # (N,)
    a_bar = unet.alphas_cumprod[t_int].item()
    xt    = (a_bar**0.5) * z0 + ((1-a_bar)**0.5) * torch.randn_like(z0)
    Xs.append(xt)        # fp16 節省空間
    Ts.append(t)

# ------------------- save -------------------------------
torch.save({'xs': Xs, 'ts': Ts, 'cs': cs, 'ucs': ucs}, args.out)
print(f'✅ saved → {args.out} | steps={len(TIMESTEPS)} | per-step={N}')
