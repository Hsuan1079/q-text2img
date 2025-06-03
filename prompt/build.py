import os, argparse
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# ------------------------ 參數 ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img-dir',    default='cali_images', help='1024 張實圖資料夾')
parser.add_argument('--ckpt',       default='models/ldm/stable-diffusion-v1/model.ckpt')
parser.add_argument('--cfg',        default='configs/stable-diffusion/v1-inference.yaml')
parser.add_argument('--steps',      type=int, default=50,    help='從 0~999 均勻抽多少步 (這將決定輸出的列表長度)')
parser.add_argument('--enc-batch',  type=int, default=16,    help='encode_first_stage batch size')
parser.add_argument('--out',        default='sd_cali_steps_revised.pt')
args = parser.parse_args()

# TIMESTEPS 將包含 args.steps 個唯一的時間步值
# 例如，如果 args.steps = 50, model_total_timesteps = 1000, TIMESTEPS = [1, 21, 41, ..., 981]
TIMESTEPS = sorted(list(set(np.round(np.linspace(1, 981, args.steps)).astype(int))))
print(f"將為以下 {len(TIMESTEPS)} 個唯一時間步生成校準數據: {TIMESTEPS}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------- 載入 UNet (實際上是 LDM 模型) -----------------------
print('→ 載入 LDM 模型 (fp32) …')
unet_cfg = OmegaConf.load(args.cfg).model
# unet 變數實際上是 LDM 模型，它包含了 UNet, VAE, Text Encoder
ldm_model = instantiate_from_config(unet_cfg)
sd = torch.load(args.ckpt, map_location='cpu')['state_dict']
m, u = ldm_model.load_state_dict(sd, strict=False)
if len(m) > 0: print("載入模型時缺少的鍵:", m)
if len(u) > 0: print("載入模型時未預期的鍵:", u)
ldm_model = ldm_model.to(device).eval()

# ----------------------- 影像前處理 -----------------------
img_paths = sorted([str(p) for p in Path(args.img_dir).glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg')])
assert img_paths, f'⚠ {args.img_dir} 資料夾沒有圖！'

# 讀取對應的提示詞
prompts_path = os.path.join(args.img_dir, 'selected_prompts.txt')
with open(prompts_path, 'r') as f:
    prompts = [line.strip() for line in f.readlines()]
assert len(prompts) == len(img_paths), f'提示詞數量 ({len(prompts)}) 與圖片數量 ({len(img_paths)}) 不符！'

pre = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3), # 假設是 RGB
])

# ------------------- encode_first_stage -------------------
print('→ 將圖像編碼到潛在空間 (encode_first_stage) …')
z_list = []
with torch.no_grad():
    for i in tqdm(range(0,len(img_paths), args.enc_batch), desc="Encoding images"):
        batch_imgs_pil = [Image.open(p).convert('RGB') for p in img_paths[i:i+args.enc_batch]]
        batch_tensor = torch.stack([pre(img_pil) for img_pil in batch_imgs_pil]).to(device) # (B,3,512,512)
        # 使用 LDM 模型的方法來獲取潛在表示 z
        z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(batch_tensor)) # (B,4,64,64)
        z_list.append(z.cpu())
z0 = torch.cat(z_list) # (N,4,64,64) float32
N = z0.size(0)
print(f'潛在表示批次形狀: {z0.shape}')

# ---------------------- cond / ucond ----------------------
print('→ 生成 (無)條件嵌入 …')
with torch.no_grad():
    # 使用實際的提示詞生成條件嵌入
    cond_embedding = ldm_model.get_learned_conditioning(prompts).cpu() # (N, 77, 768)
    # 無條件嵌入仍然使用空字串
    ucs_embedding = ldm_model.get_learned_conditioning(['']*N).cpu() # (N, 77, 768)

# ------------------- 每個 timestep 做 forward noise -------------------
# 預分配張量
xs = torch.zeros((len(TIMESTEPS), N, 4, 64, 64), dtype=torch.float32)
ts = torch.zeros((len(TIMESTEPS), N), dtype=torch.long)
cs = torch.zeros((len(TIMESTEPS), N, 77, 768), dtype=torch.float32)
ucs = torch.zeros((len(TIMESTEPS), N, 77, 768), dtype=torch.float32)

for i, t_int in enumerate(tqdm(TIMESTEPS, desc='生成 x_t 並組織校準數據')):
    # 設置時間步
    ts[i] = t_int
    
    # 加噪過程
    a_bar = ldm_model.alphas_cumprod[t_int].cpu().float() # scalar
    sqrt_ab   = torch.sqrt(a_bar)
    sqrt_1mab = torch.sqrt(1.0 - a_bar)
    noise = torch.randn_like(z0) # z0 已在 CPU 上, (N,4,64,64)
    xs[i] = sqrt_ab * z0 + sqrt_1mab * noise # (N,4,64,64)
    
    # 複製條件嵌入
    cs[i] = cond_embedding
    ucs[i] = ucs_embedding

# --------------------------- 儲存 ---------------------------
data_to_save = {
    'xs': xs,      # (steps, N, 4, 64, 64)
    'ts': ts,      # (steps, N)
    'cs': cs,      # (steps, N, 77, 768)
    'ucs': ucs,    # (steps, N, 77, 768)
    'prompts': prompts
}
torch.save(data_to_save, args.out)
print(f"✅ 校準數據已儲存到 → {args.out}")
print(f"  數據結構:")
print(f"    xs: {xs.shape}")
print(f"    ts: {ts.shape}")
print(f"    cs: {cs.shape}")
print(f"    ucs: {ucs.shape}")
print(f"    prompts 長度: {len(prompts)}")