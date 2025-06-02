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
parser.add_argument('--steps',      type=int, default=5,    help='從 0~999 均勻抽多少步 (這將決定輸出的列表長度)')
parser.add_argument('--enc-batch',  type=int, default=16,    help='encode_first_stage batch size')
parser.add_argument('--out',        default='sd_cali_steps.pt')
args = parser.parse_args()

# TIMESTEPS 將包含 args.steps 個唯一的時間步值
# 例如，如果 args.steps = 50, model_total_timesteps = 1000, TIMESTEPS = [0, 20, 40, ..., 980]
TIMESTEPS = sorted(list(set(np.round(np.linspace(0, 999, args.steps)).astype(int))))
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
    # 為所有 N 個圖像生成無條件嵌入 (使用空字串)
    # 注意：get_learned_conditioning 輸入是一個列表，輸出的 batch size 與列表長度相同
    # 如果要為每個 z0 樣本都生成一個，則需要 ['']*N。
    # 或者，如果校準時 cs/ucs 是針對 "批次" 的，而不是針對每個 "時間步組" 內的 N 個樣本都不同，
    # 則這裡的 N 應該是 txt2img.py 中 --cali_n 的值，但这與 get_train_samples 的結構不符。
    # get_train_samples 期望 sample_data["cs"][i] 的形狀是 (num_samples, 77, 768)，
    # 其中 num_samples 是 args.cali_n。
    # 而我們這裡的 N 是總圖像數。
    #
    # 為了匹配 get_train_samples 的期望結構，cs 和 ucs 列表中的每個元素
    # 都應該是 (N, 77, 768) 的張量，其中 N 是當前時間步組中所有圖像的數量。
    # 在這個 build_qdiff_cali_pt.py 腳本中，每個時間步的 xt 都是基於全部 N 張原始圖像的 z0 生成的。
    # 所以，對於每個時間步組，cs 和 ucs 也應該是這 N 個圖像對應的（無）條件嵌入。
    
    # 生成一次即可，因為對於這個腳本，所有樣本共享相同的（無）條件嵌入（空字串）
    shared_cond_embedding = ldm_model.get_learned_conditioning(['']*N).cpu() # (N, 77, 768)
    shared_ucs_embedding = shared_cond_embedding.clone()

# ------------------- 每個 timestep 做 forward noise -------------------
# Xs, Ts, Cs, Ucs 都將是列表，長度為 len(TIMESTEPS)
# 每個元素是該時間步下所有 N 個樣本的數據
Xs_list, Ts_list, Cs_list, Ucs_list = [], [], [], []

for t_int in tqdm(TIMESTEPS, desc='生成 x_t 並組織校準數據'):
    t_tensor_for_all_N_samples = torch.full((N,), t_int, dtype=torch.long) # (N,) on CPU
    
    # 加噪過程 (與原腳本類似，但確保在 CPU 上操作以節省 GPU 記憶體)
    # ldm_model.alphas_cumprod 是在 GPU 上的，先取值再 .cpu()
    a_bar = ldm_model.alphas_cumprod[t_int].cpu().float() # scalar
    sqrt_ab   = torch.sqrt(a_bar)
    sqrt_1mab = torch.sqrt(1.0 - a_bar)
    noise = torch.randn_like(z0) # z0 已在 CPU 上, (N,4,64,64)
    xt = sqrt_ab * z0 + sqrt_1mab * noise # (N,4,64,64), on CPU

    Xs_list.append(xt)
    Ts_list.append(t_tensor_for_all_N_samples)
    Cs_list.append(shared_cond_embedding.clone()) # 每個時間步組都用相同的 (N, 77, 768)
    Ucs_list.append(shared_ucs_embedding.clone()) # 每個時間步組都用相同的 (N, 77, 768)

# --------------------------- 儲存 ---------------------------
# 儲存的字典，其值是張量的列表
data_to_save = {
    'xs': Xs_list,    # 列表，長度為 len(TIMESTEPS)，每個元素是 (N, 4, 64, 64)
    'ts': Ts_list,    # 列表，長度為 len(TIMESTEPS)，每個元素是 (N,)
    'cs': Cs_list,    # 列表，長度為 len(TIMESTEPS)，每個元素是 (N, 77, 768)
    'ucs': Ucs_list   # 列表，長度為 len(TIMESTEPS)，每個元素是 (N, 77, 768)
}
torch.save(data_to_save, args.out)
print(f"✅ 校準數據已儲存到 → {args.out}")
print(f"  數據結構: 'xs', 'ts', 'cs', 'ucs' 都是列表，列表長度為 {len(TIMESTEPS)}")
if Xs_list:
    print(f"  例如，列表中第一個元素的形狀:")
    print(f"    xs[0]: {Xs_list[0].shape}")
    print(f"    ts[0]: {Ts_list[0].shape}")
    print(f"    cs[0]: {Cs_list[0].shape}")
    print(f"    ucs[0]: {Ucs_list[0].shape}")
