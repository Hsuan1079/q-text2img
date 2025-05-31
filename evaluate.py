import os
import argparse
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_fid import fid_score
import json
import datetime

def compute_clip_score(img_dir, prompt_file):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    model.eval().cuda()

    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    assert len(prompts) == len(img_paths), "Image/prompt 數量不一致"

    scores = []
    for prompt, path in tqdm(zip(prompts, img_paths), total=len(prompts), desc="CLIP Score"):
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).cuda()
        text = tokenizer([prompt]).cuda()

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            score = (image_features * text_features).sum().item()
            scores.append(score)
    print(f"CLIP Score (mean): {sum(scores)/len(scores):.4f}")

    # 保存分數到 JSON 文件 
    scores_dict = {
        "CLIP_Score": float(sum(scores)/len(scores)),
        "timestamp": str(datetime.datetime.now())
    }
    
    # 確保 scores 目錄存在
    os.makedirs("scores", exist_ok=True)
    
    # 保存分數
    with open("scores/CLIP_scores.json", "w") as f:
        json.dump(scores_dict, f, indent=4)
    print("Scores have been saved to scores/CLIP_scores.json")

def compute_fid_score(img_dir, real_dir):
    from torchvision.transforms.functional import to_tensor
    import shutil

    temp_img_dir = os.path.join(img_dir, 'temp_resized')
    temp_real_dir = os.path.join(real_dir, 'temp_resized')
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_real_dir, exist_ok=True)

    transform = transforms.Resize((299, 299))  # Inception v3 尺寸

    # 僅處理 .png/.jpg 檔，並排序以保證順序一致
    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
    real_files = sorted(f for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg', '.jpeg')))

    assert len(img_files) == len(real_files), f"圖片數量不一致：{len(img_files)} vs {len(real_files)}"

    for f in img_files:
        img = Image.open(os.path.join(img_dir, f)).convert('RGB')
        img = transform(img)
        save_image(to_tensor(img), os.path.join(temp_img_dir, f))

    for f in real_files:
        img = Image.open(os.path.join(real_dir, f)).convert('RGB')
        img = transform(img)
        save_image(to_tensor(img), os.path.join(temp_real_dir, f))

    fid_value = fid_score.calculate_fid_given_paths(
        [temp_img_dir, temp_real_dir],
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048
    )

    print(f"FID Score: {fid_value:.4f}")

    scores_dict = {
        "FID_Score": float(fid_value),
        "timestamp": str(datetime.datetime.now())
    }

    os.makedirs("scores", exist_ok=True)
    with open("scores/FID_scores.json", "w") as f:
        json.dump(scores_dict, f, indent=4)

    shutil.rmtree(temp_img_dir)
    shutil.rmtree(temp_real_dir)


def main():
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加載模型
    model, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()

    # 設置數據轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加載數據
    dataset = ImageFolder(root='./data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 計算特徵
    features = []
    labels = []
    with torch.no_grad():
        for images, image_labels in tqdm(dataloader, desc="Computing features"):
            images = images.to(device)
            image_features = model.encode_image(images)
            features.append(image_features.cpu())
            labels.extend(image_labels.numpy())

    features = torch.cat(features, dim=0)
    labels = np.array(labels)

    # 計算相似度矩陣
    similarity_matrix = cosine_similarity(features)

    # 計算每個類別的平均相似度
    unique_labels = np.unique(labels)
    intra_class_similarities = []
    inter_class_similarities = []

    for label in unique_labels:
        # 獲取當前類別的所有樣本索引
        class_indices = np.where(labels == label)[0]
        
        # 計算類內相似度
        for i in range(len(class_indices)):
            for j in range(i + 1, len(class_indices)):
                intra_class_similarities.append(similarity_matrix[class_indices[i], class_indices[j]])
        
        # 計算類間相似度
        other_indices = np.where(labels != label)[0]
        for i in class_indices:
            for j in other_indices:
                inter_class_similarities.append(similarity_matrix[i, j])

    # 計算平均分數
    intra_class_score = np.mean(intra_class_similarities)
    inter_class_score = np.mean(inter_class_similarities)

    print(f"Intra-class similarity score: {intra_class_score:.4f}")
    print(f"Inter-class similarity score: {inter_class_score:.4f}")

    # 保存分數到 JSON 文件
    scores = {
        "intra_class_score": float(intra_class_score),
        "inter_class_score": float(inter_class_score),
        "timestamp": str(datetime.datetime.now())
    }
    
    # 確保 scores 目錄存在
    os.makedirs("scores", exist_ok=True)
    
    # 保存分數
    with open("scores/similarity_scores.json", "w") as f:
        json.dump(scores, f, indent=4)
    
    print("Scores have been saved to scores/similarity_scores.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True, help="生成圖片的資料夾")
    parser.add_argument("--prompt-file", type=str, help="生成圖片所對應的 prompt 檔案")
    parser.add_argument("--real-dir", type=str, help="真實圖片資料夾，用於 FID 計算")
    args = parser.parse_args()

    # if args.prompt_file:
    #     compute_clip_score(args.img_dir, args.prompt_file)
    if args.real_dir:
        compute_fid_score(args.img_dir, args.real_dir)
    # main()
