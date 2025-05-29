import json
import random

# 設定路徑與輸出檔名
captions_path = "eval_data/captions_val2017.json"  # 請確保你已經有這個檔案
output_txt = "eval_data/eval_prompts_10k.txt"
num_prompts = 10000

# 讀取 COCO annotations
with open(captions_path, "r") as f:
    data = json.load(f)

# 抽取 captions
captions = [ann["caption"].strip() for ann in data["annotations"]]
print(f"總共有 {len(captions)} 筆可用 caption")

# 隨機抽樣
sampled = random.sample(captions, num_prompts)

# 儲存成 txt
with open(output_txt, "w") as f:
    for cap in sampled:
        f.write(cap + "\n")

print(f"已儲存 {num_prompts} 筆 prompt 至 {output_txt}")
