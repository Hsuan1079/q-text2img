import os
import argparse
import torch
import open_clip # For compute_clip_score
from PIL import Image
from tqdm import tqdm
# from torchvision.datasets.folder import default_loader # Not explicitly used
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_fid import fid_score
import json
import datetime
import shutil # For compute_fid_score (rmtree)
from torchvision.transforms.functional import to_tensor # For compute_fid_score

def compute_clip_score(img_dir, prompt_file):
    """
    Computes the CLIP score between images in img_dir and prompts in prompt_file.
    """
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    # Ensure paths are constructed correctly and files are filtered
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")]) # Assuming .png, adjust if other types
    img_paths = [os.path.join(img_dir, f) for f in img_files]

    if not img_paths:
        print(f"No .png images found in {img_dir} for CLIP score calculation.")
        return
    if not prompts:
        print(f"No prompts found in {prompt_file} for CLIP score calculation.")
        return

    # If you want to limit CLIP score images as well, you'd apply similar logic here
    # For now, it uses all found images that match prompts.
    # Example: img_paths = img_paths[:1000]
    # prompts = prompts[:len(img_paths)] # Ensure prompts match the number of images

    if len(prompts) != len(img_paths):
        print(f"Warning: Number of prompts ({len(prompts)}) and images ({len(img_paths)}) for CLIP score do not match. Truncating to the shorter list.")
        min_len = min(len(prompts), len(img_paths))
        prompts = prompts[:min_len]
        img_paths = img_paths[:min_len]
        if min_len == 0:
            print("No matching image-prompt pairs for CLIP score calculation.")
            return


    scores = []
    for prompt, path in tqdm(zip(prompts, img_paths), total=len(prompts), desc="CLIP Score"):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                score = (image_features * text_features).sum().item()
                scores.append(score)
        except Exception as e:
            print(f"Error processing image {path} or prompt '{prompt}' for CLIP score: {e}")
            # Optionally, decide to skip or handle this error, e.g., append a NaN or default score
            
    if not scores:
        print("CLIP Score could not be computed (no successful comparisons).")
        return

    mean_clip_score = sum(scores) / len(scores)
    print(f"CLIP Score (mean): {mean_clip_score:.4f} (based on {len(scores)} image-prompt pairs)")

    scores_dict = {
        "CLIP_Score": float(mean_clip_score),
        "num_pairs_evaluated": len(scores),
        "timestamp": str(datetime.datetime.now())
    }
    
    os.makedirs("scores", exist_ok=True)
    
    with open("scores/CLIP_scores.json", "w") as f:
        json.dump(scores_dict, f, indent=4)
    print("CLIP scores have been saved to scores/CLIP_scores.json")


def compute_fid_score(img_dir, real_dir):
    """
    Computes the FID score between images in img_dir and real_dir.
    Processes up to the first 1000 images from img_dir and real_dir.
    """
    # Temporary directories for resized images
    # Using more specific names to avoid potential conflicts
    temp_img_dir = os.path.join(img_dir, 'temp_resized_for_fid_generated')
    temp_real_dir = os.path.join(real_dir, 'temp_resized_for_fid_real')
    
    # Ensure directories are clean or non-existent before creation
    if os.path.exists(temp_img_dir):
        shutil.rmtree(temp_img_dir)
    if os.path.exists(temp_real_dir):
        shutil.rmtree(temp_real_dir)
        
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_real_dir, exist_ok=True)

    # Transformation for Inception v3
    # The FID score is typically calculated on images resized to 299x299
    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        # ToTensorV2() might be part of some setups, but fid_score often handles PIL images
        # The pytorch_fid library expects paths to image files, not tensors directly for calculate_fid_given_paths
        # So we save resized images to disk.
    ])

    # Get all relevant file names, sorted to ensure consistency
    # Using .lower() for case-insensitive extension checking
    all_img_filenames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    all_real_filenames = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    NUM_IMAGES_TARGET = 1000

    # Select the first NUM_IMAGES_TARGET from img_dir
    selected_img_filenames = all_img_filenames[:NUM_IMAGES_TARGET]
    # Select the first NUM_IMAGES_TARGET from real_dir
    selected_real_filenames = all_real_filenames[:NUM_IMAGES_TARGET]
    
    count_from_img_dir = len(selected_img_filenames)
    count_from_real_dir = len(selected_real_filenames)

    if count_from_img_dir == 0:
        print(f"Error: No suitable images (png, jpg, jpeg) found in '{img_dir}' (or first {NUM_IMAGES_TARGET} were not suitable). Cannot compute FID.")
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_real_dir)
        return
    
    if count_from_real_dir == 0:
        print(f"Error: No suitable images (png, jpg, jpeg) found in '{real_dir}' (or first {NUM_IMAGES_TARGET} were not suitable). Cannot compute FID.")
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_real_dir)
        return

    # Determine the actual number of images to process for FID (must be same for both paths)
    actual_num_to_process = min(count_from_img_dir, count_from_real_dir)

    if actual_num_to_process == 0: # Should be caught by above checks, but as a safeguard
        print(f"Error: Zero images available for comparison after selection. Cannot compute FID.")
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_real_dir)
        return

    if actual_num_to_process < NUM_IMAGES_TARGET:
        print(f"Warning: Will compare {actual_num_to_process} images from each directory (target was {NUM_IMAGES_TARGET}).")
        if count_from_img_dir < actual_num_to_process or (count_from_img_dir < NUM_IMAGES_TARGET and count_from_img_dir == actual_num_to_process):
             print(f"  '{img_dir}' provided {count_from_img_dir} images (within the first {NUM_IMAGES_TARGET}).")
        if count_from_real_dir < actual_num_to_process or (count_from_real_dir < NUM_IMAGES_TARGET and count_from_real_dir == actual_num_to_process):
             print(f"  '{real_dir}' provided {count_from_real_dir} images (within the first {NUM_IMAGES_TARGET}).")
    else:
         print(f"Processing {NUM_IMAGES_TARGET} images from '{img_dir}' and {NUM_IMAGES_TARGET} images from '{real_dir}'.")

    # Final lists of filenames to use for FID
    img_files_for_fid = selected_img_filenames[:actual_num_to_process]
    real_files_for_fid = selected_real_filenames[:actual_num_to_process]

    # Process and save images to temporary directories
    print(f"Preparing {len(img_files_for_fid)} images from generated set ('{img_dir}')...")
    for f_name in tqdm(img_files_for_fid, desc="Processing generated images"):
        try:
            img_path = os.path.join(img_dir, f_name)
            img = Image.open(img_path).convert('RGB') # Ensure RGB
            img_resized = inception_transform(img)
            # Save the resized image. Note: save_image expects a tensor.
            save_image(to_tensor(img_resized), os.path.join(temp_img_dir, f_name.rsplit('.', 1)[0] + '.png')) # Standardize to PNG for FID tool
        except Exception as e:
            print(f"Error processing/saving generated image {os.path.join(img_dir, f_name)}: {e}. Skipping.")
            # If an image fails, the counts might mismatch. Robust handling would be to collect successful paths.
            # For now, this might lead to pytorch-fid complaining if files are missing.

    print(f"Preparing {len(real_files_for_fid)} images from real set ('{real_dir}')...")
    for f_name in tqdm(real_files_for_fid, desc="Processing real images"):
        try:
            img_path = os.path.join(real_dir, f_name)
            img = Image.open(img_path).convert('RGB') # Ensure RGB
            img_resized = inception_transform(img)
            save_image(to_tensor(img_resized), os.path.join(temp_real_dir, f_name.rsplit('.', 1)[0] + '.png')) # Standardize to PNG
        except Exception as e:
            print(f"Error processing/saving real image {os.path.join(real_dir, f_name)}: {e}. Skipping.")

    # It's crucial that temp_img_dir and temp_real_dir contain the same number of images for FID.
    # Let's count them before proceeding to FID calculation.
    final_generated_count = len(os.listdir(temp_img_dir))
    final_real_count = len(os.listdir(temp_real_dir))

    if final_generated_count != final_real_count:
        print(f"Error: Mismatch in processed image counts. Generated: {final_generated_count}, Real: {final_real_count}. FID cannot be reliably computed.")
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_real_dir)
        return
    
    if final_generated_count == 0: # Or final_real_count
        print(f"Error: No images were successfully processed into temporary directories. Cannot compute FID.")
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_real_dir)
        return
        
    print(f"Calculating FID score using {final_generated_count} images from each set...")
    fid_value = fid_score.calculate_fid_given_paths(
        [temp_img_dir, temp_real_dir],
        batch_size=32,  # This can be adjusted based on GPU memory
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048  # Standard dimension for Inception v3 features
    )

    print(f"FID Score: {fid_value:.4f}")

    scores_dict = {
        "FID_Score": float(fid_value),
        "num_images_compared_per_set": final_generated_count,
        "timestamp": str(datetime.datetime.now())
    }

    os.makedirs("scores", exist_ok=True)
    with open("scores/FID_scores.json", "w") as f:
        json.dump(scores_dict, f, indent=4)
    print(f"FID scores have been saved to scores/FID_scores.json")

    # Clean up temporary directories
    try:
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_real_dir)
        print("Temporary directories cleaned up.")
    except Exception as e:
        print(f"Error cleaning up temporary directories: {e}")


# This main function calculates intra/inter-class similarity, not directly related to FID/CLIP
# It was part of your original script but commented out in the __main__ block.
# I'm leaving it as is.
def main_similarity_calculation(): # Renamed to avoid confusion
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for similarity calculation: {device}")

    # Load OpenCLIP model (ViT-B-32 as per original main)
    # Note: compute_clip_score uses ViT-B-16. Ensure this is intentional if using both.
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()

    # Setup data transformations (as per original main)
    # This transform is different from Inception transform for FID or CLIP preprocess
    transform_similarity = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if 'numpy' and 'sklearn' are available if this function is to be used
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"Missing libraries for similarity_calculation (numpy, sklearn, torchvision): {e}")
        print("Please install them if you intend to use this functionality.")
        return

    # Load data (assuming './data' structure for ImageFolder)
    # You might need to pass the data root as an argument
    data_root = './data' 
    if not os.path.isdir(data_root):
        print(f"Data directory '{data_root}' for similarity calculation not found. Skipping this calculation.")
        return
        
    try:
        dataset = ImageFolder(root=data_root, transform=transform_similarity)
        if not dataset.classes:
            print(f"No classes found in '{data_root}'. Skipping similarity calculation.")
            return
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Error loading dataset from '{data_root}': {e}. Skipping similarity calculation.")
        return

    # Compute features
    features_list = []
    labels_list = []
    print("Computing features for similarity calculation...")
    with torch.no_grad():
        for images, image_labels in tqdm(dataloader, desc="Computing similarity features"):
            images = images.to(device)
            image_features = model.encode_image(images)
            features_list.append(image_features.cpu())
            labels_list.extend(image_labels.numpy())

    if not features_list:
        print("No features computed. Skipping similarity calculation.")
        return

    features_tensor = torch.cat(features_list, dim=0)
    labels_array = np.array(labels_list)

    # Compute similarity matrix
    print("Calculating similarity matrix...")
    # Ensure features_tensor is 2D for cosine_similarity
    if features_tensor.ndim > 2:
         features_tensor = features_tensor.view(features_tensor.shape[0], -1)
    similarity_matrix = cosine_similarity(features_tensor.numpy()) # Convert to numpy array

    # Compute average intra-class and inter-class similarities
    unique_labels = np.unique(labels_array)
    if len(unique_labels) < 2:
        print("Need at least two classes to compute inter-class similarity. Skipping.")
        # You could still compute intra-class similarity if desired
        # For now, skipping if not enough classes for full comparison
        return

    intra_class_similarities = []
    inter_class_similarities = []

    print("Calculating intra-class and inter-class similarities...")
    for label in tqdm(unique_labels, desc="Class similarities"):
        class_indices = np.where(labels_array == label)[0]
        other_indices = np.where(labels_array != label)[0]

        # Intra-class
        if len(class_indices) > 1:
            for i in range(len(class_indices)):
                for j in range(i + 1, len(class_indices)):
                    intra_class_similarities.append(similarity_matrix[class_indices[i], class_indices[j]])
        
        # Inter-class
        if len(other_indices) > 0 and len(class_indices) > 0 : # Check if other_indices is not empty
            for i in class_indices:
                for j in other_indices:
                    # Ensure i and j are within bounds of similarity_matrix rows/cols
                    if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                         inter_class_similarities.append(similarity_matrix[i, j])
                    else:
                        print(f"Warning: Index out of bounds ({i}, {j}) for similarity_matrix of shape {similarity_matrix.shape}")


    # Compute average scores
    intra_class_score = np.mean(intra_class_similarities) if intra_class_similarities else 0.0
    inter_class_score = np.mean(inter_class_similarities) if inter_class_similarities else 0.0

    print(f"Intra-class similarity score: {intra_class_score:.4f}")
    print(f"Inter-class similarity score: {inter_class_score:.4f}")

    # Save scores to JSON file
    scores_output = {
        "intra_class_score": float(intra_class_score),
        "inter_class_score": float(inter_class_score),
        "num_unique_classes": len(unique_labels),
        "num_samples": len(labels_array),
        "timestamp": str(datetime.datetime.now())
    }
    
    os.makedirs("scores", exist_ok=True)
    with open("scores/similarity_scores.json", "w") as f:
        json.dump(scores_output, f, indent=4)
    
    print("Similarity scores have been saved to scores/similarity_scores.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLIP and/or FID scores for generated images.")
    parser.add_argument("--img-dir", type=str, required=True, help="Directory containing the generated images.")
    parser.add_argument("--prompt-file", type=str, help="File containing prompts corresponding to generated images (for CLIP score). Each prompt on a new line.")
    parser.add_argument("--real-dir", type=str, help="Directory containing real images for FID score comparison.")
    # Argument for the similarity calculation data (if you want to enable it via CLI)
    # parser.add_argument("--similarity-data-dir", type=str, default="./data", help="Root directory for ImageFolder for similarity calculation.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.img_dir):
        print(f"Error: Generated images directory '{args.img_dir}' not found.")
        exit(1)

    if args.prompt_file:
        if not os.path.isfile(args.prompt_file):
            print(f"Error: Prompt file '{args.prompt_file}' not found.")
        else:
            print(f"\n--- Computing CLIP Score for '{args.img_dir}' ---")
            compute_clip_score(args.img_dir, args.prompt_file)
    
    if args.real_dir:
        if not os.path.isdir(args.real_dir):
            print(f"Error: Real images directory '{args.real_dir}' not found.")
        else:
            print(f"\n--- Computing FID Score between '{args.img_dir}' and '{args.real_dir}' ---")
            compute_fid_score(args.img_dir, args.real_dir)
            
    # If you want to run the similarity calculation:
    # print(f"\n--- Computing Similarity Scores (using data from '{args.similarity_data_dir}') ---")
    # main_similarity_calculation() # Call the renamed function
    
    print("\nEvaluation script finished.")

