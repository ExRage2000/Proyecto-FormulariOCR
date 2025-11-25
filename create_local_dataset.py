import os
import json
import math
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datasets import load_dataset



def save_shard(shard_data):
    """
    Worker function to save a chunk of images.
    Args:
        shard_data: tuple (indices_range, dataset_split, split_dir, shard_id)
    Returns:
        List of metadata entries for this shard
    """
    indices, dataset_subset, split_dir, shard_id = shard_data
    
    # Create a subfolder for this shard (e.g., "train/shard_001")
    # This prevents having 150k files in one folder (which is slow/bad for OS)
    shard_dir = os.path.join(split_dir, f"shard_{shard_id:04d}")
    os.makedirs(shard_dir, exist_ok=True)
    
    local_metadata = []
    
    for i, example in zip(indices, dataset_subset):
        image = example["im"]
        
        # Unique filename: shard_001_image_0.png
        filename = f"shard_{shard_id:04d}_image_{i}.png"
        image_path = os.path.join(shard_dir, filename)
        
        # Convert & Save
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Optimize=False is faster. 
        # PNG compression is CPU heavy; default is usually fine but parallelizing is key.
        image.save(image_path, optimize=False) 
        
        # Relative path for metadata (e.g., "shard_0001/shard_0001_image_0.png")
        # This allows the metadata file to stay in the root split folder
        rel_path = os.path.join(f"shard_{shard_id:04d}", filename)
        
        entry = {
            "file_name": rel_path,
            "text": example["text"]
        }
        
        # Add extra fields if they exist
        for field in ['century', 'language', 'script_type']:
            if field in example:
                entry[field] = example[field]
                
        local_metadata.append(entry)
        
    return local_metadata

def fast_save_dataset(dataset_dict, output_root="local_dataset", num_workers=None, shard_size=1000):
    """
    Saves dataset in parallel using ProcessPoolExecutor.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 4
        
    print(f"Starting parallel export with {num_workers} workers...")
    
    for split_name, dataset in dataset_dict.items():
        print(f"Processing split: {split_name} ({len(dataset)} items)")
        
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Prepare tasks for workers
        num_shards = math.ceil(len(dataset) / shard_size)
        tasks = []
        
        for shard_id in range(num_shards):
            start_idx = shard_id * shard_size
            end_idx = min(start_idx + shard_size, len(dataset))
            indices = range(start_idx, end_idx)
            
            # We pass a slice of the dataset to avoid pickling the whole thing repeatedly
            # Note: HuggingFace datasets are efficient at slicing (memory mapped)
            subset = dataset.select(indices)
            
            tasks.append((indices, subset, split_dir, shard_id))
        
        # Run in parallel
        all_metadata = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Use tqdm to show progress bar
            results = list(tqdm(executor.map(save_shard, tasks), total=len(tasks), unit="shard"))
            
            for res in results:
                all_metadata.extend(res)
        
        # Write consolidated metadata.jsonl
        print(f"Writing metadata for {split_name}...")
        meta_path = os.path.join(split_dir, "metadata.jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for entry in all_metadata:
                f.write(json.dumps(entry) + "\n")
                
    print(f"Done! Saved to {output_root}")

# --- Usage ---

ds = load_dataset("CATMuS/medieval")
fast_save_dataset(ds, num_workers=8)