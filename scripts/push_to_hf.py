import os
import argparse
from huggingface_hub import HfApi, create_repo

def push_checkpoints(hf_username):
    """
    Pushes the saved LoRA adapters from SFT and GRPO directly to the HuggingFace Hub.
    By using upload_folder, it safely pushes the adapter_config.json, safetensors, 
    and tokenizer files exactly as saved by the TRL trainers.
    """
    api = HfApi()
    
    checkpoints = {
        "Qwen3-VL-8B-dentex-sft-warmup": "checkpoints/sft_warmup/final",
        "Qwen3-VL-8B-dentex-rlvr-grpo": "checkpoints/grpo_run1/final",
    }
    
    for repo_name, local_path in checkpoints.items():
        if not os.path.exists(local_path):
            print(f"⚠️ Directory not found: {local_path}. Skipping...")
            continue
            
        repo_id = f"{hf_username}/{repo_name}"
        print(f"\n🚀 Preparing repository: {repo_id}")
        
        try:
            # Create the repo if it doesn't exist (set private=True for medical data safety initially)
            create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
            print(f"📦 Uploading {local_path} to {repo_id}...")
            
            api.upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✅ Successfully uploaded to https://huggingface.co/{repo_id}")
            
        except Exception as e:
            print(f"❌ Error uploading {repo_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload DENTEX LoRA adapters to HuggingFace Hub")
    parser.add_argument("--username", required=True, help="Your HuggingFace username (e.g. harshvardhanvatsa)")
    args = parser.parse_args()
    
    # Needs huggingface-cli login beforehand
    push_checkpoints(args.username)
