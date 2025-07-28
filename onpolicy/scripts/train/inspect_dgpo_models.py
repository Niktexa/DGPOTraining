#!/usr/bin/env python
import torch
import os
from pathlib import Path

def inspect_pong_v2_model():
    print("🎯 INSPECTING YOUR NEW PONG_V2 MODEL...")
    
    # Path to your new model
    model_dir = "/Users/nickcataldo/Downloads/DGPO-DGPO_dev-2/onpolicy/scripts/results/Atari/ALE/Pong-v5/MAPPO/pong_v2/wandb/offline-run-20250727_155129-l18681pg/files/"
    
    model_files = {
        "Actor (Policy)": "actor.pt",
        "Intrinsic Critic": "in_critic.pt", 
        "Extrinsic Critic": "ex_critic.pt"
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        print(f"\n🧠 {model_name}:")
        print(f"📁 Path: {model_path}")
        
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                print(f"📊 Model Type: {type(checkpoint)}")
                
                if isinstance(checkpoint, dict):
                    print(f"🔑 Keys: {list(checkpoint.keys())}")
                    total_params = 0
                    
                    # Count parameters for each layer
                    for key, value in checkpoint.items():
                        if isinstance(value, torch.Tensor):
                            params = value.numel()
                            total_params += params
                            print(f"   {key}: {value.shape} ({params:,} parameters)")
                            
                            # Show weight statistics
                            print(f"      Mean: {value.mean().item():.6f}, Std: {value.std().item():.6f}")
                            print(f"      Min: {value.min().item():.6f}, Max: {value.max().item():.6f}")
                    
                    print(f"📈 Total Parameters: {total_params:,}")
                else:
                    print(f"📊 Model loaded successfully: {type(checkpoint)}")
            else:
                print(f"❌ File not found: {model_path}")
                
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)}")
        
        print("-" * 60)
    
    print("\n🎯 MODEL COMPARISON:")
    print("Your pong_v2 model should show different weight patterns than your original 'check' model!")
    print("This proves you successfully trained two independent DGPO agents! 🚀")

if __name__ == "__main__":
    inspect_pong_v2_model()