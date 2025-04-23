# inspect_model.py
import torch
import os
from pathlib import Path

def inspect_model(model_path):
    print(f"\n{'='*60}")
    print(f"Inspecting model file: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        print(f"File size: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")
        
        # Load the model file
        print("\nLoading model file...")
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        print("\nModel Contents:")
        print("=" * 60)
        
        # Print information about each layer
        for key, value in model_data.items():
            print(f"\nLayer: {key}")
            print(f"Type: {type(value)}")
            print(f"Shape: {value.shape}")
            
            # Show sample values but handle different tensor types
            if torch.is_tensor(value):
                print(f"Sample values: {value.flatten()[:5].tolist()}")
                print(f"Min value: {value.min().item():.4f}")
                print(f"Max value: {value.max().item():.4f}")
                print(f"Mean value: {value.mean().item():.4f}")
            
            print("-" * 50)
            
        # Print summary
        print("\nSummary:")
        print(f"Total number of layers: {len(model_data)}")
        total_params = sum(v.numel() for v in model_data.values() if torch.is_tensor(v))
        print(f"Total number of parameters: {total_params:,}")
        
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    # Model path relative to script location
    model_path = "assets/wyckoff_model.pth"
    
    # Try different paths if the first one doesn't work
    possible_paths = [
        "assets/wyckoff_model.pth",
        "../assets/wyckoff_model.pth",
        "./assets/wyckoff_model.pth",
        "wyckoff_model.pth"
    ]
    
    # Try each path until we find the model
    for path in possible_paths:
        if os.path.exists(path):
            inspect_model(path)
            break
    else:
        print("\nError: Could not find model file. Searched in:")
        for path in possible_paths:
            print(f"- {os.path.abspath(path)}")