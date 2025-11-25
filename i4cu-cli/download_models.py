#!/usr/bin/env python3
"""
Helper script to download pre-trained deepfake detection models.
"""

import os
import sys
import urllib.request
from pathlib import Path
import argparse


def get_models_dir():
    """Get the models directory."""
    models_dir = Path.home() / '.deepfake_detector' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_file(url: str, dest: Path, description: str = ""):
    """Download a file with progress."""
    print(f"\nDownloading {description}...")
    print(f"URL: {url}")
    print(f"Destination: {dest}")
    
    try:
        # Check if URL is accessible
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest, progress_hook)
        print(f"\n✓ Downloaded successfully to {dest}")
        
        # Verify file was downloaded (not empty)
        if dest.stat().st_size > 0:
            return True
        else:
            print("✗ Downloaded file is empty")
            dest.unlink()  # Remove empty file
            return False
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"\n✗ File not found at URL (404)")
        else:
            print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n✗ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_xception(force: bool = False):
    """Download XceptionNet weights."""
    models_dir = get_models_dir()
    dest = models_dir / 'xception.pth'
    
    if dest.exists() and not force:
        print(f"\n✓ Xception model already exists at {dest}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping XceptionNet download.")
            return False
    
    print("\n" + "="*60)
    print("XceptionNet (FaceForensics++) Model Download")
    print("="*60)
    
    # Try known download URLs
    common_urls = [
        # These are example URLs - actual URLs may vary
        "https://github.com/ondyari/FaceForensics/releases/download/v1.0.0/xception.pth",
        "https://drive.google.com/uc?export=download&id=1H73TfV5gQ9Op7CAF1T3n4dfX5Y2hwCjP",  # Example
    ]
    
    print("\nAttempting to download from known sources...")
    for url in common_urls:
        try:
            if download_file(url, dest, "XceptionNet weights"):
                return True
        except:
            continue
    
    # If automatic download fails, provide manual instructions
    print("\n" + "="*60)
    print("Automatic download not available.")
    print("="*60)
    print("\nPlease download XceptionNet weights manually:")
    print("1. Visit: https://github.com/ondyari/FaceForensics")
    print("2. Or check: https://github.com/ondyari/FaceForensics/releases")
    print("3. Download the Xception model weights (.pth file)")
    print(f"4. Place the file at: {dest}")
    print("\nAlternatively:")
    print("- Use MesoNet: python download_models.py --mesonet")
    print("- Or use the tool with heuristic model: python cli.py --model heuristic")
    
    return False


def download_mesonet(force: bool = False):
    """Download MesoNet weights."""
    models_dir = get_models_dir()
    dest = models_dir / 'mesonet.pth'
    
    if dest.exists() and not force:
        print(f"\n✓ MesoNet model already exists at {dest}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping MesoNet download.")
            return False
    
    print("\n" + "="*60)
    print("MesoNet Model Download")
    print("="*60)
    
    # Try known download URLs
    common_urls = [
        # These are example URLs - users may need to find actual sources
        "https://github.com/DariusAf/MesoNet.pytorch/raw/master/pretrained/Meso4_DF.pth",
        "https://github.com/DariusAf/MesoNet.pytorch/raw/master/pretrained/Meso4_F2F.pth",
    ]
    
    print("\nAttempting to download from known sources...")
    for url in common_urls:
        try:
            if download_file(url, dest, "MesoNet weights"):
                return True
        except:
            continue
    
    # If automatic download fails, provide manual instructions
    print("\n" + "="*60)
    print("Automatic download not available.")
    print("="*60)
    print("\nMesoNet weights are available from various sources:")
    print("1. GitHub: https://github.com/DariusAf/MesoNet.pytorch")
    print("2. Search GitHub for 'MesoNet pretrained weights'")
    print("3. Check Kaggle deepfake detection competitions")
    print(f"4. Place the .pth file at: {dest}")
    print("\nNote: MesoNet can work without pretrained weights")
    print("(it will use random initialization, but accuracy will be lower)")
    print("\nFor now, the tool will use MesoNet architecture without pretrained weights.")
    print("This will still work but with lower accuracy than with pretrained weights.")
    
    return False


def list_models():
    """List available models."""
    models_dir = get_models_dir()
    print("\n" + "="*60)
    print("Available Models")
    print("="*60)
    
    models = {
        'clip_vit': 'CLIP-ViT (Recommended - Auto-downloads from Hugging Face)',
        'uia_vit': 'UIA-ViT (Recommended - Auto-downloads from Hugging Face)',
        'face_xray.pth': 'Face X-Ray',
        'xception.pth': 'XceptionNet (FaceForensics++)',
        'mesonet.pth': 'MesoNet',
        'efficientnet.pth': 'EfficientNet',
    }
    
    found = False
    for filename, description in models.items():
        if filename in ['clip_vit', 'uia_vit']:
            # These models auto-download from Hugging Face, check if transformers is available
            try:
                from transformers import CLIPProcessor, ViTImageProcessor
                print(f"✓ {description}")
                print(f"  Status: Available (auto-downloads when first used)")
                found = True
            except ImportError:
                print(f"✗ {description} - Not available")
                print(f"  Install: pip install transformers")
        else:
            path = models_dir / filename
            if path.exists():
                size = path.stat().st_size / (1024 * 1024)  # MB
                print(f"✓ {description}")
                print(f"  Location: {path}")
                print(f"  Size: {size:.2f} MB")
                found = True
            else:
                print(f"✗ {description} - Not found")
                print(f"  Expected: {path}")
    
    if not found:
        print("\nNo models found.")
        print("\nTo download models:")
        print("  python download_models.py --download    # Try to download all")
        print("  python download_models.py --mesonet     # Download MesoNet")
        print("  python download_models.py --xception    # Download XceptionNet")
    else:
        print(f"\nModels directory: {models_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Download pre-trained deepfake detection models'
    )
    
    parser.add_argument(
        '--xception',
        action='store_true',
        help='Download XceptionNet model'
    )
    
    parser.add_argument(
        '--mesonet',
        action='store_true',
        help='Download MesoNet model'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available models'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download all available models (same as --all)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing models'
    )
    
    args = parser.parse_args()
    
    # Handle --download as alias for --all
    if args.download:
        args.all = True
    
    # Default to listing if no action specified
    if args.list or (not args.xception and not args.mesonet and not args.all and not args.download):
        list_models()
        if not args.list:
            print("\nTo download models, use:")
            print("  python download_models.py --download    # Download all")
            print("  python download_models.py --xception    # Download XceptionNet")
            print("  python download_models.py --mesonet     # Download MesoNet")
    
    # Download models
    downloaded_any = False
    
    if args.all or args.xception:
        if download_xception(args.force):
            downloaded_any = True
    
    if args.all or args.mesonet:
        if download_mesonet(args.force):
            downloaded_any = True
    
    if (args.all or args.xception or args.mesonet) and not downloaded_any:
        print("\n" + "="*60)
        print("Note: Automatic downloads may not be available for all models.")
        print("="*60)
        print("\nThe tool can still work with:")
        print("1. MesoNet/EfficientNet architectures (without pretrained weights)")
        print("2. Heuristic model (always available)")
        print("\nFor best results, download pretrained weights manually.")
        print("See MODEL_SETUP.md for detailed instructions.")


if __name__ == '__main__':
    main()

