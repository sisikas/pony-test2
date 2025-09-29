#!/usr/bin/env python3
"""
Simple script to check image metadata
Usage: python check_metadata.py image.png
"""

import sys
from PIL import Image
from PIL.ExifTags import TAGS

def check_image_metadata(image_path):
    """Check metadata of an image file"""
    try:
        # Open image
        with Image.open(image_path) as img:
            print(f"📸 Image: {image_path}")
            print(f"📐 Size: {img.size}")
            print(f"🎨 Mode: {img.mode}")
            print(f"📁 Format: {img.format}")
            
            # Check for EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                print("\n📊 EXIF Data:")
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    print(f"   {tag}: {value}")
            
            # Check for other metadata
            if hasattr(img, 'info'):
                print("\n📋 Image Info:")
                for key, value in img.info.items():
                    print(f"   {key}: {value}")
            
            # Check for custom metadata
            if hasattr(img, 'text'):
                print("\n📝 Custom Text:")
                for key, value in img.text.items():
                    print(f"   {key}: {value}")
                    
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_metadata.py <image_file>")
        sys.exit(1)
    
    check_image_metadata(sys.argv[1])

