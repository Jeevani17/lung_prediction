"""
Create a sample medical image for testing the application
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_xray():
    # Create a 512x512 grayscale image that looks like an X-ray
    width, height = 512, 512
    
    # Create base image with dark background
    img = Image.new('L', (width, height), color=20)
    draw = ImageDraw.Draw(img)
    
    # Add some chest-like structures
    # Ribcage outline
    for i in range(6):
        y_pos = 100 + i * 40
        # Left ribs
        draw.arc([50, y_pos, 200, y_pos + 30], 0, 180, fill=180, width=3)
        # Right ribs
        draw.arc([312, y_pos, 462, y_pos + 30], 0, 180, fill=180, width=3)
    
    # Spine
    draw.line([256, 80, 256, 450], fill=200, width=8)
    
    # Lung areas (darker regions)
    draw.ellipse([80, 120, 220, 350], fill=60)
    draw.ellipse([292, 120, 432, 350], fill=60)
    
    # Heart shadow
    draw.ellipse([200, 180, 320, 300], fill=40)
    
    # Add some texture/noise to make it look more realistic
    pixels = np.array(img)
    noise = np.random.normal(0, 10, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    return img

def main():
    # Create sample X-ray image
    sample_img = create_sample_xray()
    sample_img.save('sample_image.jpeg', 'JPEG', quality=85)
    print("✅ Created sample_image.jpeg")
    
    # Also create a copy in the uploads folder
    os.makedirs('static/uploads', exist_ok=True)
    sample_img.save('static/uploads/sample_xray.jpeg', 'JPEG', quality=85)
    print("✅ Created static/uploads/sample_xray.jpeg")

if __name__ == "__main__":
    import os
    main()