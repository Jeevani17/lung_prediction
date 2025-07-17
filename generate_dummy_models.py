"""
Generate dummy model files for demonstration purposes when TensorFlow is not available
"""
import os

def create_dummy_model_file(filename):
    """Create a dummy model file for demonstration"""
    with open(filename, 'w') as f:
        f.write("# Dummy model file for demonstration\n")
        f.write("# This would normally be a trained TensorFlow model\n")
    print(f"✅ Created dummy model file: {filename}")

def main():
    # Create dummy model files if they don't exist
    model_files = [
        'pneumonia-detection-model.h5',
        'cancer-detection-model.h5'
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            create_dummy_model_file(model_file)
        else:
            print(f"ℹ️ Model file already exists: {model_file}")

if __name__ == "__main__":
    main()