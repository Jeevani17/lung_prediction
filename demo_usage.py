"""
Demo script to show how to use the medical diagnosis system
"""
import requests
import os

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")

def main():
    print("🏥 Medical Image Diagnosis System - Demo Usage")
    print("=" * 50)
    
    print("\n1. Starting the Flask application...")
    print("   Run: python app.py")
    
    print("\n2. Open your browser and go to: http://localhost:5000")
    
    print("\n3. Upload a medical image (or use the sample_image.jpeg)")
    
    print("\n4. Select disease type (Pneumonia or Cancer detection)")
    
    print("\n5. Click 'Analyze Image' to get AI diagnosis")
    
    print("\n6. View results with confidence scores")
    
    print("\n📝 Note: This is a demonstration system with simulated AI models")
    print("   for educational purposes only.")

if __name__ == "__main__":
    main()