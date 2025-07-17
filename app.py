#!/usr/bin/env python3
"""
Simplified Medical Image Diagnosis System
Uses only Python standard library modules
"""

import os
import sys
import json
import random
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import cgi
import io

# Simple template engine using string formatting
def render_template(template_content, **kwargs):
    """Simple template rendering using string formatting"""
    for key, value in kwargs.items():
        template_content = template_content.replace('{{ ' + key + ' }}', str(value) if value else '')
        # Handle conditionals
        if isinstance(value, list):
            if value:
                template_content = template_content.replace('{% if ' + key + ' %}', '')
                template_content = template_content.replace('{% endif %}', '')
            else:
                # Remove content between if and endif
                start = template_content.find('{% if ' + key + ' %}')
                end = template_content.find('{% endif %}')
                if start != -1 and end != -1:
                    template_content = template_content[:start] + template_content[end + 11:]
    return template_content

def predict_disease(img_data, model_type):
    """Simulate disease prediction"""
    try:
        # Simulate AI prediction with random but realistic results
        confidence = random.uniform(60, 95)
        
        if model_type == "pneumonia":
            label = "PNEUMONIA DETECTED" if confidence > 75 else "NORMAL"
        elif model_type == "cancer":
            label = "CANCER DETECTED" if confidence > 75 else "NO CANCER DETECTED"
        else:
            label = "UNKNOWN"
            confidence = 0.0
            
        return label, confidence
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error in prediction", 0.0

class MedicalDiagnosisHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_index()
        elif self.path == '/health':
            self.serve_health()
        elif self.path.startswith('/static/'):
            self.serve_static()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/':
            self.handle_upload()
        else:
            self.send_error(404)
    
    def serve_index(self):
        """Serve the main page"""
        template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Image Diagnosis System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px; margin-bottom: 20px; }
        .upload-area:hover { border-color: #667eea; background: #f9f9f9; }
        input[type="file"] { margin: 10px 0; }
        input[type="radio"] { margin: 10px; }
        button { background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #5a67d8; }
        .result { padding: 20px; border-radius: 10px; margin: 20px 0; }
        .result.normal { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
        .result.abnormal { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
        .error { background: #fed7d7; color: #c53030; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .disclaimer { background: #fef5e7; color: #744210; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Image Diagnosis System</h1>
            <p>AI-Powered Pneumonia & Cancer Detection (Demo)</p>
        </div>
        
        <div class="card">
            <form method="POST" enctype="multipart/form-data">
                <div class="upload-area">
                    <h3>📤 Upload Medical Image</h3>
                    <input type="file" name="file" accept="image/*" required>
                    <p>Select a chest X-ray or medical image</p>
                </div>
                
                <h3>🔍 Select Analysis Type</h3>
                <label><input type="radio" name="disease" value="pneumonia" required> Pneumonia Detection</label><br>
                <label><input type="radio" name="disease" value="cancer" required> Cancer Detection</label><br>
                
                <button type="submit">🔬 Analyze Image</button>
            </form>
        </div>
        
        {{ error_message }}
        {{ prediction_result }}
        
        <div class="disclaimer">
            <strong>⚠️ Educational Demo:</strong> This is a demonstration system for educational purposes only. 
            The AI models shown are simulated and should never be used for actual medical diagnosis. 
            Always consult qualified healthcare professionals for medical evaluation.
        </div>
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        rendered = render_template(template, error_message='', prediction_result='')
        self.wfile.write(rendered.encode())
    
    def handle_upload(self):
        """Handle file upload and prediction"""
        try:
            # Parse multipart form data
            content_type = self.headers['content-type']
            if not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Expected multipart/form-data")
                return
            
            # Get content length
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse form data manually (simplified)
            boundary = content_type.split('boundary=')[1].encode()
            parts = post_data.split(b'--' + boundary)
            
            disease_type = None
            file_data = None
            
            for part in parts:
                if b'name="disease"' in part:
                    disease_type = part.split(b'\r\n\r\n')[1].split(b'\r\n')[0].decode()
                elif b'name="file"' in part and b'filename=' in part:
                    file_data = part.split(b'\r\n\r\n')[1].split(b'\r\n--')[0]
            
            if not disease_type or not file_data:
                self.serve_error("Please select both a file and disease type")
                return
            
            # Make prediction
            prediction, confidence = predict_disease(file_data, disease_type)
            
            # Serve result
            self.serve_result(prediction, confidence, disease_type)
            
        except Exception as e:
            self.serve_error(f"Error processing request: {str(e)}")
    
    def serve_result(self, prediction, confidence, disease_type):
        """Serve prediction results"""
        result_class = "normal" if "NORMAL" in prediction or "NO CANCER" in prediction else "abnormal"
        
        result_html = f"""
        <div class="result {result_class}">
            <h3>📊 Analysis Results</h3>
            <h2>{prediction}</h2>
            <p><strong>Analysis Type:</strong> {disease_type.title()} Detection</p>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        </div>
        """
        
        template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Image Diagnosis System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .result { padding: 20px; border-radius: 10px; margin: 20px 0; }
        .result.normal { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
        .result.abnormal { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
        .disclaimer { background: #fef5e7; color: #744210; padding: 15px; border-radius: 5px; margin: 10px 0; }
        button { background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Image Diagnosis System</h1>
            <p>AI-Powered Pneumonia & Cancer Detection (Demo)</p>
        </div>
        
        <div class="card">
            {{ prediction_result }}
            <a href="/" style="background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block;">🔄 Analyze Another Image</a>
        </div>
        
        <div class="disclaimer">
            <strong>⚠️ Educational Demo:</strong> This is a demonstration system for educational purposes only. 
            The AI models shown are simulated and should never be used for actual medical diagnosis. 
            Always consult qualified healthcare professionals for medical evaluation.
        </div>
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        rendered = render_template(template, prediction_result=result_html)
        self.wfile.write(rendered.encode())
    
    def serve_error(self, error_msg):
        """Serve error message"""
        error_html = f'<div class="error">❌ {error_msg}</div>'
        
        template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Image Diagnosis System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
        .error { background: #fed7d7; color: #c53030; padding: 15px; border-radius: 5px; margin: 10px 0; }
        button { background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Image Diagnosis System</h1>
            <p>AI-Powered Pneumonia & Cancer Detection (Demo)</p>
        </div>
        
        {{ error_message }}
        <a href="/" style="background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block;">🔄 Try Again</a>
    </div>
</body>
</html>"""
        
        self.send_response(400)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        rendered = render_template(template, error_message=error_html)
        self.wfile.write(rendered.encode())
    
    def serve_health(self):
        """Serve health check"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        health_data = {
            'status': 'healthy',
            'models_loaded': ['pneumonia', 'cancer'],
            'mode': 'demo'
        }
        self.wfile.write(json.dumps(health_data).encode())
    
    def serve_static(self):
        """Serve static files (placeholder)"""
        self.send_error(404)

def main():
    """Start the server"""
    port = 5000
    server = HTTPServer(('0.0.0.0', port), MedicalDiagnosisHandler)
    print(f"🏥 Medical Image Diagnosis System starting on http://localhost:{port}")
    print("📝 Demo mode: Using simulated AI models for educational purposes")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.server_close()

if __name__ == '__main__':
    main()