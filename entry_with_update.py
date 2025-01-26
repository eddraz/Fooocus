import os
import sys
from pathlib import Path

# Set up environment
root = str(Path(__file__).parent)
sys.path.append(root)
os.chdir(root)

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['PYTHONPATH'] = os.path.abspath(".")

# Import necessary modules
import args_manager
from modules.api_endpoints import app
import uvicorn
from pyngrok import ngrok

def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False

def main():
    """Start the FastAPI server"""
    args = args_manager.args
    
    # In Colab, we need to bind to 0.0.0.0
    host = "0.0.0.0" if is_google_colab() else (args.listen or "127.0.0.1")
    port = args.port or 8888
    
    print(f"\nFooocus API Server")
    
    # Set up ngrok if running in Colab
    if is_google_colab():
        try:
            # Install ngrok if not already installed
            os.system("pip install pyngrok")
            from pyngrok import ngrok
            
            # Start ngrok
            public_url = ngrok.connect(port)
            print(f"\nPublic URL: {public_url}")
            print(f"Local URL: http://{host}:{port}")
        except Exception as e:
            print(f"Error setting up ngrok: {str(e)}")
            print(f"Falling back to local URL: http://{host}:{port}")
    else:
        print(f"Starting server on http://{host}:{port}")
    
    print("\nAvailable endpoints:")
    print("  - POST /api/v1/generate - Generate images from text or image")
    print("  - POST /api/v1/inpaint - Inpaint images with mask")
    print("  - POST /api/v1/upscale - Upscale images")
    print("  - GET  /api/v1/styles - Get available styles")
    print("  - GET  /api/v1/models - Get available models")
    print("  - GET  /api/v1/config - Get current configuration")
    print("\nPress Ctrl+C to quit")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
