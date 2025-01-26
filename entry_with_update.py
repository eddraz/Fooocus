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

def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False

def setup_ngrok(port):
    """Set up ngrok tunnel"""
    try:
        # Install ngrok if not already installed
        os.system("pip install pyngrok")
        from pyngrok import ngrok, conf
        
        # Check if NGROK_AUTH_TOKEN is in environment variables
        auth_token = os.getenv('NGROK_AUTH_TOKEN')
        if not auth_token:
            print("\nNGROK_AUTH_TOKEN not found!")
            print("Please set your ngrok auth token by running these commands in a new cell:")
            print("1. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken")
            print("2. Run this command:")
            print("   import os; os.environ['NGROK_AUTH_TOKEN'] = 'your_auth_token_here'")
            print("\nThen run this script again.")
            sys.exit(1)
            
        # Configure ngrok
        conf.get_default().auth_token = auth_token
        
        # Start ngrok tunnel
        public_url = ngrok.connect(port)
        print(f"\nNgrok tunnel established!")
        print(f"Public URL: {public_url}")
        return public_url
        
    except Exception as e:
        print(f"\nError setting up ngrok: {str(e)}")
        print("Please make sure you have a valid auth token from https://dashboard.ngrok.com/get-started/your-authtoken")
        sys.exit(1)

def main():
    """Start the FastAPI server"""
    args = args_manager.args
    
    # In Colab, we need to bind to 0.0.0.0
    host = "0.0.0.0" if is_google_colab() else (args.listen or "127.0.0.1")
    port = args.port or 8888
    
    print(f"\nFooocus API Server")
    
    # Set up ngrok if running in Colab
    if is_google_colab():
        public_url = setup_ngrok(port)
        print(f"Local URL: http://{host}:{port}")
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
