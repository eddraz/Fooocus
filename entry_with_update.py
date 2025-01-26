import os
import sys
import platform
import signal

from modules.launch_util import fooocus_assert, check_system

fooocus_assert(check_system())

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['PYTHONPATH'] = os.path.abspath(".")

from args_manager import args
import modules.flags
import modules.config

# Disable the visual interface
args.disable_gradio_queue = True

from modules.api_endpoints import app as api_app
import uvicorn

def run_api_server():
    """Start the FastAPI server"""
    host = args.listen or '127.0.0.1'
    port = args.port or 8888
    
    print(f"\nFooocus API Server starting on http://{host}:{port}")
    print("Available endpoints:")
    print("  - POST /api/v1/inpaint-clothing")
    print("\nPress Ctrl+C to quit")
    
    uvicorn.run(api_app, host=host, port=port)

if __name__ == "__main__":
    run_api_server()
