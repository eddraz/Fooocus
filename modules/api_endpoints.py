from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from PIL import Image
import io
import numpy as np
import base64
from typing import Optional, List
import json
import asyncio

from modules.flags import inpaint_mask_models
from extras.inpaint_mask import generate_mask_from_image
import modules.config
import modules.flags as flags
import modules.async_worker as worker
import modules.inpaint_worker as inpaint_worker
import modules.meta_parser
import modules.advanced_parameters as advanced_parameters

app = FastAPI(title="Fooocus API", description="API for Fooocus image generation and editing")

def image_to_base64(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

@app.post("/api/v1/generate")
async def generate_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    style_selections: str = Form("[]"),  # JSON string of style selections
    performance_selection: str = Form("Speed"),
    aspect_ratios_selection: str = Form("1152×896"),
    image_number: int = Form(1),
    image_seed: int = Form(-1),
    sharpness: float = Form(2.0),
    guidance_scale: float = Form(7.0),
    base64_image: Optional[str] = Form(None),  # For img2img
    base64_mask: Optional[str] = Form(None),   # For inpainting
    inpaint_method: Optional[str] = Form("Inpaint"),
    outpaint_selections: Optional[str] = Form("[]"),  # JSON string for outpaint directions
    advanced_params: Optional[str] = Form("{}")  # JSON string for advanced parameters
):
    """Generate images using text prompts and optional input images"""
    try:
        # Parse JSON inputs
        style_selections = json.loads(style_selections)
        outpaint_selections = json.loads(outpaint_selections)
        advanced_params = json.loads(advanced_params)
        
        # Prepare task arguments
        task_args = advanced_parameters.AdvancedParameters(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style_selections=style_selections,
            performance_selection=performance_selection,
            aspect_ratios_selection=aspect_ratios_selection,
            image_number=image_number,
            image_seed=image_seed,
            sharpness=sharpness,
            guidance_scale=guidance_scale
        )

        # Handle img2img and inpainting
        if base64_image:
            input_image = base64_to_image(base64_image)
            task_args.input_image = input_image
            
            if base64_mask:
                mask_image = base64_to_image(base64_mask)
                task_args.input_mask = mask_image
                task_args.inpaint_method = inpaint_method
                
        # Create and process task
        task = worker.AsyncTask(args=[task_args])
        results = await worker.process_generate_forever(task)
        
        # Convert results to base64
        output_images = []
        for img in results:
            output_images.append(image_to_base64(img))
            
        return JSONResponse({
            "images": output_images,
            "parameters": task_args.to_dict()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    inpaint_method: str = Form("Inpaint"),
    style_selections: str = Form("[]"),
    performance_selection: str = Form("Speed"),
    guidance_scale: float = Form(7.0),
    advanced_params: Optional[str] = Form("{}")
):
    """Inpaint an image using a mask"""
    try:
        # Read and process input files
        image_data = await image.read()
        mask_data = await mask.read()
        input_image = Image.open(io.BytesIO(image_data))
        mask_image = Image.open(io.BytesIO(mask_data))
        
        # Parse JSON inputs
        style_selections = json.loads(style_selections)
        advanced_params = json.loads(advanced_params)
        
        # Prepare inpainting parameters
        task_args = advanced_parameters.AdvancedParameters(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style_selections=style_selections,
            performance_selection=performance_selection,
            guidance_scale=guidance_scale,
            input_image=input_image,
            input_mask=mask_image,
            inpaint_method=inpaint_method
        )
        
        # Process inpainting
        task = worker.AsyncTask(args=[task_args])
        results = await inpaint_worker.process_inpaint(task)
        
        # Convert results to base64
        output_images = []
        for img in results:
            output_images.append(image_to_base64(img))
            
        return JSONResponse({
            "images": output_images,
            "parameters": task_args.to_dict()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/upscale")
async def upscale(
    image: UploadFile = File(...),
    upscale_method: str = Form("4x_UltraSharp"),
    scale_factor: float = Form(2.0),
    prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(""),
):
    """Upscale an image with optional prompt guidance"""
    try:
        # Read and process input file
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data))
        
        # Prepare upscale parameters
        task_args = advanced_parameters.AdvancedParameters(
            prompt=prompt if prompt else "",
            negative_prompt=negative_prompt,
            input_image=input_image,
            upscale_method=upscale_method,
            scale_factor=scale_factor
        )
        
        # Process upscaling
        task = worker.AsyncTask(args=[task_args])
        results = await worker.process_upscale(task)
        
        # Convert result to base64
        output_image = image_to_base64(results[0])
            
        return JSONResponse({
            "image": output_image,
            "parameters": task_args.to_dict()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/styles")
async def get_styles():
    """Get available style presets"""
    return JSONResponse({
        "styles": flags.available_styles
    })

@app.get("/api/v1/models")
async def get_models():
    """Get available models"""
    return JSONResponse({
        "base_models": modules.config.model_filenames,
        "refiner_models": ["None"] + modules.config.model_filenames,
        "lora_models": modules.config.lora_filenames
    })

@app.get("/api/v1/config")
async def get_config():
    """Get current configuration"""
    return JSONResponse({
        "default_settings": {
            "base_model": modules.config.default_base_model_name,
            "refiner_model": modules.config.default_refiner_model_name,
            "styles": modules.config.default_styles,
            "performance": modules.config.default_performance,
            "aspect_ratio": modules.config.default_aspect_ratio,
            "guidance_scale": modules.config.default_cfg_scale,
            "sharpness": modules.config.default_sample_sharpness,
            "image_number": modules.config.default_image_number
        },
        "available_settings": {
            "performance_options": flags.performance_selections,
            "aspect_ratio_options": flags.aspect_ratios,
            "inpaint_methods": flags.inpaint_options,
            "upscale_methods": flags.upscale_methods
        }
    })

@app.post("/api/v1/inpaint-clothing")
async def inpaint_clothing(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    cloth_category: str = Form(...),
    negative_prompt: str = Form("") 
):
    """
    Endpoint to perform clothing inpainting:
    1. Receives an image and prompt
    2. Detects clothing using u2net_cloth_seg
    3. Performs inpainting on the detected area
    
    Args:
        image: Input image file
        prompt: Prompt describing the desired modification
        cloth_category: Category of clothing to detect (e.g., 'upper', 'lower', 'full')
        negative_prompt: Optional negative prompt to guide generation
    """
    try:
        # Read and convert image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        input_image = np.array(pil_image)
        
        # Generate mask for clothing
        mask = generate_mask_from_image(
            image=input_image,
            mask_model='u2net_cloth_seg',
            cloth_category=cloth_category,
            extras={},
            sam_options=None
        )
        
        # Prepare parameters for inpainting
        task_args = [
            prompt,  # prompt
            negative_prompt,  # negative prompt
            'fooocus_expansion',  # style
            'none',  # performance
            'dpmpp_2m_sde_gpu',  # sampler
            20,  # steps
            1.0,  # switch
            1,  # batch size
            7.5,  # guidance scale
            input_image,  # input image
            mask,  # mask
            'inpaint',  # inpaint method
            0.5,  # denoising strength
            False,  # mask blur
            0.0,  # mask expansion
            False,  # mask only
            'sd_xl_base_1.0',  # base model
            'sd_xl_refiner_1.0',  # refiner
            'None',  # vae
            [],  # loras
            1.0,  # image sharpness
            True,  # advanced params
            1.5,  # adm guidance
            0.8,  # adm guidance neg
            False,  # advanced params
            False,  # freeu
            1.01,  # b1
            1.02,  # b2
            0.99,  # s1
            1.0,  # s2
        ]
        
        # Create and execute task
        task = worker.AsyncTask(args=task_args)
        worker.async_tasks.append(task)
        
        # Wait for results
        while not task.has_results:
            await asyncio.sleep(0.1)
        
        # Get results
        results = task.results[0]
        if results and len(results) > 0:
            # Convert first result to bytes
            output_image = results[0]
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(content=img_byte_arr, media_type="image/png")
            
        return {"error": "No results generated"}
        
    except Exception as e:
        return {"error": str(e)}
