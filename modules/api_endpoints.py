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
            'fooocus',  # metadata_scheme
            True,  # save_metadata_to_images
            True,  # save_final_enhanced_image_only
            0,  # inpaint_erode_or_dilate
            False,  # invert_mask_checkbox
            True,  # inpaint_advanced_masking_checkbox
            48,  # inpaint_respective_field
            0.5,  # inpaint_strength
            'v1',  # inpaint_engine
            False,  # inpaint_disable_initial_latent
            False,  # debugging_inpaint_preprocessor
            1.0,  # freeu_s2
            0.99,  # freeu_s1
            1.02,  # freeu_b2
            1.01,  # freeu_b1
            False,  # freeu_enabled
            1.0,  # controlnet_softness
            'v1',  # refiner_swap_method
            100,  # canny_high_threshold
            50,  # canny_low_threshold
            False,  # skipping_cn_preprocessor
            False,  # debugging_cn_preprocessor
            False,  # mixing_image_prompt_and_inpaint
            False,  # mixing_image_prompt_and_vary_upscale
            0.75,  # overwrite_upscale_strength
            0.75,  # overwrite_vary_strength
            0,  # overwrite_height
            0,  # overwrite_width
            0.5,  # overwrite_switch
            0,  # overwrite_step
            'None',  # vae_name
            'normal',  # scheduler_name
            'dpmpp_2m_sde_gpu',  # sampler_name
            1,  # clip_skip
            True,  # adaptive_cfg
            0.5,  # adm_scaler_end
            0.8,  # adm_scaler_negative
            1.5,  # adm_scaler_positive
            False,  # black_out_nsfw
            False,  # disable_seed_increment
            False,  # disable_intermediate_results
            False,  # disable_preview
            mask,  # inpaint_mask_image_upload
            '',  # inpaint_additional_prompt
            input_image,  # inpaint_input_image
            [],  # outpaint_selections
            None,  # uov_input_image
            'None',  # uov_method
            'inpaint',  # current_tab
            True,  # input_image_checkbox
            # LoRA parameters (3 values per LoRA: enabled, name, weight) x 5
            False, '', 1.0,  # LoRA 1
            False, '', 1.0,  # LoRA 2
            False, '', 1.0,  # LoRA 3
            False, '', 1.0,  # LoRA 4
            False, '', 1.0,  # LoRA 5
            0.5,  # refiner_switch
            'sd_xl_refiner_1.0',  # refiner_model_name
            'sd_xl_base_1.0',  # base_model_name
            7.5,  # cfg_scale
            1.0,  # sharpness
            True,  # read_wildcards_in_order
            -1,  # seed
            'png',  # output_format
            1,  # image_number
            '1152×896',  # aspect_ratios_selection
            'Speed',  # performance_selection
            ['fooocus_expansion'],  # style_selections
            negative_prompt,  # negative_prompt
            prompt,  # prompt
            False,  # generate_image_grid
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
