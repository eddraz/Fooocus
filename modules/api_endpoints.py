from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import numpy as np
from modules.flags import inpaint_mask_models
from extras.inpaint_mask import generate_mask_from_image
import modules.config
import modules.flags as flags
import modules.worker as worker

app = FastAPI()

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
            dino_prompt_text=None,
            sam_model=None,
            box_threshold=None,
            text_threshold=None,
            sam_max_detections=None,
            dino_erode_or_dilate=0,
            dino_debug=False
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
