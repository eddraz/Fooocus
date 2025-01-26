from typing import Optional, List, Dict, Any
from PIL import Image
import modules.flags as flags
import modules.config as config

class AdvancedParameters:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str = "",
        style_selections: List[str] = None,
        performance_selection: str = "Speed",
        aspect_ratios_selection: str = "1152×896",
        image_number: int = 1,
        image_seed: int = -1,
        sharpness: float = 2.0,
        guidance_scale: float = 7.0,
        base_model_name: str = None,
        refiner_model_name: str = None,
        lora_model_name: str = None,
        input_image: Optional[Image.Image] = None,
        input_mask: Optional[Image.Image] = None,
        inpaint_method: str = "Inpaint",
        upscale_method: str = None,
        scale_factor: float = None
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.style_selections = style_selections or []
        self.performance_selection = performance_selection
        self.aspect_ratios_selection = aspect_ratios_selection
        self.image_number = image_number
        self.image_seed = image_seed
        self.sharpness = sharpness
        self.guidance_scale = guidance_scale
        self.base_model_name = base_model_name or config.default_base_model_name
        self.refiner_model_name = refiner_model_name or config.default_refiner_model_name
        self.lora_model_name = lora_model_name
        self.input_image = input_image
        self.input_mask = input_mask
        self.inpaint_method = inpaint_method
        self.upscale_method = upscale_method
        self.scale_factor = scale_factor

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate the parameters"""
        if self.performance_selection not in flags.performance_selections:
            raise ValueError(f"Invalid performance selection. Must be one of {flags.performance_selections}")
        
        if self.aspect_ratios_selection not in flags.aspect_ratios:
            raise ValueError(f"Invalid aspect ratio selection. Must be one of {flags.aspect_ratios}")
        
        if self.inpaint_method not in flags.inpaint_options:
            raise ValueError(f"Invalid inpaint method. Must be one of {flags.inpaint_options}")
        
        if self.upscale_method and self.upscale_method not in flags.upscale_methods:
            raise ValueError(f"Invalid upscale method. Must be one of {flags.upscale_methods}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary"""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "style_selections": self.style_selections,
            "performance_selection": self.performance_selection,
            "aspect_ratios_selection": self.aspect_ratios_selection,
            "image_number": self.image_number,
            "image_seed": self.image_seed,
            "sharpness": self.sharpness,
            "guidance_scale": self.guidance_scale,
            "base_model_name": self.base_model_name,
            "refiner_model_name": self.refiner_model_name,
            "lora_model_name": self.lora_model_name,
            "inpaint_method": self.inpaint_method if self.input_mask else None,
            "upscale_method": self.upscale_method,
            "scale_factor": self.scale_factor
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedParameters':
        """Create an instance from a dictionary"""
        return cls(**data)
