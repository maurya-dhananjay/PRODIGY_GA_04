import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os


os.makedirs("output", exist_ok=True)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to("cpu")

init_image = Image.open("input/input_image.jpg").convert("RGB")
init_image = init_image.resize((512, 512))

prompt = (
    "ultra realistic portrait, natural lighting, sharp focus, "
    "high detail, professional photography, cinematic look"
)

negative_prompt = (
    "blurry, distorted face, extra limbs, deformed body, "
    "low quality, dark image, horror, abstract"
)


result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.5,
    guidance_scale=8.0
).images[0]

result.save("output/generated_image.jpg")

print("High-quality image generated successfully!")
