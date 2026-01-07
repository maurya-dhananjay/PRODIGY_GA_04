import os
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline


if not os.path.exists("output"):
    os.mkdir("output")


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to("cpu")

img = Image.open("input/input_image.jpg").convert("RGB")
img = img.resize((512, 512))


prompt = "realistic photo, good lighting, clear details"


result = pipe(
    prompt=prompt,
    image=img,
    strength=0.5,
    guidance_scale=6
).images[0]

result.save("output/result.jpg")

print("Image generated successfully")

