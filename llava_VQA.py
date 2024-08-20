from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open('dataset/images/synpic54610.jpg')

inputs = processor(text=prompt, images=image, return_tensors="pt", num_worker=0)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=15)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
