import os
import json
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_frame_scene_graph(image_path):
    prompt = ''
    with open('vlms/prompts/generate_frame_scene_graph.txt', 'r') as file:
        prompt = file.read()
    image = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, prompt])

    print(response.text)
    json_res = json.loads(response.text.replace('json', '').replace('```',''))
    return json_res