from openai import OpenAI
import base64
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")

def classify_and_describe_mask(image_path, mask_path):
    model_input = []
    model_input.append({
        "type": "image_url",
        "image_url": {
            "url":  f"data:image/jpeg;base64,{encode_image(image_path)}"
        },
        })
    model_input.append({
        "type": "image_url",
        "image_url": {
            "url":  f"data:image/jpeg;base64,{encode_image(mask_path)}"
        },
        })
    
    prompt = ''
    with open('vlms/prompts/classify_and_describe_mask.txt', 'r') as file:
        prompt = file.read()
    model_input.append({
        "type": "text",
        "text": prompt,
        })
    
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": model_input
        }
    ],
    )
    output = response.choices[0].message.content.replace("\n\n", "\n")
    if output.find('\n') == -1 :
        return output.split('\\n')
    return output.split('\n')

def overlap(bbox1, bbox2):
    if bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0]:
        return False
    if bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1]:
        return False
    return True

def generate_rels(objs, frame_idx, image_path):
    rels = {}
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            if overlap(objs[i].frames[frame_idx]['bbox'], objs[j].frames[frame_idx]['bbox']):
                model_input = []
                model_input.append({
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{encode_image(image_path)}"
                    },
                    })
                
                prompt = ''
                with open('vlms/prompts/generate_rels.txt', 'r') as file:
                    prompt = file.read()
                prompt = prompt.replace("{first_desc}", objs[i].desc)
                prompt = prompt.replace("{sec_desc}", objs[j].desc)
                model_input.append({
                    "type": "text",
                    "text": prompt,
                    })
                
                response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": model_input
                    }
                ],
                )
                output = response.choices[0].message.content

                print('--------------------------------------')
                print(i, j)
                print(prompt)
                print('=> Output:', output)
                print('--------------------------------------')

                rel_keywords = ['ON', 'BESIDE', 'WITHIN', 'NOT TOUCHING']
                obj_pair = f'{i}, {j}'
                if output.lower().index('second') < output.lower().index('first'):
                    obj_pair = f'{j}, {i}'
                for rel_keyword in rel_keywords:
                    if output.find(rel_keyword) > -1 and rel_keyword != 'NOT TOUCHING':
                        rels[obj_pair] = rel_keyword
    return rels


                