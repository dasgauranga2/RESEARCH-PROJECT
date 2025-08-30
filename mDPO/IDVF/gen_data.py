import torch
from diffusers import StableDiffusion3Pipeline
import json
from openai import OpenAI
import json
import random

# set device
device = torch.device("cuda")

# get the OpenAI API key
with open("MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# open the text file for logging errors
with open("./IDVF/log.txt", "w") as f:
    f.write("LOG STARTED\n")
# open the text file in append mode
log_file = open("./IDVF/log.txt", "a")

# openai client
openai_client = OpenAI(api_key=API_KEY)

# function to summarize a response text
def summarize(client, response_text):
    prompt = (
        "Summarize only the visual content described in the response text in one paragraph in less than 60 words. "
        "Do not include any assumptions, causes, intentions, or speculative interpretations. "
        "Only describe what is visually observable in the scene.\n\n"
        f"Response Text: {response_text}"
    )
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )

            return response.choices[0].message.content
        except Exception as error:
            log_file.write(f"ERROR: {str(error)}\n")
            log_file.flush()
            raise
            
# function to create a hallucinated version of response text
def hallucinate(client, response_text):
    prompt = (
        "Pick the the most important object (the main subject of the scene) in the response text\n"
        "Rewrite the response text by replacing that object with another visually similar object, while keeping the sentence structure and overall tone similar.\n"
        "Separate the important object and rewritten response text using only '__'.\n\n"
        f"Response Text: {response_text}"
    )

    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )

            return response.choices[0].message.content
        except Exception as error:
            log_file.write(f"ERROR: {str(error)}\n")
            log_file.flush()
            raise

# load the stable diffusion model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", 
    torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

# open the training data json file
with open('./data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# list to save the data
save_data = []

for sample in random.sample(data, 500):
    original = sample.copy()

    # get the chosen response
    chosen_response = original['chosen']
    # summarize the chosen response
    summarized_chosen_response = summarize(openai_client, chosen_response)
    # create a hallucinated version by replacing one of the object
    hallucinated_output = hallucinate(openai_client, summarized_chosen_response)

    # check if the sepration exists
    if '__' not in hallucinated_output:
        log_file.write(f"NO SEPARATION: {hallucinated_output}\n")
        log_file.flush()
        continue
    
    # separate the object and text
    hallucinated_output = hallucinated_output.split('__')

    try:
        # get the object that was replaced
        replaced_obj = hallucinated_output[0].strip()
        # hallucinated response with the object replaced
        hallucinated_response = hallucinated_output[1].strip()

        if ':' in replaced_obj:
            log_file.write(f"COLON IN OBJECT: {replaced_obj}\n")
            replaced_obj = replaced_obj.split(':')[-1].strip()
            log_file.write(f"NEW OBJECT: {replaced_obj}\n")
            log_file.flush()

        if ':' in hallucinated_response:
            log_file.write(f"COLON IN RESPONSE TEXT: {hallucinated_response}\n")
            hallucinated_response = hallucinated_response.split(':')[-1].strip()
            log_file.write(f"NEW OBJRESPONSE TExT: {hallucinated_response}\n")
            log_file.flush()

        # generate the hallucinated image
        hall_image = pipe(
            hallucinated_response, # text prompt for generation
            height=640,
            width=640,
            num_inference_steps=40, # no. of denoising steps for finer details
            guidance_scale=10.0, # strength of prompt adherence 
        ).images[0]
        
        # print(f"SUMMARY CHOSEN RESPONSE: {summarized_chosen_response}\n")
        # print(f"HALLUCINATED OBJECT: {hallucinated_obj}")
        # print(f"HALLUCINATED RESPONSE: {hallucinated_response}\n\n")

        # append the generated data
        original['chosen_summarized'] = summarized_chosen_response
        original['replaced_obj'] = replaced_obj
        original['hallucinated_response'] = hallucinated_response
        save_data.append(original)

        # image name
        img_name = original['img_path']
        # save the hallucinated image
        hall_image.save('./IDVF/hall_images/' + img_name)
    except Exception as error:
        log_file.write(f"ERROR: {str(error)}\n")
        log_file.flush()
        raise

 # save the generated data in a json file
with open('./IDVF/idvf_data.json', 'w') as f:
    json.dump(save_data, f, indent=4)

log_file.close()