import requests
import os
import mimetypes
import json
from openai import OpenAI, RateLimitError, InternalServerError
import random

# get the OpenAI API key
with open("./mDPO/MMHal-Bench/api.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()

# get the Google API key
with open("./SOAT/google_api.txt", "r") as f:
    GOOGLE_API_KEY = f.read().strip()

# get the Google CSE ID
with open("./SOAT/google_cse.txt", "r") as f:
    GOOGLE_CSE_ID = f.read().strip()

# openai client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# function to get the most important object
def most_imp_object(client, response_text):
    prompt = f"""From the response text select one physical OBJECT CATEGORY from a caption-like paragraph about an image.
        Return ONLY an object noun phrase that is:
        - a concrete, countable object
        - include color/size/quantity adjectives if explicitly present (e.g., "large", "black", "two")
        - NO articles ("a", "an", "the")
        - NO grouping words ("bunch", "bunches", "group", "pair", "set", "collection", "stack", "pile")
        - Use plural ONLY if a numeric quantity > 1 is explicit; otherwise singular
        - Do NOT output activities, attributes, places, rooms, brands, or abstract nouns
        - Prefer an explicitly named, visible object. If multiple, choose the most imprtant object.
        - Max 4 words, lowercase

        Response Text: {response_text}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content

# function to get an object that looks visually similar
def hallucinate(client, noun_phrase):
    prompt = f"""You are given one noun phrase that names the main visible object in an image.

        You have to return ONE noun phrase that names a different object category but is visually similar to the input.
        Keep all explicit modifiers (color/size/quantity) but replace the HEAD NOUN with a sibling category.
        Examples of sibling swaps: cat→dog, car→bus, apple→pear, laptop→tablet, monitor→television, desk→table.

        STRICT RULES
        - Preserve any explicit adjectives and numbers from the input (e.g., "large", "black", "two").
        - Keep plurality: plural only if a numeric quantity > 1 is explicitly present.
        - No articles ("a", "an", "the").
        - No grouping words ("bunch", "bunches", "group", "pair", "set", "collection", "stack", "pile").
        - Output must be a concrete, countable object category, not an activity/place/brand/part.
        - Must NOT be the same category, a synonym, or a hypernym/hyponym of the input.
        - Prefer a close visual sibling under the same coarse supercategory:
        * animals: cat↔dog↔fox↔wolf
        * vehicles: car↔bus↔van↔truck↔tram↔train
        * furniture: desk↔table↔dresser↔cabinet
        * screens: monitor↔television↔tablet
        * handhelds: game controller↔remote control↔joystick
        * fruits/veg: banana↔plantain, apple↔pear, orange↔tangerine
        - Lowercase; max 4 words.
        - Return ONLY the final phrase.

        Noun Phrase: {noun_phrase}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content

# function to perform google search
# using query and how many results to return
def google_image_search(query, num_results=5):
    # google search url
    url = "https://www.googleapis.com/customsearch/v1"

    # search parameters
    params = {
        "q": query,
        "cx": GOOGLE_CSE_ID,
        "key": GOOGLE_API_KEY,
        "searchType": "image",   # This makes it image search
        "num": num_results       # Number of results (max 10 per request)
    }
    
    # make the request and get the result
    response = requests.get(url, params=params)
    # convert data to json
    data = response.json()

    if "items" not in data:
        print("No results found.")
        return []

    results = []
    for item in data["items"]:
        results.append({
            "title": item["title"], # image title
            "link": item["link"],           # image url
            "thumbnail": item["image"]["thumbnailLink"],  # Thumbnail preview
            "context": item["image"]["contextLink"]       # Page where it appears
        })
    
    return results
    
# open the training data json file
with open('./mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# set of allowed extensions for image
ALLOWED_EXTS = {".jpg", ".jpeg"}

# list to save data
save_data = []

# iterate through the data
for sample in random.sample(data, 500):
    # chosen response
    chosen = sample['chosen']
    # image name
    image_name = sample['img_path']

    # get the most important object in the image
    mio = most_imp_object(openai_client, chosen)
    # suggest a visually similar object
    vso = hallucinate(openai_client, mio)

    #print(f"CHOSEN: {chosen}")
    print(f"MOST IMPORTANT OBJECT: {mio}")
    print(f"VISUALLY SIMILAR OBJECT: {vso}\n\n")    

    # perform google image search
    image_results = google_image_search(vso, num_results=10)

    if len(image_results)==0:
        print(f'NO RESULTS FOUND FOR QUERY: {vso}')
        continue

    # check if we get an image
    success = False
    
    for img in image_results:
        # get the image link
        img_url = img['link']

        try:
            # download the image
            response = requests.get(img_url, timeout=20)
            #print(response)
        except:
            print(f"IMAGE DOWNLOAD ERROR: {img_url}")
            continue

        # check if download is successful
        if response.status_code == 200:
            # detect content-type from server response
            content_type = response.headers.get("Content-Type", "")
            mime = content_type.split(";")[0].strip().lower()
            ext = mimetypes.guess_extension(mime)

            if ext in ALLOWED_EXTS:
                # name of searched image to be saved
                search_image_name = f"{image_name.split('.')[0]}{ext}"

                # path of searched image file
                search_file_path = './SOAT/eval_images/' + search_image_name
                # save the image
                with open(search_file_path, "wb") as f:
                    f.write(response.content)

                # save searched image name
                original = sample.copy()
                original['search_img_path'] = search_image_name
                original['most_important_object'] = mio
                original['visually_similar_object'] = vso
                save_data.append(original)

                success = True

        # if one image is downloaded successfully save it
        if success:
            break
    
    if not success:
        print(f'COULD NOT RETRIEVE IMAGES FOR: {vso}')

# save the generated data in a json file
with open('./SOAT/soat_data.json', 'w') as f:
    json.dump(save_data, f, indent=4)