from datasets import load_dataset
import json

dataset = load_dataset("openbmb/RLHF-V-Dataset", split="train")

#print(dataset[0])
#print(dataset[1])

save_data = []

for i, item in enumerate(dataset):
    # get the image
    image = item['image'].convert("RGB")
    image.save(f"./rlhf_images/{i}.jpg")

    # text data
    text_data = json.loads(item['text'])

    # original question
    question = text_data["question"]
    # bunny question prompt
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question}\n<image> ASSISTANT:"

    save_data.append({
        "img_path": f"{i}.jpg",
        "chosen": text_data["chosen"],
        "rejected": text_data["rejected"],
        "prompt": prompt
    })

# save the data
with open('./rlhf_data.json', 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=4)