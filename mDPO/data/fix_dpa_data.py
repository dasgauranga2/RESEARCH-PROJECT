import json

# load original dpa dataset file
with open("./data/dpa_data.json", "r") as f:
    full_data = json.load(f)

# proper formatting for the prompt
def wrap_prompt(question_text):
    question_text = question_text.replace('<image>','').replace('\n','')
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: <image>\n{question_text} ASSISTANT:"
    )

# select only the required fields
filtered_data = []
for item in full_data:
    new_item = {
        "prompt": wrap_prompt(item["question"]),
        "chosen": item["correct_answer"],
        "rejected": item["hallucinated_answer"],
        "img_path": item["image"]
    }
    filtered_data.append(new_item)

# Save to new file
with open("./data/dpa_data_fixed.json", "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Saved {len(filtered_data)} formatted entries to 'dpa_data_aligned.json'")
