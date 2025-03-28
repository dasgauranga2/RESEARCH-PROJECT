from datasets import load_dataset
import json
import random
import re

# load the dataset
dataset = load_dataset("openbmb/UltraFeedback", split='train')
#print(dataset[0])

# list to save the data
result_data = []

# minimum score of a response to be considered good
GOOD_THRESHOLD = 7
# maximum score of a response to be considered bad
BAD_THRESHOLD = 5
# maximum score difference between responses to be considered a tie
TIE_THRESHOLD = 2

for entry in dataset:
    # get the instruction to be given to the language model
    instruction = entry['instruction']
    good_responses = []
    bad_responses = []

    for response in entry['completions']:
        overall_score = response['overall_score']
        response_text = response['response']

        # skip empty text
        if not response_text.strip():
            continue

        # remove emojis and symbols
        response_text = re.sub(r'[^\w\s.,!?\'"-]', '', response_text)

        if overall_score >= GOOD_THRESHOLD:
            good_responses.append((overall_score, response_text))
        elif overall_score <= BAD_THRESHOLD:
            bad_responses.append((overall_score, response_text))

    good_responses = sorted(good_responses, key=lambda x: x[0])
    bad_responses = sorted(bad_responses, key=lambda x: x[0])

    # check if there are responses that have clear preferences
    if len(good_responses) > 0 and len(bad_responses) > 0:

        for i in range(1,len(good_responses)):
            # check if there are tied preferences
            if good_responses[i][0]-good_responses[i-1][0] < TIE_THRESHOLD and good_responses[i][0] > good_responses[i-1][0]:

                # add the sample such that responses are tied
                result_data.append({
                    "instruction": instruction,
                    "rejected": good_responses[i-1][1],
                    "chosen": good_responses[i][1],
                    "label": 1 # label = 1 means tied responses
                })

                # add the sample such that responses have clear preferences
                result_data.append({
                    "instruction": instruction,
                    "rejected": random.choice(bad_responses)[1], # pick a random bad response
                    "chosen": good_responses[i][1],
                    "label": 0 # label = 0 means clear preference responses
                })

# pick only the first 1000 samples
result_data = result_data[:2000]

with open('ultrafeedback.json', 'w', encoding='utf-8') as file:
    json.dump(result_data, file, indent=4, ensure_ascii=False)