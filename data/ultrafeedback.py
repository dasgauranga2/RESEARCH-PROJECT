from datasets import load_dataset
import json
import random
import re

# load the dataset
dataset = load_dataset("openbmb/UltraFeedback", split='train')
#print(dataset[0])

# list to save the data
result_data = []

# criteria can be: helpfulness, honesty, instruction_following, truthfulness 
criteria = "truthfulness"

# minimum score of a response to be considered good
GOOD_THRESHOLD = 3.5
# maximum score of a response to be considered bad
BAD_THRESHOLD = 2.5
# # maximum score difference between responses to be considered a tie
# TIE_THRESHOLD = 2

for entry in dataset:
    # get the instruction to be given to the language model
    instruction = entry['instruction']
    # list of good and bad responses
    good_responses = []
    bad_responses = []

    for response in entry['completions']:
        try:
            #score = response['overall_score']
            score = float(response['annotations'][criteria]['Rating'])
        except:
            continue
        response_text = response['response']

        # skip empty text
        if not response_text.strip():
            continue

        # remove emojis and symbols from response text
        response_text = re.sub(r'[^\w\s.,!?\'"-]', '', response_text)

        # determine if response is good or bad
        if score >= GOOD_THRESHOLD:
            good_responses.append((score, response_text))
        elif score <= BAD_THRESHOLD:
            bad_responses.append((score, response_text))

    # sort the responses by their scores
    good_responses = sorted(good_responses, key=lambda x: x[0])
    bad_responses = sorted(bad_responses, key=lambda x: x[0])

    # check if there are responses that have clear preferences
    good_bad = len(good_responses) > 0 and len(bad_responses) > 0
    # check if there are multiple tied good responses
    good_good = len(good_responses) > 1
    # check if there are  multiple tied bad responses
    bad_bad = len(bad_responses) > 1

    if good_bad and (good_good or bad_bad):

        # add all the samples such that responses have clear preferences
        for good in good_responses:
            for bad in bad_responses:
                result_data.append({
                    "instruction": instruction,
                    "rejected": bad[1],
                    "rejected_label": -1, # label = -1 means response is bad
                    "chosen": good[1],
                    "chosen_label": 1 # label = 1 means response is good
                })

        # add all the good samples such that responses are tied
        if good_good:
            for i in range(1, len(good_responses)):
                result_data.append({
                    "instruction": instruction,
                    "rejected": good_responses[i-1][1],
                    "rejected_label": 1,
                    "chosen": good_responses[i][1],
                    "chosen_label": 1
                })

        # add all the bad samples such that responses are tied
        if bad_bad:
            for i in range(1, len(bad_responses)):
                result_data.append({
                    "instruction": instruction,
                    "rejected": bad_responses[i-1][1],
                    "rejected_label": -1,
                    "chosen": bad_responses[i][1],
                    "chosen_label": -1
                })
        
print(f"LENGTH OF ORIGINAL DATA: {len(result_data)}")

random.shuffle(result_data)
# pick only the first 2000 samples
result_data = result_data[:2000]

with open(f"ultrafeedback_{criteria}.json", 'w', encoding='utf-8') as file:
    json.dump(result_data, file, indent=4, ensure_ascii=False)