from openai import OpenAI
import argparse
import json
import time
import re
import base64
import mimetypes

# RUN THIS SCRIPT USING THE FOLLOWING COMMAND -
'''
CUDA_VISIBLE_DEVICES=1 python MMHal-Bench/eval_gpt4_images.py --response ./MMHal-Bench/responses/mdpo_bunny_results.json --evaluation ./MMHal-Bench/gpt_evaluation/mdpo_eval_gpt4.json --gpt-model gpt-4.1-2025-04-14
'''

with open("./MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# client = OpenAI(
#   api_key="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"
# )

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     store=True,
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "What is in this image?"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                     }
#                 }
#             ]
#         }
#     ]
# )

# print(completion.choices[0].message)

template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

You will be provided with the actual image and the model’s response to the user's question. Your evaluation should consider:
1. Whether the response is informative and relevant.
2. Whether the response includes hallucinations — i.e., false claims about the image content.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Question
{}

### LMM Response to Evaluate
{}
'''

# function to get the image path from url
def load_image_path(image_src):
    # extract the image name
    image_name = image_src.split('/')[-1]

    # path of image
    image_path = './MMHal-Bench/images/' + image_name
    
    return image_path

# open image file and convert it base64 url data
def image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"  # safe default
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='responses/idefics_80b.json', help='response file containing images, questions, and model responses')
    parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
    #parser.add_argument('--api-key', type=str, required=True)
    parser.add_argument('--gpt-model', type=str, default='gpt-4-0314')
    args = parser.parse_args()

    client = OpenAI(
        api_key=API_KEY
    )

    # load json file
    with open(args.response, 'r') as f:
        records = json.load(f)

    assert len(records) == 96

    # ask GPT-4 to evaluate
    responses = []
    for i, record in enumerate(records):
        #image_content = ', '.join(record['image_content'])
        image_path = load_image_path(record['image_src'])
        image_url = image_to_data_url(image_path)
        input_text = template.format(record['question'], record['model_answer'])
        # print(input_text)

        try:
            response = client.chat.completions.create(
                model=args.gpt_model,
                #store=True,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": input_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                temperature=0.0,
            )
        except Exception as e:
            print(e)
            print('CANNOT GENERATE RESPONSE', end='\n')
            responses.append(("ERROR", record['model_answer']))
            continue

        print(i, response.choices[0].message.content, flush=True, end='\n')
        responses.append((response.choices[0].message.content, record['model_answer']))
        time.sleep(1)

    # model response scores
    scores = []
    # output to be saved
    outputs = []
    # iterate through the gpt evaluation responses
    for i, (gpt_response, model_response) in enumerate(responses):
        #response = response['choices'][0]['message']['content']
        gpt_response_clean = gpt_response.replace('**', '').lower()
        # use regular expression
        # 'rating:' looks for this string
        # '\s*' allows any number of whitespace or newline characters
        # '([0-6])' looks for a digit between 0 and 6
        match = re.search(r'rating:\s*([0-6])', gpt_response_clean)
        if match:
            # this retrieves the score
            score = int(match.group(1))
            scores.append(score)
            outputs.append({
                'model_answer': model_response,
                'gpt_evaluation': gpt_response,
                'rating': score
            })
        else:
            print('WARNING: multiple or zero scores found')
            print(i, gpt_response_clean, end='\n')
            scores.append(0)
            outputs.append({
                'model_answer': model_response,
                'gpt_evaluation': gpt_response,
                'rating': 0
            })

    hallucination = []
    for s in scores:
        if s >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)

    scores_each = [[] for _ in range(8)]
    # assuming order of 96 questions is not changed
    for i in range(96):
        question_type = i % 8
        scores_each[question_type].append(scores[i])

    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucination) / len(hallucination)))
    print('Average score for each question type:', ','.join([str(round(sum(scores_each[i]) / len(scores_each[i]), 2)) for i in range(8)]), flush=True)

    # save responses
    if args.evaluation is not None:
        with open(args.evaluation, 'w') as f:
            # save the final output
            final_output = {
                "gpt_model_used": args.gpt_model,
                "average_score": round(sum(scores) / len(scores), 2),
                "average_score_type": [str(round(sum(scores_each[i]) / len(scores_each[i]), 2)) for i in range(8)],
                "hallucination_rate": round(sum(hallucination) / len(hallucination), 2),
                "evaluation": outputs
            }
            json.dump(final_output, f, indent=4)