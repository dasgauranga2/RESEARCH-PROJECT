from openai import OpenAI
import argparse
import json
import time

# RUN THIS SCRIPT USING THE FOLLOWING COMMAND -
'''
CUDA_VISIBLE_DEVICES=1 python MMHal-Bench/eval_gpt4.py --response ./MMHal-Bench/responses/mdpo_bunny_results.json --evaluation ./MMHal-Bench/gpt_evaluation/mdpo_eval_gpt4.json --gpt-model gpt-4.1-2025-04-14
'''

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='responses/idefics_80b.json', help='response file containing images, questions, and model responses')
    parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
    #parser.add_argument('--api-key', type=str, required=True)
    parser.add_argument('--gpt-model', type=str, default='gpt-4-0314')
    args = parser.parse_args()

    client = OpenAI(
        api_key="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"
    )

    # load json file
    with open(args.response, 'r') as f:
        records = json.load(f)

    assert len(records) == 96

    # ask GPT-4 to evaluate
    responses = []
    for i, record in enumerate(records):
        #image_content = ', '.join(record['image_content'])
        image_url = record['image_src']
        input_text = template.format(record['question'], record['model_answer'])
        # print(input_text)

        try:
            # response = openai.ChatCompletion.create(
            #     model=args.gpt_model,
            #     messages=[
            #         {"role": "user", "content": input_text}
            #     ],
            #     temperature=0.0,
            # )
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
            print('Cannot generate response')
            responses.append(("ERROR", record['model_answer']))
            continue

        print(i, response.choices[0].message.content, flush=True)
        responses.append((response.choices[0].message.content, record['model_answer']))
        time.sleep(1)

    # # save responses
    # if args.evaluation is not None:
    #     with open(args.evaluation, 'w') as f:
    #         json.dump(responses, f, indent=4)

    # analyze responses
    scores = []
    # output to be saved
    outputs = []
    for i, (gpt_response, model_response) in enumerate(responses):
        #response = response['choices'][0]['message']['content']
        gpt_response_clean = gpt_response.replace('**', '').lower()
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in gpt_response_clean.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
            outputs.append({
                'model_answer': model_response,
                'gpt_evaluation': gpt_response,
                'rating': scores_found[0]
            })
        else:
            print('Warning: multiple or zero scores found')
            print(i, gpt_response_clean)
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