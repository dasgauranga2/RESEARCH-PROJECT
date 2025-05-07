import nltk
from nltk.stem import WordNetLemmatizer
import json
import spacy
from tqdm import tqdm
import warnings
import argparse
nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings("ignore", category=UserWarning)

# function to get script arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_association", type=str, default='data/relation.json')
    parser.add_argument("--safe_words", type=str, default='data/safe_words.txt')
    parser.add_argument("--inference_data", type=str)
    parser.add_argument("--annotation", type=str, default='data/annotations.json')
    parser.add_argument("--metrics", type=str, default='data/metrics.txt')
    parser.add_argument("--similarity_score", type=float, default=0.8)
    parser.add_argument('--evaluation_type', choices=['a', 'g', 'd', 'de', 'da', 'dr'], help='a: all tasks and dimensions    g: generative task    d: descriminative task    de, da, dr: existence, attribute, relation')
    args = parser.parse_args()
    return args

# function to check if two words are synonyms using spacy
def check_synonyms_word(word1, word2, similarity_score):
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_score

# function to extract lemmatized nouns from a string
# this is because nouns are used for checking hallucinations
def extract_nouns(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns

# gets the necessary metrics and initializes them all to zero
def init():
    metrics = {}
    with open(args.metrics, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split('=')
        if len(parts) == 2:
            variable_name = parts[0].strip()
            variable_value = eval(parts[1].strip())
            metrics[variable_name] = variable_value
            
    return metrics

# main function which is run when the script is executed
def main(args):
    # get the metrics which we are measuring
    metrics = init()

    association = json.load(open(args.word_association, 'r', encoding='utf-8'))

    # list of all possible hallucinations in the entire dataset
    # this will be used to check if a response contains hallucinations
    # we are building a hallucination vocabulary
    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association[word1]:
            hallucination_words.append(word2)
    
    # list of words that are ok and not counted as hallucinations
    global_safe_words = []
    with open(args.safe_words, 'r', encoding='utf-8') as safe_file:
        for line in safe_file:
            line = line.split('\n')[0]
            global_safe_words.append(line)

    # set the dimension which determines which type of task is being evaluated
    # in our case we are only interested in generative task
    # so the 'g' key will be set to True
    dimension = {'g': False,'de': False, 'da': False, 'dr': False}
    if args.evaluation_type == 'a':
        for key in dimension.keys():
            dimension[key] = True
    elif args.evaluation_type == 'g':
        dimension['g'] = True
    elif args.evaluation_type == 'd':
        dimension['de'] = True
        dimension['da'] = True
        dimension['dr'] = True
    else:
        dimension[args.evaluation_type] = True
    
    # file which contains our model's generated responses
    inference_data = json.load(open(args.inference_data, 'r', encoding='utf-8'))
    # file which contains the ground-truth
    # for the generative task, it contains list of nouns which are present in the image
    # and a list of commonly hallucinated objects 
    ground_truth = json.load(open(args.annotation, 'r', encoding='utf-8'))

    #hal_count = 0

    for i in range(len(inference_data)):
        
        # get the id
        id = inference_data[i]['id']
        
        # check if generative task is evaluated
        if ground_truth[id-1]['type'] == 'generative':
            nouns = extract_nouns(inference_data[i]['response'])
            
            after_process_nouns = []
            for noun in nouns:
                if noun in hallucination_words:
                    after_process_nouns.append(noun)
            
            safe_words = []
            safe_list = []
            for idx, word in enumerate(ground_truth[id-1]['truth']):
                safe_words += association[word]
                safe_list += [idx] * len(association[word])
                
            ha_words = []
            ha_list = []
            for idx, word in enumerate(ground_truth[id-1]['hallu']):
                ha_words += association[word]
                ha_list += [idx] * len(association[word])
            
            safe_words += ground_truth[id-1]['truth']
            safe_len = len(ground_truth[id-1]['truth'])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)
            
            ha_words += ground_truth[id-1]['hallu']
            ha_len = len(ground_truth[id-1]['hallu'])
            ha_list += [0] * ha_len
            
            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue
                
                if noun in safe_words:
                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue
                
                if noun in ha_words:
                    for j in range(len(ha_words)):
                        if noun == ha_words[j]:
                            if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break
                
                for j, check_word in enumerate(ha_words):
                    if check_synonyms_word(noun, check_word, args.similarity_score):
                        if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break
                
                flag = False
                for j, check_word in enumerate(safe_words):
                    if check_synonyms_word(noun, check_word, args.similarity_score):
                        flag = True
                        if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if flag == True:
                    continue
            
                safe_flag_list[idx] = 1

            # list to store words in the response that are hallucinations
            hallucinated_words = []

            # loop through all processed nouns from the model's response
            for idx in range(len(after_process_nouns)):
                
                # check if this noun was marked as hallucinated
                if safe_flag_list[idx] == 1:
                    # if yes, add it to the hallucinated word list
                    hallucinated_words.append(after_process_nouns[idx])
            
            print(f"ID: {id}\tHallucinated word list: {hallucinated_words}\n")
            # if len(hallucinated_words) > 0:
            #     hal_count += 1

    #print(f"Hallucination Rate: {(hal_count/len(inference_data))*100:.2f}")

            # metrics['chair_score'] += sum(safe_flag_list)
            # metrics['chair_num'] += len(safe_flag_list)
            # metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
            # metrics['safe_cover_num'] += len(safe_list[-safe_len:])
            # metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
            # metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
            # if sum(safe_flag_list) == 0:
            #     metrics['non_hallu_score'] += 1
            # metrics['non_hallu_num'] += 1

            # list of hallucinations in the generated response
                    

    # if dimension['g']:
    #     CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1)
    #     Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1)
    #     Ha = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1)
    #     Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1)
    #     print("Generative Task:")
    #     print("CHAIR:\t\t", CHAIR)
    #     print("Cover:\t\t", Cover)
    #     print("Hal:\t\t", Ha_p)
    #     print("Cog:\t\t", Ha, "\n")
    

if __name__ == "__main__":
    args = get_args()
    main(args)