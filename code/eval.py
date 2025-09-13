import json
import re
from tqdm import tqdm
from PIL import Image
import requests
import pickle
import numpy as np
import random

class EvalAIAnswerProcessor:
    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_list, gt_list):
        pred_scores = []
        unique_answer_scores = self._compute_answer_scores(gt_list)
        for entry in pred_list:
            pred_answer = self.answer_processor(entry)
            score = unique_answer_scores.get(pred_answer, 0.0)
            if score == 0:
                continue
            pred_scores.append(score)

        if len(pred_scores) == 0:
            return 0
        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy

    def eval_pred(self, pred_list, gt):
        pred_scores = []

        for entry in pred_list:
            pred_answer = self.answer_processor(entry)
            if gt in pred_answer:
                return 1
        return 0

def eval_multi(pred, gt, options):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        pred = pred.strip(char)
    pred = " " + pred + " "

    opts = eval(options)
    choices = [chr(ord('A') + i) for i in range(len(opts))]

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in pred:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in choices: # e.g., A B C D
            if f' {choice} ' in pred:
                candidates.append(choice)

    if len(candidates) == 0 and len(pred.split()) > 5:
        for i, ans in enumerate(opts):
            if ans.lower() in pred.lower():
                candidates.append(chr(ord('A') + i))
                index_ans = False

    if len(candidates) == 0:
        return 0
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = pred.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = pred.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                order = ord(can) - ord('A')
                index = pred.lower().rfind(opts[order].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    # print(pred, pred_index)
    if pred_index == gt:
        return 1
    else:
        return 0




# with open("data/TextVQA_val.json", "r") as f:
#     datas = json.load(f)
#     datas = datas["data"]

with open("data/mmmu_data.pkl", "rb") as f:
    datas = pickle.load(f)

truth = {}
for data in datas:
    # truth[data["question_id"]] = {
    #     "url1": data["flickr_original_url"],
    #     "url2": data["flickr_300k_url"],
    #     "ans": data["answers"],
    # }

    truth[data["id"]] = {
        "type": data["question_type"],
        "ans": data["answer"],
        "options": data["options"]
    }
all_ids = list(truth.keys())

only = {}
with open("stats/ori/mmmu/only_cos_ans.json", "r") as f:
    preds = json.load(f)
    for pred in preds:
        for key, val in pred.items():
            only[key] = val

cos = {}
with open("stats/ori/mmmu/cos_cos_ans.json", "r") as f:
    preds = json.load(f)
    for pred in preds:
        for key, val in pred.items():
            cos[key] = val

probs = {}
with open("stats/mmmu_route_1.json", "r") as f:
    routes = json.load(f)
    for route in routes:
        for key, val in route.items():
            probs[key] = val

lapped = list(set(only.keys()) & set(cos.keys()))
print(len(lapped))

scores = []
multi = 0
routed = 0
evaluator = TextVQAAccuracyEvaluator()
mod = []
for question_id in tqdm(lapped):
    # ground_truth = truth[int(question_id)]["ans"]
    ground_truth = truth[question_id]["ans"]
    # prob = probs[question_id]
    yes_p, no_p = probs[question_id]
    this_route = False
    # if prob.strip().upper() != "NO":
    # if yes_p > 0.15:
    if yes_p / no_p > 1:
        routed += 1
        this_route = True
        pred = cos[question_id]
    else:
        pred = only[question_id]

    # parse = re.findall(r'"(.*?)"', pred)
    # all_pred = pred.split(" ")
    # all_pred.append(pred)
    # if len(parse) > 0:
    #     all_pred.extend(parse)
    # scores.append(evaluator.eval_pred_list(all_pred, ground_truth))

    if truth[question_id]["type"] == "multiple-choice":
        multi += 1
        scores.append(eval_multi(pred, ground_truth, truth[question_id]["options"]))
    else:
        parse = re.findall(r'"(.*?)"', pred)
        all_pred = pred.split(" ")
        all_pred.append(pred)
        if len(parse) > 0:
            all_pred.extend(parse)
        scores.append(evaluator.eval_pred(all_pred, ground_truth))

    if this_route and scores[-1] == 1:
        only_pred = only[question_id]
        if truth[question_id]["type"] == "multiple-choice":
            old_score = eval_multi(only_pred, ground_truth, truth[question_id]["options"])
        else:
            parse = re.findall(r'"(.*?)"', only_pred)
            all_pred = only_pred.split(" ")
            all_pred.append(only_pred)
            if len(parse) > 0:
                all_pred.extend(parse)
            old_score = evaluator.eval_pred(all_pred, ground_truth)
        
        if old_score != 1:
            mod.append(question_id)
            print(question_id, ground_truth, pred, only_pred)

final_score = sum(scores) / len(scores)
print('Samples: {}\nMultiple Choices: {}\nRouted: {}\nAccuracy: {:.2f}%\n'.format(len(lapped), multi, routed, 100. * final_score))
print(mod)

# validation_Literature_18

# set_correct = []
# for ind, preds in enumerate([llava, cos]):
#     scores = []
    
#     multi = 0
#     correct = []
#     for question_id in tqdm(lapped):
#         pred = preds[question_id]
#         # ground_truth = truth[int(question_id)]["ans"]
#         # parse = re.findall(r'"(.*?)"', pred)
#         # all_pred = pred.split(" ")
#         # all_pred.append(pred)
#         # if len(parse) > 0:
#         #     all_pred.extend(parse)
#         # scores.append(evaluator.eval_pred_list(all_pred, ground_truth))
        
#         ground_truth = truth[question_id]["ans"]
#         if truth[question_id]["type"] == "multiple-choice":
#             multi += 1
#             scores.append(eval_multi(pred, ground_truth, truth[question_id]["options"]))
#         else:
#             parse = re.findall(r'"(.*?)"', pred)
#             all_pred = pred.split(" ")
#             all_pred.append(pred)
#             if len(parse) > 0:
#                 all_pred.extend(parse)
#             scores.append(evaluator.eval_pred(all_pred, ground_truth))

#         if scores[-1] > 0:
#             correct.append(question_id)
#             # correct.append(int(question_id))

#     set_correct.append(correct)

#     final_score = sum(scores) / len(scores)
#     print('Type: {} (0: llava, 1: cos)'.format(ind))
#     print('Samples: {}\nMultiple Choices: {}\nAccuracy: {:.2f}%\n'.format(len(lapped), multi, 100. * final_score))


# print(len(set_correct[0]))
# print(len(set_correct[1]))
# print(len(set(set_correct[0])&set(set_correct[1])))
# print(len(set(set_correct[1])-set(set_correct[0])))

# llava_wrong = set(all_ids)-set(set_correct[0])
# print("llava uncorrected: ", len(llava_wrong))
# print("routed: ", len(routed))
# print("detected routed in wrong: ", len(set(llava_wrong)&set(routed)))
# print("detected routed in correct: ", len(set(set_correct[0])&set(routed)))
# print("correct routed still correct: ", len(set(set_correct[0])&set(routed)&set(set_correct[1])))
# print("wrong routed but correct: ", len(set(llava_wrong)&set(routed)&set(set_correct[1])))
    
