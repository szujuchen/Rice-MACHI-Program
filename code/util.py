import requests
from PIL import Image
import json
import os
import time 
from tqdm import tqdm
import torch
from llava.model import *
import numpy as np
import random

seed = 42
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# region question 
# qs = DEFAULT_IMAGE_TOKEN + '\n' + 'To answer the question: "' + qs + '", what is the region of interest in the image?'
# (USER, qs) (ASSISTANT, None)

# QA question
# qs1 = DEFAULT_IMAGE_TOKEN + '\n' + 'To answer the question: "' + cur_prompt + '", what is the region of interest in the image?'
# qs2 = f'The region of interest in the image is <image>\n. Answer the question: "{cur_prompt}"'

# conversation template
# conv_vicuna_v1 = Conversation(
#     system="A chat between a curious user and an artificial intelligence assistant. "
#     "The assistant gives helpful, detailed, and polite answers to the user's questions.",
#     roles=("USER", "ASSISTANT"),
#     version="v1",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="</s>",
# )

# ask supervisor
# ROUTE_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
# The assistant gives helpful, detailed, and polite answers to the user's questions. 
# USER: {DEFAULT_IMAGE_TOKEN}\nYou are a supervisor. Answer "YES" if the given image needs cropping to get accurate answer for the question: "{question}" Otherwise, answer "NO". 
# ASSISTANT:"""
# ROUTE_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
# The assistant gives helpful, detailed, and polite answers to the user's questions. 
# USER: {DEFAULT_IMAGE_TOKEN}\nYou are a supervisor. Answer "YES" if the given image contains too many noises and information for the question: "{question}" Otherwise, answer "NO". 
# ASSISTANT:"""
# ROUTE_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
# The assistant gives helpful, detailed, and polite answers to the user's questions. 
# USER: {DEFAULT_IMAGE_TOKEN}\nYou are a supervisor. Answer "YES" if the given image contains enough information to answer the question: "{question}" Otherwise, answer "NO". 
# ASSISTANT:"""

# get region of interest
REGION_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: {DEFAULT_IMAGE_TOKEN}\nTo answer the question: "{question}", what is the region of interest in the image? 
ASSISTANT:"""

# get answer
LLAVA_QA_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: {DEFAULT_IMAGE_TOKEN}\n{question} 
ASSISTANT:"""

# get cos answer
COS_QA_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: The region of interest in the image is {DEFAULT_IMAGE_TOKEN}\n. Answer the question: "{question}"  
ASSISTANT:"""

# get cos
CHAIN_LLAVA_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: {DEFAULT_IMAGE_TOKEN}\nTo answer the question: "{q_reg}", what is the region of interest in the image?
ASSISTANT: {bbox}</s>
USER: The region of interest in the image is {DEFAULT_IMAGE_TOKEN}\n.{question}  
ASSISTANT:"""

CHAIN_COS_PROMPT = """A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: {DEFAULT_IMAGE_TOKEN}\nTo answer the question: "{q_reg}", what is the region of interest in the image?
ASSISTANT: {bbox}</s>
USER: The region of interest in the image is {DEFAULT_IMAGE_TOKEN}\n. Answer the question: "{question}" 
ASSISTANT:"""

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class EfficientVLM():
    def __init__(self, 
        device, model=None, tok=None, img_processor=None, 
        reg_model=None, reg_tok=None, reg_img_processor=None, 
        super_model=None, super_tok=None, super_img_processor=None, thresh=0.4):
        
        self.device = device
        self.model = model
        self.tokenizer = tok
        self.image_processor = img_processor

        self.region_model = reg_model
        self.region_tokenizer = reg_tok
        self.region_image_processor = reg_img_processor

        self.super_model = super_model
        self.super_tokenizer = super_tok
        self.super_image_processor = super_img_processor

        self.image = None
        self.image_emb = None
        self.q = None

        self.route = 0
        self.thresh = thresh

        yeses = ["yes", "Yes", " Yes", " yes", "YES", " YES"]
        nos = ["no", "No", " No", " no", "NO", " NO"]
        # self.yes_ids = [self.super_tokenizer(variant, add_special_tokens=False)["input_ids"][-1] for variant in yeses]
        # self.no_ids = [self.super_tokenizer(variant, add_special_tokens=False)["input_ids"][-1] for variant in nos]
        
    @torch.no_grad()
    def tokenizer_image_token(self, prompt, region=None, image_token_index=-200):
        if region is True:
            bos_token_id = self.region_tokenizer.bos_token_id
            prompt_chunks = [self.region_tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
        elif region is False:
            bos_token_id = self.tokenizer.bos_token_id
            prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
        else:
            bos_token_id = self.super_tokenizer.bos_token_id
            prompt_chunks = [self.super_tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        self.inputs = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device=self.device, non_blocking=True)
        self.input_len = self.inputs.shape[1]

    @torch.no_grad()
    def get_answer(self):
        prompt = LLAVA_QA_PROMPT.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, question=self.q)
        # print(prompt)
        self.tokenizer_image_token(prompt, region=False)
        outputs = self.model.generate(
            self.inputs,
            images=self.image_emb,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True,
        )
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return response.strip()

    @torch.no_grad()
    def get_cos_answer(self):
        prompt = COS_QA_PROMPT.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, question=self.q)

        self.tokenizer_image_token(prompt, region=False)
        outputs = self.model.generate(
            self.inputs,
            images=self.image_emb,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True,
        )
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return response.strip()

    @torch.no_grad()
    def get_cos(self, LLAVA=False):
        if LLAVA:
            prompt = CHAIN_LLAVA_PROMPT.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, q_reg=self.q.split("\n")[0], question=self.q, bbox=self.bbox)
        else:
            prompt = CHAIN_COS_PROMPT.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, q_reg=self.q.split("\n")[0], question=self.q, bbox=self.bbox)

        self.tokenizer_image_token(prompt, region=False)
        # print(self.image_emb.shape)
        # print(self.image_emb_crop.shape)
        outputs = self.model.generate(
            self.inputs,
            images=torch.cat((self.image_emb, self.image_emb_crop), dim=0),
            max_new_tokens=200,
            do_sample=False,
            use_cache=True,
        )
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return response.strip()

    @torch.no_grad()
    def get_region_of_interest(self):
        image_emb = self.region_image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        prompt = REGION_PROMPT.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, question=self.q.split("\n")[0])

        self.tokenizer_image_token(prompt, region=True)
        # print(inputs)
        outputs = self.region_model.generate(
            self.inputs,
            images=image_emb,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True,
        )
        response = self.region_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print("BBOX: ", response)

        return response.strip()

    @torch.no_grad()
    def crop_image(self, bbox):
        if len(bbox) != 4:
            return
        x1, y1, x2, y2 = map(float, bbox)
        x1 = int(x1*self.image.size[0])
        x2 = int(x2*self.image.size[0])
        y1 = int(y1*self.image.size[1])
        y2 = int(y2*self.image.size[1])

        print(bbox)
        self.image = self.image.crop((x1, y1, x2, y2))
        self.image.save(f"crop_{self.route}.png")
        # self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        self.image_emb_crop = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        return
        
    @torch.no_grad()
    def ask_supervisor(self):
        prompt = ROUTE_PROMPT.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, question=self.q.split("\n")[0])
        # print(prompt)

        self.tokenizer_image_token(prompt, region=None)
        # print(inputs)
        outputs = self.super_model.generate(
            self.inputs,
            images=self.image_emb,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True, 
            output_scores=True,    
        )

        logits = outputs.scores[0]
        probs = torch.softmax(logits, dim=-1)
        yes_probs = [probs[:, token_id].item() for token_id in self.yes_ids]
        no_probs = [probs[:, token_id].item() for token_id in self.no_ids]

        if (sum(yes_probs)/len(yes_probs)) / (sum(no_probs)/len(no_probs)) > 1:
            return True
        else:
            return False

    @torch.no_grad()
    def only_llava(self, image, question):
        start = time.time()
        self.q = question
        self.image =  image
        self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        response = self.get_answer()
        end = time.time()
        
        return end-start, response

    @torch.no_grad()
    def only_cos(self, image, question):
        start = time.time()
        self.q = question
        self.image =  image
        self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        response = self.get_cos_answer()
        end = time.time()
        
        return end-start, response

    @torch.no_grad()
    def cos_cos(self, image, question):
        start = time.time()
        self.q = question
        self.image =  image
        self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
    
        self.route += 1
        bbox = self.get_region_of_interest()
        self.bbox = bbox
        self.crop_image(eval(bbox)) 
        response = self.get_cos_answer()

        end = time.time()
        
        return end-start, response

    @torch.no_grad()
    def cos_llava(self, image, question):
        start = time.time()
        self.q = question
        self.image =  image
        self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
    
        self.route += 1
        bbox = self.get_region_of_interest()
        self.bbox = bbox
        self.crop_image(eval(bbox)) 
        response = self.get_answer()

        end = time.time()
        
        return end-start, response

    @torch.no_grad()
    def optional_llava(self, image, question):
        start = time.time()
        self.q = question
        self.image =  image
        self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        noise = self.ask_supervisor()
        if not noise:
            # self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
            response = self.get_answer()
        else:
            self.route += 1
            bbox = self.get_region_of_interest()
            self.bbox = bbox
            self.crop_image(eval(bbox)) 
            # response = self.get_answer()
            response = self.get_cos(LLAVA=True)

        end = time.time()
        
        return end-start, response, noise

    @torch.no_grad()
    def optional_cos(self, image, question):
        start = time.time()
        self.q = question
        self.image =  image
        self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        noise = self.ask_supervisor()
        if not noise:
            # self.image_emb = self.image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
            response = self.get_cos_answer()
        else:
            self.route += 1
            bbox = self.get_region_of_interest()
            self.bbox = bbox
            self.crop_image(eval(bbox)) 
            # response = self.get_cos_answer()
            response = self.get_cos()

        end = time.time()
        
        return end-start, response, noise

    @torch.no_grad()
    def ask_route(self):
        prompt = ROUTE_PROMPT.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, question=self.q.split("\n")[0])

        self.tokenizer_image_token(prompt, region=None)
        
        outputs = self.super_model.generate(
            self.inputs,
            images=self.image_emb,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            # return_dict_in_generate=True, 
            # output_scores=True,    
        )

        # logits = outputs.scores[0]
        # probs = torch.softmax(logits, dim=-1)
        # yes_probs = [probs[:, token_id].item() for token_id in self.yes_ids]
        # no_probs = [probs[:, token_id].item() for token_id in self.no_ids]

        response = self.super_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return response.strip()


    @torch.no_grad()
    def routing(self, image, question):
        start = time.time()
        self.q = question
        self.image =  image
        self.image_emb = self.super_image_processor([self.image], return_tensors="pt")['pixel_values'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        # yes, no = self.ask_route()
        res = self.ask_route()
        end = time.time()
        
        return end-start, res
