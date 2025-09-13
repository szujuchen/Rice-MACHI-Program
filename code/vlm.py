import requests
from PIL import Image
import json
import os
import time 
from tqdm import tqdm
import argparse
import pickle

DEFAULT_IMAGE_TOKEN = "<image>"

CHOICE_PROMPT = """{}

Choices:
{}

Answer with the option's letter from the given choices directly."""

QA_PROMPT = """{}

Answer the question using a single word or phrase."""

parser = argparse.ArgumentParser(description='VLM')
parser.add_argument('--device', type=str, default=0)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--thresh', default=0.4, type=float)
parser.add_argument('--mode', choices=["only", "cos", "optional", "route"], required=True)
parser.add_argument('--model', choices=["llava", "cos"], default="llava") # this goes to answer model
args = parser.parse_args()

print(f"--- {args.mode} {args.model}---")

## mode: only -> answer (llava/cos)
## mode: cos  -> region (cos) + answer (llava/cos)
## mode: opt  -> supervisor (llava) + region (cos) + answer (llava/cos)

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["HF_HOME"] = "~/.cache/hf"

import torch
from transformers import AutoTokenizer
from llava.model import *
from util import EfficientVLM

device = "cuda" if torch.cuda.is_available() else "cpu"  

examples = ['validation_Geography_25', 'validation_History_7', 'validation_Agriculture_17', 'validation_Finance_10', 'validation_Sociology_13', 'validation_Energy_and_Power_25', 'validation_Biology_20', 'validation_Computer_Science_3', 'validation_Math_7', 'validation_Basic_Medical_Science_7', 'validation_Clinical_Medicine_14', 'validation_Design_27', 'validation_Computer_Science_24', 'validation_Architecture_and_Engineering_1', 'validation_Math_17', 'validation_Manage_8', 'validation_Manage_5', 'validation_Math_29', 'validation_Math_24', 'validation_Manage_4', 'validation_Manage_26', 'validation_Mechanical_Engineering_6', 'validation_Accounting_2', 'validation_Physics_20', 'validation_Manage_18', 'validation_Public_Health_27', 'validation_Basic_Medical_Science_12', 'validation_Literature_18', 'validation_Accounting_16', 'validation_Energy_and_Power_18', 'validation_Economics_8', 'validation_Geography_8', 'validation_Diagnostics_and_Laboratory_Medicine_6', 'validation_Physics_2', 'validation_Art_10', 'validation_Basic_Medical_Science_30', 'validation_Biology_21']

with torch.no_grad():
    if not ((args.mode == "cos" and args.model == "cos") or (args.mode == "only" and args.model == "cos")):
        model_path = "liuhaotian/llava-v1.5-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor 

    if not (args.mode == "only" and args.model == "llava"):
        region_model_path = "Zuyan/llava-CoS-13B"
        region_tokenizer = AutoTokenizer.from_pretrained(region_model_path, use_fast=False)
        region_model = LlavaLlamaForCausalLM.from_pretrained(
            region_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        mm_use_im_start_end = getattr(region_model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(region_model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            region_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            region_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        region_model.resize_token_embeddings(len(region_tokenizer))

        vision_tower = region_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        region_image_processor = vision_tower.image_processor

    if ".pkl" in args.input:
        with open(f"data/{args.input}", "rb") as f:
            datas = pickle.load(f)
    else:
        with open(f"data/{args.input}", "r") as f:
            datas = json.load(f)
            datas = datas["data"]
        
    # print(type(datas))
    if args.mode == "only" and args.model == "llava":
        vlm = EfficientVLM(
            device=device,
            model=model,
            tok=tokenizer,
            img_processor=image_processor,
            thresh=args.thresh,
        )
    elif args.mode == "only" and args.model == "cos":
        vlm = EfficientVLM(
            device=device,
            model=region_model,
            tok=region_tokenizer,
            img_processor=region_image_processor,
            thresh=args.thresh,
        )
    elif args.mode == "cos" and args.model == "llava":
        vlm = EfficientVLM(
            device=device,
            model=model,
            tok=tokenizer,
            img_processor=image_processor,
            reg_model=region_model,
            reg_tok=region_tokenizer,
            reg_img_processor=region_image_processor,
            thresh=args.thresh,
        )
    elif args.mode == "cos" and args.model == "cos":
        vlm = EfficientVLM(
            device=device,
            model=region_model,
            tok=region_tokenizer,
            img_processor=region_image_processor,
            reg_model=region_model,
            reg_tok=region_tokenizer,
            reg_img_processor=region_image_processor,
            thresh=args.thresh,
        )
    elif args.mode == "optional" and args.model == "llava":
        vlm = EfficientVLM(
            device=device,
            model=model,
            tok=tokenizer,
            img_processor=image_processor,
            reg_model=region_model,
            reg_tok=region_tokenizer,
            reg_img_processor=region_image_processor,
            super_model=model,
            super_tok=tokenizer,
            super_img_processor=image_processor,
            thresh=args.thresh,
        )
    elif args.mode == "optional" and args.model == "cos":
        vlm = EfficientVLM(
            device=device,
            model=region_model,
            tok=region_tokenizer,
            img_processor=region_image_processor,
            reg_model=region_model,
            reg_tok=region_tokenizer,
            reg_img_processor=region_image_processor,
            super_model=model,
            super_tok=tokenizer,
            super_img_processor=image_processor,
            thresh=args.thresh,
        )
    elif args.mode == "route":
        vlm = EfficientVLM(
            device=device,
            super_model=model,
            super_tok=tokenizer,
            super_img_processor=image_processor,
        )
    else:
        raise("Mode and Model not accpeted")


    times = []
    ans = []
    double_check = 0
    multi_image = 0
    routes = []
    probs = []
    count = 0
    for data in tqdm(datas):
        image = None
        question = None
        if data["id"] not in examples:
            continue
        if "mmmu" in args.input:
            images = [data[f"image_{i}"] for i in range(1, 8) if data[f"image_{i}"] is not None]
            if len(images) > 1:
                multi_image += 1
                continue
            image = data["image_1"]
            q = data["question"]
            options = eval(data["options"])
            example = ""
            
            if data["question_type"] == 'multiple-choice':
                start_chr = 'A'
                for option in options:
                    example += f"({start_chr}) {option}\n"
                    start_chr = chr(ord(start_chr) + 1)
            
                question = CHOICE_PROMPT.format(q, example)
            else: 
                if args.model == "llava":
                    question = QA_PROMPT.format(q)
                else:
                    question = q

        else:
            try:
                image = Image.open(requests.get(data["flickr_original_url"], stream=True).raw)
            except:
                try:
                    image = Image.open(requests.get(data["flickr_300k_url"], stream=True).raw)
                finally:
                    if image is None:
                        continue

            if args.model == "llava":
                question = QA_PROMPT.format(data["question"])
            else:
                question = data["question"]

        assert image is not None
        assert question is not None

        if args.mode == "only" and args.model == "llava":
            t, pred = vlm.only_llava(image, question)
        elif args.mode == "only" and args.model == "cos":
            t, pred = vlm.only_cos(image, question)
        elif args.mode == "cos" and args.model == "llava":
            t, pred = vlm.cos_llava(image, question)
        elif args.mode == "cos" and args.model == "cos":
            t, pred = vlm.cos_cos(image, question)
        elif args.mode == "optional" and args.model == "llava":
            t, pred, route = vlm.optional_llava(image, question)
        elif args.mode == "optional" and args.model == "cos":
            t, pred, route = vlm.optional_cos(image, question)
        elif args.mode == "route":
            t, res = vlm.routing(image, question)
        else:
            raise("Mode and Model not accpeted")
        
        times.append(t)
        # probs.append({
        #     data["id"]: res
        # })

        
        image.save(f"ori_{count}.png")
        print(data["id"])
        print(question)
        print(pred)
        print(data["answer"])
        count += 1

        # if "mmmu" in args.input:
        #     ans.append({data["id"]: pred})
        #     if route:
        #         routes.append(data["id"])
        # else:
        #     ans.append({data["question_id"]: pred})
        #     if route:
        #         routes.append(data["question_id"])

    exit(0)  

    print(f"multi image: {multi_image}")
    print(f"second stage: {vlm.route}")
    print(f"total queries: {len(times)}")
    print(f"maxmimum time: {max(times)}")
    print(f"mean time: {sum(times)/len(times)}")
    print(f"minimum time: {min(times)}")
    print(f"total time: {sum(times)}")

    times.insert(0, {
        "route": vlm.route,
        "total q": len(times),
        "max time": max(times),
        "mean time": sum(times)/len(times),
        "min time": min(times),
        "total time": sum(times),
    })

    with open(f"stats/mmmu_route_2_decode.json", "w") as f:
        json.dump(probs, f, indent=4, ensure_ascii=False)

    with open(f"stats/mmmu_time_2_decode.json", "w") as f:
        json.dump(times, f, indent=4, ensure_ascii=False)
    
    # with open(f"stats/{args.output}_ans.json", "w") as f:
    #     json.dump(ans, f, indent=4, ensure_ascii=False)

    # with open(f"stats/{args.output}_time.json", "w") as f:
    #     json.dump(times, f, indent=4, ensure_ascii=False)

    # with open(f"stats/{args.output}_route.json", "w") as f:
    #     json.dump(routes, f, indent=4, ensure_ascii=False)