import json

with open("TextVQA_0.5.1_val.json", "r") as f:
    data = json.load(f)

with open("TextVQA_val.json", "w") as f:
    json.dump(data, f, indent=4)