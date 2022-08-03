import json
json_pth = './dataset_specs/cu_birds_dataset_spec.json'
with open(json_pth, 'r') as load_f:
    load_dict = json.load(load_f)
print(load_dict.keys())
print(load_dict['__class__'])
