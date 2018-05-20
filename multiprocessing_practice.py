import multiprocessing as mp
from multiprocessing import Pool
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

count = 0
s = {}
def long_for_loops():
    global count
    print("I want to work")
    for x in range(10000):
        print("aye")
        for y in range(1000):
            count += 1
    count +=1
    return count

def run_through_list(bbox):
    yo = [1,2,3,4]

    value = {}

    for x in yo:
        value = bbox * yo
'''
def create_bbox_for_class(annotations)     
    pool = Pool(1)
'''

def load_json(json_path):
    with open(json_path, "r") as file:
        json_data = json.load(file)
    
    return json_data

def get_dict_of_bbox(json_data):
    bbox_list = {}

   # for ann in json_
    #    bbox_list.append()

def test(lists):
    global s
    count = 0
    for x in lists:
        if str(x) not in s:
            s[str(x)] = count
        else:
            count += 1
            s[str(x)] = count
    print(s)

def make_overlap_dict(json_data, process_num):
    anns_dict = {}

    for ann in tqdm(json_data["annotations"]):
        if ann["category_id"] not in anns_dict:
            anns_dict[ann["category_id"]] = [ann]
        else:
            anns_dict[ann["category_id"]].append(ann)
    
    overlap_dict = json_data

    overlap_dict ={}
    pool = Pool(process_num)

    values = pool.map(get_over_lap_list, anns_dict.values())

    i = 0
    for cat in tqdm(json_data["categories"]):
        id = cat["id"]
        if id not in overlap_dict:
            overlap_dict[id] = values[i]
    return overlap_dict

def get_over_lap_list(anns):
    list_to_hold_overlap = []

    for ann in tqdm(anns):
        for ann2 in anns:
            if ann != ann2:
                list_to_hold_overlap.append(ann["bbox"][0] +ann2["bbox"][0])

    return list_to_hold_overlap


json_data = load_json("/home/ben/Desktop/shared/practice_area/annotations/instances_val2017.json")

print(len(json_data["categories"]))

#print(json_data["categories"][0]["id"])
dicti = make_overlap_dict(json_data, 2)

print(dicti[json_data["categories"][0]["id"]][0])

print(json_data["annotations"][0]["bbox"])
print(json_data["annotations"][1]["bbox"])
print(json_data["annotations"][0]["category_id"])
print(json_data["annotations"][1]["category_id"])
print(json_data["categories"][0]["id"])

colormap = sns.color_palette("husl", len(json_data["categories"]))
cid = 0

for key in dicti:
    sns.distplot(dicti[key], color=colormap[cid % len(json_data["categories"])], label = key)

    cid +=1

plt.show()


