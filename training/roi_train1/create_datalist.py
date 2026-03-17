import os
import random
import json


N_FOLDS = 5

train_data = {"images": [], "labels": [], "fold": []}
test_data = {"images": [], "labels": []}

datalist = {"training": [], "testing": []}

prl_data = {"images": [], "labels": []}
with open("/home/srs-9/Projects/prl_project/training/roi_train1/prl_data.txt", 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        image = line.split(",")[0].strip()
        assert os.path.exists(image)
        label = line.split(",")[1].strip()
        assert os.path.exists(label)
        prl_data['images'].append(image)
        prl_data['labels'].append(label)
        
inds = list(range(len(prl_data['images'])))
random.shuffle(inds)
train_end_ind = 5*len(inds) // 6
for i in range(train_end_ind):
    train_data['images'].append(prl_data['images'][inds[i]])
    train_data['labels'].append(prl_data['labels'][inds[i]])
for i in range(train_end_ind, len(inds)):
    test_data['images'].append(prl_data['images'][inds[i]])
    test_data['labels'].append(prl_data['labels'][inds[i]])
        
        
lesion_data = {"images": [], "labels": []}
with open("/home/srs-9/Projects/prl_project/training/roi_train1/lesion_data.txt", 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        image = line.split(",")[0].strip()
        assert os.path.exists(image)
        label = line.split(",")[1].strip()
        assert os.path.exists(label)
        lesion_data['images'].append(image)
        lesion_data['labels'].append(label)
        
inds = list(range(len(lesion_data['images'])))
random.shuffle(inds)
train_end_ind = 5*len(inds) // 6
for i in range(train_end_ind):
    train_data['images'].append(lesion_data['images'][inds[i]])
    train_data['labels'].append(lesion_data['labels'][inds[i]])
for i in range(train_end_ind, len(inds)):
    test_data['images'].append(lesion_data['images'][inds[i]])
    test_data['labels'].append(lesion_data['labels'][inds[i]])
    

for i in range(len(train_data['images'])):
    train_data['fold'].append(i%N_FOLDS)
    datalist["training"].append({"image": train_data["images"][i], "label": train_data["labels"][i], "fold": train_data["fold"][i]})

for i in range(len(test_data['images'])):
    datalist["testing"].append({"image": test_data["images"][i], "label": test_data["labels"][i]})
    
with open("datalist0.json", 'w') as f:
    json.dump(datalist, f, indent=4)


