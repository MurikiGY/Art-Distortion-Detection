import os
from collections import deque

working_dir = './'
init_dir = working_dir+"data"

fila = deque([init_dir])
items= [0, 0]
while len(fila) > 0:
    item = fila.popleft()
    if os.path.isdir(item):
        for i in os.listdir(item):
            if item != ".":
                fila.append(item+"/"+i)

    elif os.path.isfile(item):
        if "original" in item:
            items[0] += 1
        elif "modified" in item:
            items[1] += 1
        else:
            print("ERROR: ", item, " neither original nor modified (???)")
    else:
        print("ERROR: analyzing something that is neither a file nor a dir (", item, ")")
print(items)

class ImageDataset(Dataset):

    def __init__(self, annotations, img_dir, transform=None, target_transform=None):
        

    def __len__(self):
        return items[0]+items[1]

    def __getitem__(self, idx):
