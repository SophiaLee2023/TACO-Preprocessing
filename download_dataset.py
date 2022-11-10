import os, random, shutil
from pathlib import Path
from read_from_url import condense_annotations, image_from_url

os.chdir("TACO-data") # set the current working directory
ANNOTATIONS: dict = condense_annotations("annotations.json")

IMAGE_DIR, LABEL_DIR = "images", "labels" # formatting requirement

def make_directories() -> None:
    for dir_name in IMAGE_DIR, LABEL_DIR: 
        for subdir_name in "train", "val", "test":
            Path(f"{dir_name}/{subdir_name}").mkdir(parents=True, exist_ok=True) 

def to_YOLOv5_annotation(save_path: str, bbox_list: list, image_size: tuple) -> None:
    with open(save_path, "w") as file: # create a new file
        image_width, image_height = image_size

        for i, (bbox, category_id, category) in enumerate(bbox_list): # for each bounding box
            x, y, width, height = bbox
            x_center, y_center = (x + (width / 2)) / image_width, (y + (height / 2)) / image_height

            file.write(f"{category_id} {x_center} {y_center} {width / image_width} {height / image_height}" +\
                ("\n" if (i < len(bbox_list) - 1) else "")) # last line does not need a newline character

def download_dataset() -> None:
    for id, image_data in ANNOTATIONS.items():
        image_url, bbox_list, image_size = image_data

        image_from_url(image_url).save(f"{IMAGE_DIR}/{id}.jpg") # download the image
        to_YOLOv5_annotation(f"{LABEL_DIR}/{id}.txt", bbox_list, image_size) # create the file of annotations

def to_image_and_label_list(id_list: list) -> tuple:
    image_list, label_list = [], []

    for id in id_list: # convert a list of ids to separate image path and label path lists
        image_list.append(f"{IMAGE_DIR}/{id}.jpg")
        label_list.append(f"{LABEL_DIR}/{id}.txt")

    return (image_list, label_list)

def move_files(file_list: list, dir_path: str) -> None:
    for file_path in file_list:
        shutil.move(file_path, dir_path)

def partition_dataset(ratio: tuple = (0.8, 0.1, 0.1)) -> None: # default to 80-10-10% 
    id_list: list = list(ANNOTATIONS.keys())
    random.shuffle(id_list)

    length: int = len(id_list)
    index_1, index_2 = round(ratio[0] * length), length - round(ratio[1] * length) # indices to split at
    train_ids, val_ids, test_ids = id_list[:index_1], id_list[index_1:index_2], id_list[index_2:]
    
    train_images, train_labels = to_image_and_label_list(train_ids)
    move_files(train_images, f"{IMAGE_DIR}/train")
    move_files(train_labels, f"{LABEL_DIR}/train")

    val_images, val_labels = to_image_and_label_list(val_ids)
    move_files(val_images, f"{IMAGE_DIR}/val")
    move_files(val_labels, f"{LABEL_DIR}/val")

    test_images, test_labels = to_image_and_label_list(test_ids)
    move_files(test_images, f"{IMAGE_DIR}/test")
    move_files(test_labels, f"{LABEL_DIR}/test")

make_directories() # create the required directory structure
download_dataset() # download the all the images and create their annotation files (this takes awhile)
partition_dataset() # move all the files to their respective folders