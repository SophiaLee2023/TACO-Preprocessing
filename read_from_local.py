import json, os, cv2 as cv

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

os.chdir("TACO-data/")

def read_json(file_path: str) -> dict:
    with open(file_path, "r") as json_file:
        return json.load(json_file)
    
def condense_annotations(file_path: str) -> dict:
    raw_data: dict = read_json(file_path) # https://cocodataset.org/#format-data
    
    categories: dict = {}
    for type_info in raw_data["categories"]:
        categories[type_info["id"]] = (type_info["name"], type_info["supercategory"])

    annotations: dict = {} # NOTE: id: (image_path, [(bounding_box, category_id, (category, supercategory)),])

    for image_info in raw_data["images"]: 
        annotations[image_info["id"]] = (image_info["file_name"], list())

    for image_data in raw_data["annotations"]:
        id: int = image_data["image_id"]
        category_id: int = image_data["category_id"]
        annotations[id][1].append((image_data["bbox"], category_id, categories[category_id]))

    return annotations

ANNOTATIONS: dict = condense_annotations("annotations.json")

def display_image(image_path: str, bbox_list: list[tuple]) -> None:
    mpl.rcParams["toolbar"] = "None"
    plt.gcf().canvas.manager.set_window_title(image_path)
    plt.axes([0, 0, 1, 1]) # remove margins
    plt.axis("off") 

    plt.imshow(cv.imread(image_path))
    ax = plt.gca() # get the current reference

    for bbox_data in bbox_list: # add all the bounding boxes to the image
        x, y, width, height = bbox_data[0]
        ax.add_patch(Rectangle((x, y), width, height, edgecolor="w", facecolor="none"))
        ax.annotate(bbox_data[2][0], (x, y), color="w", fontsize=10, ha="left", va="bottom")
    plt.show()

for id, image_data in ANNOTATIONS.items():
    print(f"{id}: {image_data[1]}")
    display_image(image_data[0], image_data[1])