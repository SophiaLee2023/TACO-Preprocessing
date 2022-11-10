import json, requests
from PIL import Image 

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def read_json(file_path: str) -> dict: 
    with open(file_path, "r") as json_file:
        return json.load(json_file)
    
def condense_annotations(file_path: str) -> dict:
    raw_data: dict = read_json(file_path) # https://cocodataset.org/#format-data for reference
    
    categories: dict = {}
    for type_info in raw_data["categories"]:
        categories[type_info["id"]] = (type_info["name"], type_info["supercategory"])

    annotations: dict = {} # NOTE: id: (image_url, [(bounding_box, category_id, (category, supercategory)),], (image_width, image_height))

    for image_info in raw_data["images"]: 
        annotations[image_info["id"]] = (image_info["flickr_url"], list(), (image_info["width"], image_info["height"]))

    for image_data in raw_data["annotations"]:
        id: int = image_data["image_id"]
        category_id: int = image_data["category_id"]
        annotations[id][1].append((image_data["bbox"], category_id, categories[category_id]))

    return annotations

def image_from_url(url: str) -> Image:
    return Image.open(requests.get(url, stream=True).raw)

def display_image(image_url: str, bbox_list: list[tuple]) -> None:
    mpl.rcParams["toolbar"] = "None"
    plt.gcf().canvas.manager.set_window_title(image_url)
    plt.axes([0, 0, 1, 1])
    plt.axis("off") 

    plt.imshow(image_from_url(image_url))
    ax = plt.gca() # get the current reference

    for bbox_data in bbox_list: # add all the bounding boxes to the image
        x, y, width, height = bbox_data[0]
        ax.add_patch(Rectangle((x, y), width, height, edgecolor="w", facecolor="none"))
        ax.annotate(bbox_data[2][0], (x, y), color="w", fontsize=10, ha="left", va="bottom")
    plt.show()

def run_demo(file_path: str = "TACO-data/annotations.json") -> None: 
    for id, image_data in condense_annotations(file_path).items():
        print(f"{id}: {image_data[1]} {image_data[2]}")
        display_image(image_data[0], image_data[1])