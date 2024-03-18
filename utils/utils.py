import numpy as np
from matplotlib import pyplot as plt
import os

# mapping for class type
# def get_class_name_imagenet(c):
#     labels = np.loadtxt('.synset_words.txt', str, delimiter='\t')
#     return ' '.join(labels[c].split(',')[0].split()[1:])
def get_class_name_imagenet(c):
    # Get the directory of the current file (utils.py)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Construct the path to the .txt file
    txt_file_path = os.path.join(current_dir, 'synset_words.txt')
    # Load the labels from the .txt file
    labels = np.loadtxt(txt_file_path, dtype=str, delimiter='\t')
    # Return the class name
    return ' '.join(labels[c].split(',')[0].split()[1:])

# mapping for class type
def get_class_name_coco(c):
    
    coco_classes = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush"
    }
    return coco_classes[c]

def tensor_imshow(inp, title=None, normalize=False, **kwargs):
    """Imshow for Tensor."""

    inp = inp.cpu().numpy().transpose((1, 2, 0))

    if normalize:
        # Mean and std for ImageNet (values given for H,W,channel)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        
    plt.imshow(inp, **kwargs)