# import some common libraries
import sys
import numpy as np
import os, json, cv2, random, sys
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Any

import torch, detectron2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import Metadata

# Custom utilities
from importlib.machinery import SourceFileLoader
import uuid

def initialize_detic():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("mps available:", torch.backends.mps.is_available())
    print("detectron2:", detectron2.__version__)

    # Setup detectron2 logger
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # Running under the Detic repo
    SCRIPT_DIR = Path(__file__).resolve().parent
    os.chdir(SCRIPT_DIR)

    # Setup Detic PATHS
    sys.path.insert(0, 'third_party/CenterNet2/')

initialize_detic()

# Detic libraries
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test
from Detic.detic.modeling.text.text_encoder import build_text_encoder

def build_detic():
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
    predictor = DefaultPredictor(cfg)
    return predictor

def build_general_detic(predictor: DefaultPredictor) -> DefaultPredictor:
    # Setup the model's vocabulary using build-in datasets
    BUILDIN_CLASSIFIER = {
        'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
        'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
        'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
        'coco': 'datasets/metadata/coco_clip_a+cname.npy',
    }

    BUILDIN_METADATA_PATH = {
        'lvis': 'lvis_v1_val',
        'objects365': 'objects365_v2_val',
        'openimages': 'oid_val_expanded',
        'coco': 'coco_2017_val',
    }

    vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
    metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
    classifier = BUILDIN_CLASSIFIER[vocabulary]
    num_classes = len(metadata.thing_classes)
    reset_cls_test(predictor.model, classifier, num_classes)
    return predictor


def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def build_custom_detic(predictor: DefaultPredictor, metadata: Metadata) -> DefaultPredictor:
    vocabulary = 'custom'
    # metaname = f"custom_{uuid.uuid4().hex[:8]}"
    # metadata = MetadataCatalog.get(metaname)
    # metadata.thing_classes = ['avocada', 'carrot', 'tomato', 'raspberry']  # Change here to try your own vocabularies!
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    reset_cls_test(predictor.model, classifier, num_classes)
    # Reset visualization threshold
    output_score_threshold = 0.3
    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

    return predictor

def build_metadata(object_list: list[str]) -> Metadata:
    metaname = f"custom_{uuid.uuid4().hex[:8]}"
    metadata = MetadataCatalog.get(metaname)
    metadata.thing_classes = object_list  # Change here to try your own vocabularies!
    return metadata

def inference_detic(predictor: DefaultPredictor, image: Any, object_list: list[str]=[]):
    # Run model and show results
    outputs = predictor(image)
    return outputs

def cv2_imshow(a):  # --- IGNORE ---
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    This function converts images from BGR to RGB format and displays them inline.
    Args:
        a: The image to be displayed, as a numpy array.
    """

    if a.ndim == 2:  # grayscale
        plt.imshow(a, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def visualize_detic(im: Any, outputs: dict, metadata: Metadata):
    v = Visualizer(im[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

def print_detic_results(outputs: dict, metadata: Metadata):
    print(outputs["instances"].pred_classes) # class index
    print([metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]) # class names
    print(outputs["instances"].scores)
    print(outputs["instances"].pred_boxes)

def predict_pipe_line(image: Any, object_list: list[str]=[]) -> dict[str, Any]:
    predictor = build_detic()
    if object_list:
        metadata = build_metadata(object_list)
        predictor = build_custom_detic(predictor, metadata)
    else:
        predictor = build_general_detic(predictor)
    outputs = inference_detic(predictor, image, metadata.thing_classes)
    print_detic_results(outputs, metadata)

    # DEBUG: visualize results
    visualize_detic(image, outputs, metadata)
    return outputs

if __name__ == "__main__":
    predictor = build_detic()
    # predictor, metadata = build_general_detic(predictor)
    # Or use custom vocabulary
    object_list = ['avocada', 'carrot', 'tomato', 'raspberry']
    metadata = build_metadata(object_list)
    predictor = build_custom_detic(predictor, metadata)

    image_path = "food.jpg"  # change to your input image path!
    im = cv2.imread(image_path)
    outputs = inference_detic(predictor, im, metadata.thing_classes)
    print_detic_results(outputs, metadata)
    visualize_detic(im, outputs, metadata)