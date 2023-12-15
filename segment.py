import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import torch
import sys
import mlflow
from PIL import Image, TiffTags


# Tiré de  : https://github.com/facebookresearch/segment-anything/blob/main/notebooks
# /automatic_mask_generator_example.ipynb
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def segment_base_sam(image, print_objects=True, pps=32,pred_iou=0.88,stab_tresh=0.95):
    """

    :param print_objects (bool, default=True): True pour afficher les résultats de la segmentation
    :param image (numpy ndarray): Image en format RGB
    :param pps (int): Nombre de points d'échantillonnage par côté d'image
    :param pred_iou (float): Valeur seuil de la qualité des masques
    :param stab_tresh (float): Valeur seuil de la stabilité des masques
    :return: liste de dictionnaires (masks)
    """

    #sam_checkpoint = '/home/e_asselin/sam_vit_b_01ec64.pth'    #sur laptob
    
    sam_checkpoint='/home/EA/Documents/sam_vit_b_01ec64.pth'
    #sam_checkpoint = '../../models/sam_vit_b_01ec64.pth'

    #sam_vit_b_01ec64.pth sinon

    model_type = "vit_b"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=pps,pred_iou_thresh=pred_iou,stability_score_thresh=stab_tresh)

    masks = mask_generator.generate(image)

    # Affichage des objets détectés

    if print_objects:

        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        show_anns(masks)
        plt.savefig('image_segmentee.png')
        plt.title('Image segmentee')
        plt.axis('off')
        
        plt.show()

        #plt.savefig('image_segmentee2.png')
        #lt.imsave('image_sementee3.png')
                
        # Pour Mlflow -------------------------------------------
        #plt.savefig('image_segmentee.png')
        image=Image.open('image_segmentee.png')
        #mlflow.log_image(image, "segmentee.png")

    return masks