"""
A utility file to visualize results from AttributeAlteration operation
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
import matplotlib.gridspec as gridspec

from environment_setup import PROJECT_ROOT_DIR

attribute_list = np.asarray(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                  'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
                  'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                  'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                  'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                  'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                  'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                  'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])


def generate_annotation_text(labels):
    """
    Selects only the labels for which there was a mis-prediction due to perturbations
    :param labels: All the labels produced by the model
    :return: Formatted text for labels producing mis-prediction
    """
    correct_labels = torch.nonzero(labels, as_tuple=False)
    matching_terms = attribute_list[correct_labels]
    # squeeze the extra dimension
    matching_terms = matching_terms.squeeze()
    text = " ".join(matching_terms.tolist())
    return text


def extract_caption(caption):
    """
    A tabular structure to display the mis-predictions
    :param caption: json object having perturbed and original predictions
    :return: Tabular structure with the mis-predictions
    """
    orig_prediction = caption['orig']
    perturbed_prediction = caption['perturb']
    mask = [idx for idx, _ in enumerate(orig_prediction) if orig_prediction[idx] != perturbed_prediction[idx]]
    colLabels = [attribute_list[attr] for attr in mask]
    if len(colLabels) == 0:
        # No entry so every attribute matches
        colLabels = ['', 'For all Attributes']
        cellText = [['Original', 'Same Pred'], ['Perturbed', 'Same Pred']]
        return colLabels, cellText
    # include the category labels
    colLabels.insert(0, ' ')
    orig_prediction.insert(0, 'original')
    perturbed_prediction.insert(0, 'perturbed')
    # There is no easy way of handling text wraps. So,
    colLabels = [label.replace("_", "\n") for label in colLabels]
    orig_pred_text = [orig_prediction[0], *[orig_prediction[x+1] for x in mask]]
    perturb_pred_text = [perturbed_prediction[0], *[perturbed_prediction[x+1] for x in mask]]
    cellText = [orig_pred_text, perturb_pred_text]
    return colLabels, cellText


def visualize_samples(orig_image, perturb_image, caption, entry_name, viz_save_folder):
    """
    Function to save the tabular structure along with the image
    :param orig_image: Original image
    :param perturb_image: perturbed sample
    :param caption: A json object containing original and perturbed predictions
    :param entry_name: An identifier for the input data sample
    :param viz_save_folder: Target folder to save the predictions to
    :return: None
    """
    colLabels, cellText = extract_caption(caption)
    orig_image = orig_image.copy()
    orig_image = cv2.resize(orig_image, (960, 960))
    # perturbed samples
    perturb_image = perturb_image.copy()
    perturb_image = cv2.resize(perturb_image, (960, 960))

    f, axarr = plt.subplots(2, 2)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=f)

    f_1 = f.add_subplot(spec2[0, 0])
    f_1.set_title('original')
    f_1.imshow(orig_image)
    f_2 = f.add_subplot(spec2[0, 1])
    f_2.set_title('perturbed')
    f_2.imshow(perturb_image)
    f_3 = f.add_subplot(spec2[1, :])
    table = f_3.table(cellText=cellText, loc="best",
                         colLabels=colLabels)
    table.scale(1, 4)
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    f_1.axis('off')
    f_2.axis('off')
    f_3.axis('off')
    for ax in axarr:
        for subaxis in ax:
            subaxis.axis('off')
    # plt.savefig(os.path.join(viz_save_folder, entry_name+".png"), dpi=500)
    plt.show()


if __name__ == '__main__':
    image_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'adv5', 'perturb_samples', 'fgsm', 'images')
    orig_image_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'adv5', 'perturb_samples', 'fgsm', 'orig_images')
    text_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'adv5', 'perturb_samples', 'fgsm', 'json')
    viz_save_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'adv5', 'perturb_samples', 'fgsm', 'viz')
    if not os.path.exists(viz_save_folder):
        os.makedirs(viz_save_folder)
    samples = sorted(os.listdir(image_folder))
    for sample in samples:
        perturb_image = np.load(open(os.path.join(image_folder, sample), 'rb'))
        orig_image = np.load(open(os.path.join(orig_image_folder, sample), 'rb'))
        entry_name = sample[:sample.find('.npy')]
        json_file_name = entry_name + '.json'
        caption = json.load(open(os.path.join(text_folder, json_file_name)))
        visualize_samples(orig_image=orig_image, perturb_image=perturb_image, caption=caption, entry_name=entry_name, viz_save_folder=viz_save_folder)
        break


