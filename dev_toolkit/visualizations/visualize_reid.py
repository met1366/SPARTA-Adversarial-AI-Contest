import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
import matplotlib.gridspec as gridspec

from environment_setup import PROJECT_ROOT_DIR


def extract_caption(caption):
    """
    A tabular structure to display the mis-predictions
    :param caption: json object having perturbed and original predictions
    :return: Tabular structure with the mis-predictions
    """
    orig_prediction = [caption['orig']]
    perturbed_prediction = [caption['perturb']]
    colLabels = ['Class']
    # include the category labels
    colLabels.insert(0, ' ')
    orig_prediction.insert(0, 'original')
    perturbed_prediction.insert(0, 'perturbed')
    cellText = [orig_prediction, perturbed_prediction]
    return colLabels, cellText


def visualize_samples(perturb_image, gt_image, caption, entry_name, viz_save_folder):
    """
    Function to save the tabular structure along with the image
    :param perturb_image: The perturbed image
    :param gt_image: Original image
    :param caption: A json object containing original and perturbed predictions
    :param entry_name: An identifier for the input data sample
    :param viz_save_folder: Target folder to save the predictions to
    :return: None
    """
    colLabels, cellText = extract_caption(caption)
    perturb_image = perturb_image.copy()
    perturb_image = cv2.resize(perturb_image, (960, 960))

    # Same for gt
    gt_image = gt_image.copy()
    gt_image = cv2.resize(gt_image, (960, 960))

    f, axarr = plt.subplots(2, 1)
    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=f)

    f_1 = f.add_subplot(spec2[0, 0])
    f_1.set_title('original')
    f_1.imshow(gt_image)

    f_2 = f.add_subplot(spec2[0, 1])
    f_2.imshow(perturb_image)
    f_2.set_title('perturbed')
    # axarr[0, 0].imshow(image)
    # axarr.plot(text)
    f_3 = f.add_subplot(spec2[0, 2])
    table = f_3.table(cellText=cellText, loc="center",
                      colLabels=colLabels)
    table.scale(1.5, 1)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    f_1.axis('off')
    f_2.axis('off')
    f_3.axis('off')
    for ax in axarr:
        ax.axis('off')
    plt.savefig(os.path.join(viz_save_folder, entry_name + ".png"), dpi=500)
    plt.show()


if __name__ == '__main__':
    image_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'aa7', 'perturb_samples', 'fgsm', 'images')
    gt_image_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'aa7', 'perturb_samples', 'fgsm', 'orig_images')
    text_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'aa7', 'perturb_samples', 'fgsm', 'json')
    viz_save_folder = os.path.join(PROJECT_ROOT_DIR, 'execution_results', 'aa7', 'perturb_samples', 'fgsm', 'viz')
    if not os.path.exists(viz_save_folder):
        os.makedirs(viz_save_folder)
    samples = sorted(os.listdir(image_folder))
    for sample in samples:
        image = np.load(open(os.path.join(image_folder, sample), 'rb'))
        gt_image = np.load(open(os.path.join(gt_image_folder, sample), 'rb'))
        entry_name = sample[:sample.find('.npy')]
        json_file_name = entry_name + '.json'
        caption = json.load(open(os.path.join(text_folder, json_file_name)))
        visualize_samples(perturb_image=image, gt_image=gt_image, caption=caption, entry_name=entry_name, viz_save_folder=viz_save_folder)


