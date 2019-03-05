import numpy as np
from skimage.draw import line_aa
import matplotlib.pyplot as plt

def draw_box_on_image(pred, label, images):
    ''' Function to draw bounding boxes on the images. Predicted bounding boxes will be
    presented with a dotted line and actual boxes are presented with a solid line.

    Parameters
    ----------
    
    pred: [[x, y, w, h]]
        The predicted bounding boxes in percentages

    label: [[x, y, w, h]]
        The actual bounding boxes in percentages

    images: [[np.array]]
        The correponding images.

    Returns
    -------

    images: [[np.array]]
        Images with bounding boxes printed on them.
    '''

    image_h, image_w = images.shape[-2:]
    pred[:, 0], pred[:, 1] = pred[:, 0] * image_w, pred[:, 1] * image_h
    pred[:, 2], pred[:, 3] = pred[:, 2] * image_w, pred[:, 3] * image_h

    label[:, 0], label[:, 1] = label[:, 0] * image_w, label[:, 1] * image_h
    label[:, 2], label[:, 3] = label[:, 2] * image_w, label[:, 3] * image_h

    for i in range(images.shape[0]):
        image = images[i, 0]
        image = draw_box(pred[i, :], image, line_type="dotted")
        image = draw_box(label[i, :], image, line_type="solid")
        images[i, 0, :, :] = image
    return images
