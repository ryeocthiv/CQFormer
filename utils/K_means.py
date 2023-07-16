import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def color_quantization(img, k):
    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))
    print(data.shape)
    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
    # Applying cv2.kmeans function
    ret, label, center = cv2.kmeans(data, k, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


if __name__ == '__main__':
    x_RGB = Image.open('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/pictures/n01443537_1970.JPEG')
    x_RGB = np.array(x_RGB)
    color = color_quantization(x_RGB, 5)
    color = Image.fromarray(color)
    color.save("/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/pictures/Kmeans_5.png")
    plt.axis('off')
    plt.imshow(x_RGB)
    plt.show()
    plt.axis('off')
    plt.imshow(color)
    plt.show()