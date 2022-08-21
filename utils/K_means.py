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
    img_path = '/home/ssh685/CV_project_AAAI2023/color_distillation-master/imgs/beach-1536x864.jpg'
    img = cv2.imread(img_path)
    # cv2.imshow('img', img)
    color_3 = color_quantization(img, 256)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    color_3 = Image.fromarray(cv2.cvtColor(color_3, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.axis('off')
    plt.imshow(color_3)
    plt.show()