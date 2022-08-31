from PIL import Image
from io import BytesIO
from utils.dither.palette import Palette
from utils.dither.dithering import error_diffusion_dithering
from torchvision.transforms import *


class MedianCut(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img):
        if self.num_colors is not None:
            if not self.dither:
                sampled_img = img.quantize(colors=self.num_colors, method=0).convert('RGB')
            else:
                palette = Palette(img.quantize(colors=self.num_colors, method=0))
                sampled_img = error_diffusion_dithering(img, palette).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class OCTree(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, method=2).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class KMeans(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, kmeans=2).convert('RGB')
        else:
            sampled_img = img
        return sampled_img


class PNGCompression(object):
    def __init__(self, buffer_size_counter):
        self.buffer_size_counter = buffer_size_counter

    def __call__(self, img):
        png_buffer = BytesIO()
        img.save(png_buffer, "PNG")
        self.buffer_size_counter.update(png_buffer.getbuffer().nbytes)
        return img


class JpegCompression(object):
    def __init__(self, buffer_size_counter, quality=5):
        self.buffer_size_counter = buffer_size_counter
        self.quality = quality

    def __call__(self, img):
        jpeg_buffer = BytesIO()
        img.save(jpeg_buffer, "JPEG", quality=self.quality)
        self.buffer_size_counter.update(jpeg_buffer.getbuffer().nbytes)
        jpeg_buffer.seek(0)
        sampled_img = Image.open(jpeg_buffer)
        return sampled_img


if __name__ == '__main__':
    from color_distillation.utils.buffer_size_counter import BufferSizeCounter
    import matplotlib.pyplot as plt

    img_path = '/home/ssh685/CV_project_AAAI2023/color_distillation-master/imgs/beach-1536x864.jpg'
    img = Image.open(img_path)
    sample_type = 'mcut'
    num_colors = 16
    buffer_size_counter = BufferSizeCounter()
    sampled_img = img
    if sample_type == 'mcut':
        quantized_data = img.quantize(colors=num_colors, method=0)
        sampled_img = quantized_data.convert('RGB')
        palette = Palette(quantized_data)
        # print(quantized_data.mode)
        # print(palette.colours)
        # print(img.convert("RGB").palette.palette)
    elif sample_type == 'octree':
        sampled_img = img.quantize(colors=num_colors, method=2).convert('RGB')
    elif sample_type == 'kmeans':
        sampled_img = img.quantize(colors=num_colors, kmeans=2).convert('RGB')
    elif sample_type == 'jpeg':
        quality = 5
        jpeg_buffer = BytesIO()
        img.save(jpeg_buffer, "JPEG", quality=quality)
        buffer_size_counter.update(jpeg_buffer.getbuffer().nbytes)
        jpeg_buffer.seek(0)
        sampled_img = Image.open(jpeg_buffer)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    # plt.axis('off')
    # plt.imshow(img.convert("P"))
    # plt.show()
    plt.axis('off')
    plt.imshow(sampled_img)
    plt.show()
