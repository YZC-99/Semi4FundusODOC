import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms
from segment.dataloader.boundary_utils import class2one_hot,one_hot2dist


def dist_transform(mask):
    # mask = np.array(mask)
    # mask_arr_ex = np.expand_dims(mask, axis=0)
    mask_tensor = torch.unsqueeze(mask,dim=0)
    mask_tensor = mask_tensor.to(torch.int64)
    mask_tensor[mask_tensor == 255] = 0
    # mask_tensor = torch.tensor(mask_arr_ex, dtype=torch.int64)
    mask_trans = class2one_hot(mask_tensor, 3)
    mask_trans_arr = mask_trans.cpu().squeeze().numpy()
    bounadry = one_hot2dist(mask_trans_arr, resolution=[1, 1])
    return bounadry

def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def add_salt_pepper_noise(img,mask, p=0.5,noise_level=0.02):
    if random.random() < p:
        img_array = np.array(img)

        h, w, _ = img_array.shape
        num_pixels = int(h * w * noise_level)

        # Add salt noise
        salt_coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
        img_array[salt_coords[0], salt_coords[1], :] = 255

        # Add pepper noise
        pepper_coords = [np.random.randint(0, i - 1, num_pixels) for i in (h, w)]
        img_array[pepper_coords[0], pepper_coords[1], :] = 0

        img = Image.fromarray(img_array)
    return img,mask

def random_scale(img, mask, min_scale=0.8, p=0.5, max_scale=1.2):
    if random.random() < p:
        w_scale_factor = random.uniform(min_scale, max_scale)
        h_scale_factor = random.uniform(min_scale, max_scale)
        new_width = int(img.width * w_scale_factor)
        new_height = int(img.height * h_scale_factor)

        img = img.resize((new_width, new_height), Image.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.NEAREST)

    return img, mask

def random_rotate(img, mask, p=0.5, max_rotation_angle=90):
    if random.random() < p:
        rotation_angle = random.uniform(-max_rotation_angle, max_rotation_angle)
        img = img.rotate(rotation_angle, resample=Image.BILINEAR, expand=True)
        mask = mask.rotate(rotation_angle, resample=Image.NEAREST, expand=True)

    return img, mask

def random_translate(img, mask, p=0.5, max_translate_percent=0.15):
    if random.random() < p:
        img_width, img_height = img.size
        translate_x = random.uniform(-max_translate_percent, max_translate_percent) * img_width
        translate_y = random.uniform(-max_translate_percent, max_translate_percent) * img_height

        img = img.transform(
            img.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )

        mask = mask.transform(
            mask.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )

    return img, mask

def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Normalize((0.0,), (1.0,))
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def resize(img, mask,size):
    img = img.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.NEAREST)
    return img, mask

# class torch_resize(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, image, target):
#         size = self.size
#         # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
#         image = F.resize(image, [size,size])
#         # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
#         # 如果是之前的版本需要使用PIL.Image.NEAREST
#         target = F.resize(target, [size,size], interpolation=T.InterpolationMode.NEAREST)
#         return image, target

def randomresize(img, mask, base_size, ratio_range):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask
