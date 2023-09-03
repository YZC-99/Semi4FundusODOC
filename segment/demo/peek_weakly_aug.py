from PIL import Image
from segment.dataloader.transform import *

image_path = 'augs/T0001.jpg'

img = Image.open(image_path)
img_rotated = img.rotate(90, resample=Image.BILINEAR, expand=True)
img_rotated = img.rotate(90, resample=Image.BILINEAR, expand=True)

img_width, img_height = img.size
max_translate_percent = 0.15
translate_x = random.uniform(-max_translate_percent, max_translate_percent) * img_width
translate_y = random.uniform(-max_translate_percent, max_translate_percent) * img_height
img_translated = img.transform(
    img.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
)

min_scale,max_scale = 0.8,1.2
scale_factor = random.uniform(min_scale, max_scale)
new_width = int(img.width * scale_factor)
new_height = int(img.height * scale_factor)

img_scaled = img.resize((new_width, new_height), Image.BILINEAR)

img_rotated.save('augs/img_rotated.jpg')
img_translated.save('augs/img_translated.jpg')
img_scaled.save('augs/img_scaled.jpg')