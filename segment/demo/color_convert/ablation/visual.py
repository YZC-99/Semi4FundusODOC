from PIL import Image
import numpy as np

def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([217, 217, 217])
    cmap[1] = np.array([125,186, 248])
    cmap[2] = np.array([252, 226, 142])
    cmap[3] = np.array([0, 0, 255])
    cmap[4] = np.array([0, 255, 0])
    return cmap

cmap = color_map()

def image_to_array(img):
    return np.asarray(img)

def resize_to_match(base, target):
    """Resize the target image to match the size of the base image."""
    base_width, base_height = base.size
    target = target.resize((base_width, base_height))
    return target

def compute_difference(img1, img2):
    return img1 - img2

def overlay_difference(original, diff):
    # Overlay the difference and clip values to be between 0 and 255
    return np.clip(original + diff, 0, 255)

def process_image_difference(gt_path, target_path, cmap):
    # Load the images
    gt_img = Image.open(gt_path).convert('L')
    target_img = Image.open(target_path).convert('L')

    # Ensure the target image is the same size as the gt image
    target_img = resize_to_match(gt_img, target_img)

    # Convert the images to arrays
    gt_array = image_to_array(gt_img)
    target_array = image_to_array(target_img)

    # Compute the difference
    difference = compute_difference(target_array, gt_array)

    # Overlay the difference onto the original
    overlayed = overlay_difference(target_array, difference)

    # Convert overlayed image to indexed mode using the provided colormap
    indexed_img = Image.fromarray(overlayed.astype('uint8')).convert('P')
    indexed_img.putpalette(cmap.flatten().tolist())

    # Save the resulting image
    result_path = target_path.replace('.png', '_diff.png')
    indexed_img.save(result_path)

    print(f"Processed image saved as {result_path}")

# Process the images
process_image_difference('gt.png', 'proposed.png', cmap)
process_image_difference('gt.png', 'remove_D3CM.png', cmap)
process_image_difference('gt.png', 'remove_HFFRM.png', cmap)
process_image_difference('gt.png', 'remove_LGAM.png', cmap)
