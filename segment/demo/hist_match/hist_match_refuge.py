import numpy as np
import cv2


def hist_match(source, template):
    # Compute the histograms and their cumulative distributions
    src_hist, bin_edges = np.histogram(source.ravel(), bins=256, density=True)
    src_cdf = src_hist.cumsum()
    src_cdf /= src_cdf[-1]

    tgt_hist, _ = np.histogram(template.ravel(), bins=256, density=True)
    tgt_cdf = tgt_hist.cumsum()
    tgt_cdf /= tgt_cdf[-1]

    # Use the inverse of the CDF to map values from source to target
    inverse_cdf = np.interp(src_cdf, tgt_cdf, bin_edges[:-1])

    return inverse_cdf[source].reshape(source.shape)


source_img_path = './n0006.jpg'
target_img_path = './T0053.jpg'

source = cv2.imread(source_img_path)
template = cv2.imread(target_img_path)

matched_channels = []
for d in range(source.shape[2]):
    matched_c = hist_match(source[:, :, d], template[:, :, d])
    matched_channels.append(matched_c)

matched_image = cv2.merge(matched_channels).astype(np.uint8)
cv2.imwrite('matched_image.jpg', matched_image)
