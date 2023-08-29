import torch
import torch.nn.functional as F

# Input image (1 channel, 5x5 size)
input_image = torch.arange(1, 26).reshape(1, 1, 5, 5).float()
input_d = input_image.permute(0, 2, 3, 1)
image_d = torch.nn.functional.pad(input_d, (0, 0, 1, 1, 1, 1, 0, 0), mode='constant')  # N(H+2)(W+2)C
image_d[:, 0, :, :] = image_d[:, 1, :, :]  # N(H+2)(W+2)C
image_d[:, -1, :, :] = image_d[:, -2, :, :]  # N(H+2)(W+2)C
image_d[:, :, 0, :] = image_d[:, :, 1, :]  # N(H+2)(W+2)C
image_d[:, :, -1, :] = image_d[:, :, -2, :]  # N(H+2)(W+2)C
image_d = image_d.permute(0,3,1,2)
# Apply unfold to input image with a kernel size of 3x3
unfolded = F.unfold(image_d, kernel_size=5)

# Reshape the unfolded tensor to have 9 channels
output_reshaped = unfolded.reshape(1, 9, 5, 5)

print("Input Image:")
print(input_image.size())
print(image_d.size())
print("\nUnfolded and Reshaped Output (9 channels):")
print(output_reshaped.size())





