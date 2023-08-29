import torch
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Simulated segmentation mask
segmentation_mask = np.zeros((256, 256), dtype=np.uint8)
cv2.circle(segmentation_mask, (128, 128), 60, 1, -1)  # Simulate a circular object

# Convert to PyTorch tensor
segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.float32)


# Define the shape constraint loss as a PyTorch function
def shape_constraint_loss(params):
    center_x, center_y, radius = params
    predicted_mask = torch.zeros_like(segmentation_mask)
    predicted_mask = cv2.circle(predicted_mask.numpy(), (int(center_x), int(center_y)), int(radius), 1, -1)
    predicted_mask = torch.tensor(predicted_mask, dtype=torch.float32)

    loss = torch.mean((predicted_mask - segmentation_mask) ** 2)
    return loss


# Initial guess for center (x, y) and radius (r)
initial_params = torch.tensor([128.0, 128.0, 30.0], requires_grad=True)

# Define an optimizer to minimize the loss
optimizer = optim.LBFGS([initial_params], lr=0.1, max_iter=100)


# Optimization loop
def closure():
    optimizer.zero_grad()
    loss = shape_constraint_loss(initial_params)
    loss.backward()
    return loss


# Perform optimization
optimizer.step(closure)

# Extract optimized parameters
optimized_params = initial_params.detach().numpy()
optimized_center_x, optimized_center_y, optimized_radius = optimized_params

# Create the final segmentation mask based on the optimized parameters
final_segmentation = np.zeros_like(segmentation_mask)
cv2.circle(final_segmentation, (int(optimized_center_x), int(optimized_center_y)), int(optimized_radius), 1, -1)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(segmentation_mask, cmap='gray')
plt.title('Ground Truth Mask')

plt.subplot(122)
plt.imshow(final_segmentation, cmap='gray')
plt.title('Final Segmentation with Shape Constraint')

plt.tight_layout()
plt.show()
