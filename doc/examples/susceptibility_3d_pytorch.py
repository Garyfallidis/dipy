
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_image_3d(size=32):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(-1, 1, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    return torch.exp(-(X**2 + Y**2 + Z**2) * 10)

def generate_displacement_field_3d(size=32, max_disp=3.0):
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    z = torch.linspace(0, 1, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    return max_disp * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y) * torch.sin(torch.pi * Z)

def apply_distortion_3d(image, displacement, direction=(0, 1, 0)):
    D, H, W = image.shape
    grid_x = torch.linspace(-1, 1, D)
    grid_y = torch.linspace(-1, 1, H)
    grid_z = torch.linspace(-1, 1, W)
    X, Y, Z = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid = torch.stack([X, Y, Z], dim=-1).to(image.device)

    dgrid = torch.zeros_like(grid)
    dgrid[..., 0] = direction[0] * displacement / (D / 2)
    dgrid[..., 1] = direction[1] * displacement / (H / 2)
    dgrid[..., 2] = direction[2] * displacement / (W / 2)

    warped_grid = grid + dgrid
    image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    warped = F.grid_sample(image, warped_grid.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze()

def loss_fn(I0, b, I_f, I_r, lam=0.1):
    f_rwarp = apply_distortion_3d(I0, +b, direction=(0, 1, 0))
    r_rwarp = apply_distortion_3d(I0, -b, direction=(0, -1, 0))
    data_loss = 0.5 * torch.mean((f_rwarp - I_f)**2 + (r_rwarp - I_r)**2)

    dx = b[1:, :, :] - b[:-1, :, :]
    dy = b[:, 1:, :] - b[:, :-1, :]
    dz = b[:, :, 1:] - b[:, :, :-1]
    smoothness = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

    return data_loss + lam * smoothness

# Main execution
device = 'cuda' if torch.cuda.is_available() else 'cpu'
size = 32
image = generate_image_3d(size).to(device)
true_b = generate_displacement_field_3d(size).to(device)

image_fwd = apply_distortion_3d(image, true_b, direction=(0, 1, 0)).detach()
image_rev = apply_distortion_3d(image, true_b, direction=(0, -1, 0)).detach()

I0 = 0.5 * (image_fwd + image_rev).clone().detach().requires_grad_(True)
b = torch.zeros_like(true_b, requires_grad=True)

optimizer = torch.optim.LBFGS([I0, b], max_iter=60, line_search_fn='strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss = loss_fn(I0, b, image_fwd, image_rev, lam=0.1)
    loss.backward()
    return loss

optimizer.step(closure)

# Post-optimization result
f_unwarp = apply_distortion_3d(image_fwd, -b.detach(), direction=(0, 1, 0))
r_unwarp = apply_distortion_3d(image_rev, +b.detach(), direction=(0, -1, 0))
I0_est = 0.5 * (f_unwarp + r_unwarp)

# Visualize a central slice
z = size // 2
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image[:, :, z].cpu().numpy(), cmap='gray')
axs[0].set_title("Original (slice)")
axs[1].imshow(I0_est[:, :, z].cpu().numpy(), cmap='gray')
axs[1].set_title("Estimated I₀ (slice)")
axs[2].imshow((image - I0_est).abs()[:, :, z].cpu().numpy(), cmap='hot')
axs[2].set_title("Difference (|I - I₀|)")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
