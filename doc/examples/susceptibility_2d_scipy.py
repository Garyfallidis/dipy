
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

def generate_image_2d(size=64):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.exp(-(X**2 + Y**2) * 10)

def apply_distortion_2d(image, displacement, direction, order=3):
    h, w = image.shape
    X, Y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.array([X + direction[0]*displacement, Y + direction[1]*displacement])
    warped = map_coordinates(image, coords, order=order, mode='reflect')
    return warped

def smoothness_penalty(b, shape):
    b = b.reshape(shape)
    dx = np.diff(b, axis=0, prepend=b[0:1, :])
    dy = np.diff(b, axis=1, prepend=b[:, 0:1])
    return np.sum(dx**2 + dy**2)

def loss_with_latent_I0(b_vec, image_fwd, image_rev, shape, direction=(0, 1), lam=0.1):
    b = b_vec.reshape(shape)
    f_unwarp = apply_distortion_2d(image_fwd, -b, direction)
    r_unwarp = apply_distortion_2d(image_rev, +b, (-direction[0], -direction[1]))
    I0 = 0.5 * (f_unwarp + r_unwarp)

    f_rwarp = apply_distortion_2d(I0, +b, direction)
    r_rwarp = apply_distortion_2d(I0, -b, (-direction[0], -direction[1]))

    data_loss = 0.5 * np.mean((f_rwarp - image_fwd)**2 + (r_rwarp - image_rev)**2)
    smoothness = smoothness_penalty(b, shape)
    return data_loss + lam * smoothness

def main():
    size = 64
    image = generate_image_2d(size)
    true_b = 5.0 * np.sin(np.pi * np.linspace(0, 1, size))[:, None] * np.sin(np.pi * np.linspace(0, 1, size))[None, :]

    image_fwd = apply_distortion_2d(image, true_b, direction=(0, 1))
    image_rev = apply_distortion_2d(image, true_b, direction=(0, -1))

    b0 = np.zeros_like(true_b).flatten()
    result = minimize(
        loss_with_latent_I0, b0,
        args=(image_fwd, image_rev, true_b.shape, (0, 1), 0.1),
        method='L-BFGS-B',
        options={'maxiter': 100}
    )
    b_opt = result.x.reshape(true_b.shape)

    f_unwarp = apply_distortion_2d(image_fwd, -b_opt, direction=(0, 1))
    r_unwarp = apply_distortion_2d(image_rev, +b_opt, direction=(0, -1))
    I0_est = 0.5 * (f_unwarp + r_unwarp)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(image, cmap='gray'); axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(image_fwd, cmap='gray'); axs[0, 1].set_title("Forward Distorted")
    axs[0, 2].imshow(image_rev, cmap='gray'); axs[0, 2].set_title("Reverse Distorted")
    axs[1, 0].imshow(b_opt, cmap='coolwarm'); axs[1, 0].set_title("Optimized Displacement")
    axs[1, 1].imshow(I0_est, cmap='gray'); axs[1, 1].set_title("Estimated I₀")
    axs[1, 2].imshow(np.abs(image - I0_est), cmap='hot'); axs[1, 2].set_title("|Original - I₀|")
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
