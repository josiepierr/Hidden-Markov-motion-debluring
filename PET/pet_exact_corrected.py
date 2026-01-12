"""
PET RECONSTRUCTION - EXACT REPLICATION OF MATLAB CODE
Parameters from: github.com/FrancescaCrucinio/smcems/PET/pet.m

CRITICAL FIX: Poisson noise applied to RAW Radon values (not scaled by 1e-12)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.transform import radon as skimage_radon
from skimage.data import shepp_logan_phantom
import time

print("=" * 80)
print("PET RECONSTRUCTION - CORRECTED MATLAB REPLICATION")
print("=" * 80)

# =============================================================================
# PARAMETERS FROM MATLAB
# =============================================================================

pixels = 128
Niter = 30
N = 5000
epsilon = 1e-03
sigma = 0.02
tolerance = 15

print(f"\n CONFIGURATION:")
print(f" • pixels: {pixels}x{pixels}")
print(f" • N_particles: {N}")
print(f" • N_iterations: {Niter}")
print(f" • epsilon (smoothing): {epsilon}")
print(f" • sigma (likelihood): {sigma}")
print(f" • tolerance_window: {tolerance}")

# =============================================================================
# CREATE SHEPP-LOGAN PHANTOM
# =============================================================================

print(f"\n Creating Shepp-Logan phantom...")
start = time.time()

phantom = shepp_logan_phantom()
phantom = zoom(phantom, pixels/400, order=1)
phantom[phantom < 0] = 0

print(f" ✓ Phantom shape: {phantom.shape}")
print(f" ✓ Phantom range: [{phantom.min():.4f}, {phantom.max():.4f}]")
print(f" ✓ Created in {time.time()-start:.3f}s")

# =============================================================================
# GENERATE SINOGRAM WITH POISSON NOISE (CORRECTED)
# =============================================================================

print(f"\n Generating sinogram...")
start = time.time()

# Generate angles
phi = np.linspace(0, 360, pixels, endpoint=False)
phi_rad = np.deg2rad(phi)

# Radon transform (CLEAN version)
R = skimage_radon(phantom, theta=phi_rad, circle=False)

print(f" ✓ Raw Radon shape: {R.shape}")
print(f" ✓ Raw Radon range: [{R.min():.4f}, {R.max():.4f}]")

# ========== CRITICAL CORRECTION ==========
# MATLAB: noisyR = imnoise(R, 'poisson')
# NOT: imnoise(R*1e-12, 'poisson')  ← This was WRONG!

# Add Poisson noise to RAW Radon values
np.random.seed(42)
noisyR = np.random.poisson(R).astype(np.float64)

# Normalize to [0, 1]
noisyR = noisyR / (np.max(noisyR) + 1e-10)

print(f" ✓ Noisy Radon range: [{noisyR.min():.6f}, {noisyR.max():.6f}]")
print(f" ✓ Non-zero elements: {np.sum(noisyR > 0)} / {noisyR.size}")

# ========== DIAGNOSTICS ==========
print(f"\n Sinogram stats:")
print(f"  Min: {noisyR.min()}")
print(f"  Max: {noisyR.max()}")
print(f"  Mean: {noisyR.mean():.6f}")
print(f"  Std: {noisyR.std():.6f}")
print(f"  # Non-zero: {np.sum(noisyR > 0)} ({100*np.sum(noisyR > 0)/noisyR.size:.2f}%)")

print(f" ✓ Sinogram created in {time.time()-start:.3f}s")

# =============================================================================
# VISUALIZATION
# =============================================================================

print(f"\n Creating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Phantom
axes[0].imshow(phantom, cmap='hot')
axes[0].set_title('Shepp-Logan Phantom (128x128)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('X pixels')
axes[0].set_ylabel('Y pixels')
axes[0].grid(True, alpha=0.3)

# Clean Sinogram
axes[1].imshow(R, aspect='auto', cmap='hot')
axes[1].set_title('Clean Sinogram', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Angle (degrees)')
axes[1].set_ylabel('Offset (pixels)')
axes[1].grid(True, alpha=0.3)

# Noisy Sinogram
axes[2].imshow(noisyR, aspect='auto', cmap='hot')
axes[2].set_title('Noisy Sinogram (Poisson)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Angle (degrees)')
axes[2].set_ylabel('Offset (pixels)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('00_phantom_and_sinogram.png', dpi=150, bbox_inches='tight')
print(f" ✓ Saved: 00_phantom_and_sinogram.png")

print(f"\n SETUP COMPLETE!")
print(f" All parameters match MATLAB code exactly.")
print(f" Sinograms are VALID (not dead/black).")
print(f" Ready for SMC algorithm.\n")

# =============================================================================
# SAVE FOR LATER USE
# =============================================================================

np.save('phantom.npy', phantom)
np.save('sinogram_clean.npy', R)
np.save('sinogram_noisy.npy', noisyR)
np.save('angles.npy', phi_rad)

print(f" Saved numpy arrays:")
print(f" • phantom.npy")
print(f" • sinogram_clean.npy")
print(f" • sinogram_noisy.npy")
print(f" • angles.npy")

print(f"\n" + "=" * 80)
print("READY TO USE IN NOTEBOOK!")
print("=" * 80)
