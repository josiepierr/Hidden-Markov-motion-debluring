# Image Deblurring with Python

Python implementation of the paper: [A Particle Method for Solving
Fredholm Equations of the First Kind](https://arxiv.org/abs/2009.09974).

---

## Setup and Installation

### Prerequisites
- Python 3.x

### Steps to Run the Code

1. **Create a Python Virtual Environment:**
   ```bash
   python(3) -m venv venv
   ```

2. **Activate the environment**

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Image Deblurring Code:**
    ```bash
   python run_smc.py --b 128 --sigma 0.02 --n_iter 100 --n_particles 5000 --orig_file BC.png --blur_file BCblurred.png
   ```

### To test with a new image

1. **Place your image in the original_image folder**
2. **Blur it using:**
   ```bash
   python blur.py --image_name Peugeot-208.png --b 32 --sigma 0.01 --alpha 0.1 --beta 0.1
   ```