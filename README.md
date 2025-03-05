# U-net Segmentation

TODO

# Project Overview
This project focuses on building a U-Net model for image segmentation using the Oxford-IIIT Pet Dataset. 

# Architecture

![architecture](docs/u-net-architecture.png)

It follows an encoder-decoder structure with skip connections to preserve spatial information during upsampling.

### 1. Convolutional Block (ConvBlock)

Each ConvBlock consists of:

- Two 3x3 convolution layers with padding to maintain spatial dimensions.
- Batch Normalization to stabilize training and speed up convergence.
- ReLU activation for non-linearity.

This block is a fundamental component used in both the encoder and decoder to learn feature representations.

### 2. Encoder (Contracting Path)
The encoder progressively reduces the spatial dimensions while increasing feature depth. Each encoder block consists of:

- A MaxPooling layer (2x2) for downsampling.
- A ConvBlock to extract features.
- The depth of feature maps increases at each stage (64 → 128 → 256 → 512 → 1024).

The encoder helps capture high-level semantic features while reducing resolution.

### 3. Decoder (Expanding Path)
The decoder restores the spatial resolution of the image while refining segmentation details. Each decoder block consists of:

- A Transposed Convolution layer (ConvTranspose2d) to upsample the feature maps.
- Concatenation with the corresponding encoder feature map (skip connection) to retain fine details.
- A ConvBlock to refine the features.

The decoder progressively reduces feature depth while increasing spatial resolution (1024 → 512 → 256 → 128 → 64).

# Dataset
- The dataset used is the Oxford-IIIT Pet Dataset: Link
- It contains 37 pet breeds, with around 200 images per breed.
- Each image has an annotation mask for segmentation, including three classes: pet, background, and boundary.
- The dataset will be downloaded automatically by torchvision library

# How to run

+ training

    ```bash
    python main.py
    ```

+ tensorboard

    ```bash
    tensorboard --logdir=output/unet_logs
    ```

# Results

## Training Performance
The training process yielded the following results:
- **Training Loss**: The loss metric observed during the training phase.
- **Test Loss**: The loss metric evaluated on the test dataset.

For a detailed visualization of the training and test loss trends, refer to the graph below:

![Training Result](/docs/training_result.png)

---

## Visualizations
The following figures provide insights into the experimental results. Figures are displayed in rows for better comparison:

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 1</strong>: [Description of Figure 1]</p>
    <img src="/docs/result/Figure_1.png" alt="Figure 1" style="width: 100%;">
  </div>
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 2</strong>: [Description of Figure 2]</p>
    <img src="/docs/result/Figure_2.png" alt="Figure 2" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 3</strong>: [Description of Figure 3]</p>
    <img src="/docs/result/Figure_3.png" alt="Figure 3" style="width: 100%;">
  </div>
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 4</strong>: [Description of Figure 4]</p>
    <img src="/docs/result/Figure_4.png" alt="Figure 4" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 5</strong>: [Description of Figure 5]</p>
    <img src="/docs/result/Figure_5.png" alt="Figure 5" style="width: 100%;">
  </div>
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 6</strong>: [Description of Figure 6]</p>
    <img src="/docs/result/Figure_6.png" alt="Figure 6" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 7</strong>: [Description of Figure 7]</p>
    <img src="/docs/result/Figure_7.png" alt="Figure 7" style="width: 100%;">
  </div>
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 8</strong>: [Description of Figure 8]</p>
    <img src="/docs/result/Figure_8.png" alt="Figure 8" style="width: 100%;">
  </div>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 9</strong>: [Description of Figure 9]</p>
    <img src="/docs/result/Figure_9.png" alt="Figure 9" style="width: 100%;">
  </div>
  <div style="flex: 1 1 45%;">
    <p><strong>Figure 10</strong>: [Description of Figure 10]</p>
    <img src="/docs/result/Figure_10.png" alt="Figure 10" style="width: 100%;">
  </div>
</div>




# SOMEDO

# WHYDO

# MAYBEDO

# NEXTDO