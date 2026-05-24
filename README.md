# Custom Data Generator for OPG Dental X-ray Images


## Project Overview

This project implements pipeline for processing, visualizing, and generating training data from Orthopantomogram (OPG) dental X-ray DICOM images for endodontic surgery classification. The goal is to build a custom data generator that can classify dental X-ray images as either **Normal** or **Abnormal** in the context of endodontic surgical procedures.


### Glossary

- **Dental Imaging**: OPG is the standard panoramic radiograph used in dentistry

- **Medical Imaging**: Healthcare providers rely on the DICOM (Digital Imaging and Communications in Medicine) standard to reliably store, manage, and transmit medical images—such as X-rays, MRIs, and CT scans—between imaging equipment, workstations, and electronic health records.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Data Preprocessing](#data-preprocessing)
3. [Data Visualization](#data-visualization)
4. [Input Pipeline Options](#input-pipeline-options)
5. [Data Generation Methods](#data-generation)

---

## Setup and Installation

### Required Libraries

The following libraries are installed and used:

- **pydicom**: For reading and processing DICOM medical images
- **split-folders**: For organizing and splitting data into train/validation/test sets
- **gdown**: For downloading files from Google Drive

### Installation Commands

```bash
!pip install pydicom
!pip install split-folders[full]
!pip install -U gdown
!pip install visualkeras
```

### Hardware Acceleration

The notebook supports three types of execution environments:

- **CPU**: Default strategy using TensorFlow on CPU
- **GPU**: Enable GPU acceleration with `tf.config.list_physical_devices('GPU')`
- **TPU**: Support for TPU distributed training

---

## Data Preprocessing

### Data Source

The project uses three parts of DICOM dental X-ray data downloaded from Google Drive:

1. **Part 1 (iaaa-data_v2.zip)**: Initial dataset with DICOM files and labels
2. **Part 2 (iaaa-data-v3.zip)**: Extended dataset
3. **Part 3 (data-v3.zip)**: Additional labeling variations

Each dataset includes:
- DICOM folder containing `.dcm` image files
- `labels.csv` containing SOPInstanceUID and corresponding Label ('normal' or 'abnormal')

### Data Organization

The preprocessing workflow:

1. **Download** all three data parts using gdown
2. **Extract** zip files to designated directories
3. **Organize** DICOM files by creating directory structure:
   ```
   DICOM_Data/
   ├── normal/       (normal tooth samples)
   └── abnormal/     (abnormal tooth samples)
   ```
4. **Transfer** DICOM files to appropriate folders based on labels in CSV files
5. **Handle duplicates** in Part 3 by checking if files already exist

### Dataset Statistics

After preprocessing all parts:
- **Normal samples**: 1,264 images
- **Abnormal samples**: 116 images
- **Total**: 1,380 images (class-imbalanced dataset)

---

## Data Visualization

### DICOM Image Properties

- **Format**: 16-bit grayscale images
- **Data Type**: uint16 (integer values from 0 to 2^16)

### Image Processing Pipeline

The visualization section demonstrates key preprocessing operations:

#### 1. **Normalization**
Converts 16-bit grayscale values to float16 range [0, 1]:
```python
normalized_array = array / 2**16
normalized_array = normalized_array.astype("float16")
```

#### 2. **Center Cropping**
Crops image to region of interest (1800x900 pixels):
```python
width = 1800
height = 900
startx = x//2 - (width//2)
starty = y//2 - (height//2)
cropped_image = image[starty:starty+height, startx:startx+width]
```

#### 3. **Resizing**
Uses bilinear interpolation to resize images:
- Original cropped: 900x1800
- Resized factor: 0.5x (resulting in 450x900)
- Preserves aspect ratio



### Visualization Methods

- **Bone Colormap**: Classic medical imaging visualization
- **Rainbow Colormap**: Enhanced color utilization for detailed inspection
- **Histograms**: Distribution of pixel values
- **Side-by-side Comparison**: Normal vs. Abnormal samples

---

## Input Pipeline Options

The notebook provides three different approaches to handle the input data pipeline:

### Option 1: `tf.data.Dataset.from_tensor_slices()`

**Use Case**: When all input data fits in memory

**Advantages**:
- Simple and straightforward
- Good performance for smaller datasets
- All data loaded at once

**Process**:
1. Load all images and labels into NumPy arrays
2. Create `tf.data.Dataset` objects
3. Apply shuffling and batching
4. Add data augmentation to training set
5. Enable prefetching for GPU optimization


### Option 2: `tf.keras.utils.Sequence` with On-Disk Preprocessing

**Use Case**: When dataset is too large to fit in memory

**Advantages**:
- Memory-efficient (loads batch at training time)
- On-the-fly preprocessing
- Real-time image reading from disk

**Process**:
- Read DICOM files directly from disk
- Apply cropping, normalization, and expansion in the generator
- Configurable batch size and preprocessing parameters

### Option 3: `tf.keras.utils.Sequence` with Preprocessed Arrays

**Use Case**: Best balance between efficiency and flexibility

**Advantages**:
- Preprocessed images saved as `.npy` files for faster loading
- Reduced computational overhead during training
- Memory efficient (batch-wise loading)
- Fastest training speed

**Process**:
1. Preprocess all DICOM images into `.npy` files in stages
2. Organize `.npy` files into train/validation/test directory structure
3. Load preprocessed arrays batch-by-batch using custom generator
4. No real-time preprocessing overhead

---

## Data Generation

### Custom DICOM Data Generator Class

A custom data generator class `DICOMDataGenerator` is implemented that inherits from `tf.keras.utils.Sequence` class:

#### Key Features

```python
class DICOMDataGenerator(tf.keras.utils.Sequence):
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `list_paths` | List of file paths to DICOM or NPY files | Required |
| `batch_size` | Number of samples per batch | 8 |
| `crop_height` | Height for center crop | 1000 |
| `crop_width` | Width for center crop | 2000 |
| `n_channels` | Number of color channels | 3 |
| `zoom_factor_height` | Resize factor for height | 0.5 |
| `zoom_factor_width` | Resize factor for width | 0.5 |
| `mode` | 'fit' for training, 'predict' for inference | 'fit' |
| `label_mode` | 'binary' or 'categorical' output | 'binary' |
| `n_classes` | Number of classes | 2 |
| `shuffle` | Whether to shuffle data | True |
| `random_state` | Random seed for reproducibility | 1 |

#### Methods

- `__len__()`: Returns number of batches per epoch
- `__getitem__(index)`: Returns one batch of data
- `on_epoch_end()`: Shuffles data after each epoch
- `__generate_X()`: Generates batch of images
- `__generate_y()`: Generates batch of labels
- `__read_dcm_file()`: Reads DICOM file
- `__center_crop()`: Applies center cropping
- `__normalize()`: Normalizes pixel values
- `__expand_channel()`: Converts grayscale to RGB
- `__resize_array()`: Resizes images

#### Usage Example

```python
# Create data loaders
train_loader = DICOMDataGenerator(
    partition['train'], 
    batch_size=4, 
    height=720, 
    width=1440,
    label_mode='binary',
    n_channels=3,
    n_classes=2,
    shuffle=True,
    mode='fit'
)

# Iterate through batches
for image_batch, labels_batch in train_loader:
    print(f'Image Batch Shape: {image_batch.shape}')
    print(f'Labels Batch Shape: {labels_batch.shape}')
```

### Preprocessing Pipeline

The complete preprocessing pipeline for Option 3:

1. **Read DICOM**: Extract pixel array from DICOM file
2. **Crop**: Center crop 
3. **Normalize**: Scale to [0, 1] float32 range
4. **Expand Channel**: Convert to 3-channel RGB
5. **Resize**: Scale down by 0.8x (or 0.5x depending on memory constraints)

Final Output Shape: `(720, 1440, 3)` or `(450, 900, 3)` depending on zoom factor.

### Saving Preprocessed Data

The notebook demonstrates saving preprocessed images to disk in stages:

```python
# Process abnormal samples
abnormal_images = np.array([process_scan(path) for path in abnormal_dcm_paths[0:96]])

# Save to disk
for i in range(abnormal_images.shape[0]):
    data = abnormal_images[i]
    np.save(f'/content/DICOM_Data/abnormal/abnormal{i}.npy', data)
```

This approach:
- Prevents memory overflow with large datasets
- Processes data in stages
- Saves processed arrays for reuse
- Enables quick loading during training

### Data Directory Structure After Splitting

```
splitted_data/
├── train/
│   ├── normal/      (binary .npy files)
│   └── abnormal/    (binary .npy files)
├── val/
│   ├── normal/
│   └── abnormal/
└── test/
    ├── normal/
    └── abnormal/
```

---

## Key Insights

1. **Class Imbalance**: The dataset is heavily imbalanced (1,264 normal vs. 116 abnormal). Consider using:
   - Class weights in loss function
   - Data augmentation strategies
   - Resampling techniques

2. **Memory Management**: Processing 16-bit DICOM images is memory-intensive. The three-option approach allows flexibility:
   - Load all (Option 1) for small datasets
   - On-disk processing (Option 2) for real-time flexibility
   - Preprocessed arrays (Option 3) for optimal balance

3. **Image Preprocessing**: The pipeline is designed to:
   - Normalize intensity values for neural networks
   - Standardize image sizes
   - Focus on region of interest (center crop)
   - Convert to compatible format (3-channel for pre-trained models)

4. **Reproducibility**: Fixed `random_state` parameters ensure:
   - Consistent data shuffling
   - Reproducible train/validation/test splits
   - Comparable experimental results

---

## Running the Notebook

Simply execute the cells in order:

1. Install dependencies (Setup section)
2. Select hardware backend (CPU/GPU/TPU)
3. Download data (Preprocessing section)
4. Organize files and visualize samples (Preprocessing & Visualization)
5. Choose and implement desired input pipeline (Options 1-3)
6. Use generated data with your classification model

The processed data and generators are ready for use with TensorFlow/Keras models for Deep Learning tasks.

## Contact

For contributions, questions, or collaborations, please don't hesitate to reach out via email :)
hosseini.sc95@gmail.com 
