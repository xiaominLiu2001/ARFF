# Adaptive Regional Feature Fusion for Brain WSI Iron Concentration Grading in Neurodegenerative Research

This project addresses the classification of iron concentration levels in brain whole slide images (WSIs) to support neurodegenerative disease research. The approach utilizes adaptive regional feature fusion to accurately grade iron concentrations in brain tissue samples.

## Environment Setup

Before running the project, ensure all required dependencies are installed. Use the following command to install the dependencies:


### Step 1: Mask Generation
The first step involves using a segmentation model to generate mask files (in JSON format) for the dataset. These masks identify regions of interest in the brain WSIs.

### Step 2: Feature Extraction
Run the Python scripts in the `process_tiff` directory to process the TIFF images. This step produces:
- NPZ files containing extracted features
- JSON files containing cluster indices information

### Step 3: Training and Testing
Execute the Python scripts in the `execute` directory to train the classification model and test its performance on the dataset.
