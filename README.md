# Adaptive Regional Feature Fusion for Brain WSI Iron Concentration Grading in Neurodegenerative Research

This project addresses the classification of iron concentration levels in brain whole slide images (WSIs) to support neurodegenerative disease research. The approach utilizes adaptive regional feature fusion to accurately grade iron concentrations in brain tissue samples.
![图片描述（可选）](images/framework_ARFF.png)
## Environment Setup

Before running the project, ensure all required dependencies are installed. Use the following command to install the dependencies:


### Step 1: Mask Generation
The first step of the workflow is to generate mask files in JSON format. In this study, we employed a DeepLabV3-based segmentation model to create masks highlighting gray matter regions in brain WSIs. If you already have your own ROI annotations in JSON format (for example, for other specific regions), you can directly use them instead of generating new masks.

### Step 2: Feature Extraction
Run the Python scripts in the `process_tiff` directory to process the TIFF images. This step produces:
- NPZ files containing extracted features
- JSON files containing cluster indices information

### Step 3: Training and Testing
Execute the Python scripts in the `execute` directory to train the classification model and test its performance on the dataset.
