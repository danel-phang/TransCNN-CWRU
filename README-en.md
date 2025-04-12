# TransCNN-CWRU: A CNN and Transformer-Based Bearing Fault Classification Model with 100% Accuracy

## Project Overview
This project implements a deep learning model combining Convolutional Neural Networks (CNN) and Transformers for classifying the CWRU bearing fault dataset. Validation and testing accuracy reach 100%.

## Dataset
- **Source**: CWRU Bearing Fault Dataset.
- **Format**: CSV files, with each column representing vibration signals of different fault categories.
- **Preprocessing**:
  - Each signal is segmented into sample blocks of length 1024.
  - Each sample block is labeled with the corresponding fault category.

## Model Architecture
The model integrates CNN and Transformer architectures. It first extracts local features of the signals through two layers of 1D convolution and downsamples using max pooling layers. Then, it captures global features using the multi-head self-attention mechanism and feedforward networks in the Transformer module. Finally, classification is performed through global average pooling and fully connected layers, outputting the probability distribution of 10 fault categories.

## Results and Visualization
### Cross-Validation Results
- Average validation accuracy: `99.8%`.
- Standard deviation: `±0.23%`.
- Accuracy and loss curves for training and validation are visualized.

### Test Set Results
- Test set accuracy: `100%`.

## Installation of Dependencies
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
   ```

## References
- Case Western Reserve University (CWRU) Bearing Fault Dataset: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
