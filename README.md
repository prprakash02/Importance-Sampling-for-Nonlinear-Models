# Importance Sampling for Nonlinear Models

This repository contains code for training a model and computing nonlinear importance scores on both **image classification** tasks and **regression** tasks. A few straightforward modifications could be necessary depending on the dataset and the sampling methods.

---

## File Structure

1. **`ImageModel.py`**  
   - Defines a simple neural network architecture (`SimpleNNBinary`) for image-based binary classification (after reducing the multi-class labelled dataset if required).  

2. **`ImageTrain.py`**  
   - Contains the main training loop for image-based tasks.  
   - Demonstrates data loading, transforms, and train the model defined in `ImageModel.py`.  
   - Includes code for saving the trained model weights.

3. **`ImageNLLS.py`**  
   - Computes **Non-Linear Leverage Scores (NLLS)** for image data.  
   - Loads the previously trained model and performs `F*` calculation.  
   - Outputs results in CSV format, along with any relevant statistics and diagnostic plots.

4. **`RegTrain.py`**  
   - Provides a training pipeline for a **regression** problem using a single-neuron (index) model.  
   - Demonstrates custom activation functions (Swish) and uses typical regression losses (MSELoss).  

5. **`RegIS.py`**  
   - Similar to `ImageNLLS.py` but specialized for **regression** tasks and other importance sampling methods.  

6. **`README.md`**  
   - This document provides an overview of the repository and instructions on how to use the scripts.

---

## Setup and Installation

1. **Environment**  
   - This codebase requires Python 3.11+ (or compatible).  
   - Libraries such as PyTorch, NumPy, Pandas, Matplotlib, and scikit-learn are needed.  
2. **Project Structure**  
   - Ensure that the `.py` files (e.g., `ImageModel.py`, `Imagerain.py`) are in the same directory so that imports work correctly.
   - For tasks, ensure your paths are defined properly (e.g., `DATA_PATH`).

---

**End of README**