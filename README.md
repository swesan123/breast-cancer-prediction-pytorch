# Breast Cancer Prediction: Neural Network Regularization Techniques

This project demonstrates the application of **feedforward neural networks** in **PyTorch** for binary classification on the **Breast Cancer dataset** from scikit-learn.  
It explores optimization and regularization techniques including **early stopping**, **mini-batch SGD**, **dropout**, and **L1/L2 regularization**.

---

## Project Overview

Developed as part of **CS/SE 4AL3 – Applied Machine Learning (Fall 2025)** at McMaster University.  
The project investigates how different regularization strategies affect neural-network performance when predicting tumor malignancy from 30 biomedical features.

### Key Concepts
- Feedforward neural networks in PyTorch  
- Early stopping for stochastic gradient descent  
- Mini-batch SGD for training efficiency  
- Dropout and weight regularization for generalization  
- Model evaluation: accuracy, precision, recall  
- Visualization of training/validation loss curves  

---

## Repository Structure
```
4AL3-BreastCancerPrediction-PyTorch/
│
├── Assignment3_starter.ipynb     # Completed notebook
├── 4AL3_Assignment3.pdf          # Assignment instructions
│
├── plots/                        # Generated plots
│   ├── earlystop_loss_curve.png
│   ├── minibatch_vs_earlystop.png
│   ├── dropout_comparison.png
│   └── l1_l2_regularization.png
│
├── environment.yml               # Conda environment
├── README.md
└── .gitignore
```

---

## Model Specification

**Architecture**
- Input: 30 features  
- Hidden Layer 1: 32 neurons + ReLU  
- Hidden Layer 2: 16 neurons + ReLU  
- Output Layer: 1 neuron + Sigmoid  

**Loss Function:** Binary Cross-Entropy  
**Optimizers:** SGD / Mini-Batch SGD / Adam  
**Split:** 80 % train  |  20 % validation  
**Metrics:** Accuracy | Precision | Recall  

---

## Experimental Sections

### Part 1 – Early Stopped SGD
- Implement manual early-stopping logic  
- Stop when validation-loss improvement < threshold (≈ 1e-3)  
- Plot train/validation loss and report final accuracy  

### Part 2 – Mini-Batch SGD
- Train using DataLoader mini-batches  
- Compare accuracy, precision, recall vs early-stop SGD  
- Plot combined loss curves  

### Part 3 – Dropouts
- Add dropout layers (0.1 / 0.3 / 0.5 rates)  
- Train with mini-batch SGD and compare accuracies  
- Identify best dropout rate for generalization  

### Part 4 – L1 / L2 Regularization
- Add L2 via `weight_decay` (λ = 0, 1e-4, 1e-3)  
- Add L1 manually (λ = 1e-4)  
- Plot training/validation loss for each  
- Discuss effects on sparsity and generalization  

---

## Installing Conda

Install [Miniconda](https://www.anaconda.com/download/success) and verify:
```bash
conda --version
```

### Environment Setup
```bash
conda env create -f environment.yml
conda activate 4al3-breastcancer
```

---

## Running the Notebook
```bash
jupyter notebook Assignment3_starter.ipynb
```

Run all cells in order and ensure:
- All code blocks executed before submission  
- All plots visible  
- AI-usage disclaimer present in the first markdown cell  

---

## Example Outputs

| Experiment | Plot | Description |
|-------------|------|-------------|
| Early Stop | earlystop_loss_curve.png | Loss plateaus before overfitting |
| Mini-Batch | minibatch_vs_earlystop.png | Smoother convergence curve |
| Dropout | dropout_comparison.png | Effect of different dropout rates |
| L1/L2 | l1_l2_regularization.png | Validation loss with regularization |

---

## Tools and Technologies
- Python 3.12  
- PyTorch  
- scikit-learn  
- NumPy & Matplotlib  
- Conda for environment management  
- Jupyter Notebook / VS Code  

---

## Academic Integrity Notice
This notebook was completed individually for course 4AL3.  
Any AI-assisted code or text is documented in accordance with course policy.

---

## License
Licensed under the **MIT License** – see [LICENSE](./LICENSE).

---

## Author
**S. Pathmanathan**  
5th Year Software Engineering Student @ McMaster University  
Previous Co-op @ AMD (Datacenter GPU Validation)  
Focus: Applied ML, model interpretability, and software systems integration.
