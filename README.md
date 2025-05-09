## Overview
This project explores whether machine learning can predict the optical depth (τ) of the early universe from kSZ (kinetic Sunyaev-Zel'dovich) heat maps, using a dataset of 1,000 simulations provided by Dr. Paul La Plante. These simulations model various reionization scenarios and generate corresponding τ values and 2D kSZ maps of the Cosmic Microwave Background (CMB).

## Motivation
The τ value summarizes how much ionized gas scattered CMB photons along their journey from the early universe. Since the relationship between spatial kSZ features and τ is complex and nonlinear, we trained a neural network to learn this mapping and estimate τ values directly from kSZ maps.

## Dataset

<img width="269" alt="Tau Value" src="https://github.com/user-attachments/assets/5a5986a8-162c-403b-8614-a89f69a8ee3f" />

![kSZ Map](https://github.com/user-attachments/assets/3dd55538-1bff-46af-a598-87d00592aa9c)


* **Source**: Simulations by Dr. Paul La Plante
* **Size**: 1,000 reionization scenarios
* **Each simulation includes**:
  * **kSZ Heat Map**: A 1024×1024 image of temperature shifts in the CMB
  * **Tau (τ) Value**: Optical depth to the CMB
* **Model Type**: Semi-numeric "inside-out" reionization model

## Code Workflow
* Data loading and normalization
* Custom PyTorch dataset object
* Neural network architecture (CNN-based)
* CUDA acceleration for GPU training
* Training/validation split
* Model evaluation and prediction

## Analysis & Results

### Training Behavior

![Loss Curve](https://github.com/user-attachments/assets/9ccc1274-3622-4812-82bc-19a62da68a96)

* Sharp drop in loss after epoch 1
* No divergence between traiing and validation loss
* No signs of overfitting

### Acccuracy Check

![Predicted vs True Tau Values](https://github.com/user-attachments/assets/e7db4921-1d6a-4e4a-b525-b49c5acab269)

* Model performs best on mid-range τ values (~0.4–0.5)
* Underestimates high τ and overestimates low τ

### Residuals

![Residual Plot](https://github.com/user-attachments/assets/11d675fe-3c47-4ec5-beec-3b7a135c112a)

* Clear signs of regression to the mean
* Overestimation at low τ, underestimation at high τ

### Prediction Error Histogram

![Histogram of Prediction Errors](https://github.com/user-attachments/assets/89916b02-0456-40a5-bca5-6aba954291bd)

* Slight skew toward underestimating τ
* Most errors centered around zero

## Conclusion
Successfully built a neural network to predict τ from simulated kSZ maps. Model captures trends but shows systematic bias toward the mean. Future work could include more complex models or better feature extraction techniques.

### Contributors
- Olivia Betancourt
- Sidney Lopez
- Alexander Gauthier
- Robert Pearce
