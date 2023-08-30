# CatBERTa: Text-Based Catalyst Property Prediction

CatBERTa is a Transformer-based energy prediction model designed for efficient catalyst property prediction. It addresses the challenge of predicting adsorption energy, a critical property in catalyst reactivity, without relying on precise atomic coordinates or complex graph representations. Instead, CatBERTa leverages the power of Transformer-based language models to process human-interpretable textual inputs and predict energies accurately.

## Key Features

- Predicts energy using textual inputs, by harnessing the power of a pre-trained RoBERTa encoder.
- Processes human-interpretable text to embed target features for energy prediction.
- Analyzes attention scores to reveal how CatBERTa focuses on the incorporated features.
- Achieves a mean absolute error (MAE) of 0.75 eV, comparable to vanilla Graph Neural Networks (GNNs).
- Enhances energy difference predictions by effectively canceling out systematic errors for chemically similar systems.

## Getting Started

Follow these steps to start using CatBERTa for predicting catalyst adsorption energy:

### Prerequisites

- Python 3.6 or later
- PyTorch [version]
- Transformers library [version]
- [Any other specific dependencies]

### Installation

1. Clone the CatBERTa repository:

   ```bash
   git clone https://github.com/your-username/CatBERTa.git
   cd CatBERTa
   
![framework_github](https://github.com/hoon-ock/CatBERTa/assets/93333323/cafeb4de-b859-4b2e-abb1-f5e56ac0e22f)

2. Dataset

The **Open Catalyst Project** dataset provides the foundation for training and evaluating CatBERTa. This dataset is a valuable resource containing essential information for catalyst screening and reactivity prediction. The dataset includes a diverse collection of structural relaxation trajectories of adsorbate-catalyst systems with their associated energies. 

To access the Open Catalyst Project dataset and learn more about its attributes, please refer to the official repository: [Open Catalyst Project Dataset](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md)

   
4. Checkpoints

For access to the model checkpoints, please reach out to us through jock@andrew.cmu.edu

### Contact
For questions or support, feel free to contact us at [jock@andrew.cmu.edu].
