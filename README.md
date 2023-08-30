# CatBERTa: Text-Based Catalyst Property Prediction

CatBERTa is a Transformer-based energy prediction model designed for efficient catalyst property prediction. It addresses the challenge of predicting adsorption energy, a critical property in catalyst reactivity, without relying on precise atomic coordinates or complex graph representations. Instead, CatBERTa leverages the power of Transformer-based language models to process human-interpretable textual inputs and predict energies accurately.

## Key Features

- Employs the power of a pre-trained RoBERTa encoder to predict energy levels using textual inputs.
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

### Dataset

1. Structural data
  
   The **Open Catalyst Project** dataset provides the foundation for training and evaluating CatBERTa. This dataset is a valuable resource containing essential information for catalyst screening and reactivity prediction. The dataset includes a diverse collection of structural relaxation trajectories of adsorbate-catalyst systems with their associated energies. 

To access the Open Catalyst Project dataset and learn more about its attributes, please refer to the official repository: [Open Catalyst Project Dataset](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md)

2. Preprocessed textual data

   The `data` folder houses the preprocessed textual data derived from the Open Catalyst 2020 dataset. Due to storage limitations, we offer a small subset of our training and validation data as an illustrative example. This subset showcases the format and structure of the data that CatBERTa utilizes for energy prediction.

If you are interested in accessing the full training and validation dataset for more comprehensive experimentation, please don't hesitate to reach out to us.

### Checkpoints

For access to the model checkpoints, please reach out to us.

## Model Training

### Finetuning for energy prediction

The training configurations for CatBERTa can be found in the `config/ft_config.yaml` file. This configuration file contains important settings and hyperparameters that influence the training process.

During the training process, CatBERTa automatically creates and manages checkpoints to keep track of model progress. The checkpoints are saved in the `checkpoint/finetune` folder. This folder is created automatically if it doesn't exist and serves as the storage location for your model's intermediate states throughout the training procedure.

   ```bash
   $ python finetune_regression.py
   ```
## Analysis

### Energy and embedding prediction


   ```bash
   $ python catberta_prediction.py
   ```

### Attention score analysis

## Contact
questions or support, feel free to contact us at [jock@andrew.cmu.edu].
