# Plant Classification Project

## Overview
This repository hosts a PyTorch-based Convolutional Neural Network (CNN) implementation for classifying 12 plant species from the Plant Seedlings Classification dataset. The project evaluates three activation functions (ReLU, Leaky ReLU, ELU) and three classifiers (CNN with Softmax, SVM, XGBoost), optimized for GPU training. Due to resource constraints, the latest results reflect a subset of configurations, but the code supports generating performance metrics (accuracy, F1-score, ROC-AUC) and visualizations.

## Dataset
- **Source**: Included within the repository.
- **Classes**: 12 species (e.g., Black-grass, Common Chickweed) with aliases (e.g., BG, CW).
- **Size**: ~4,750 training images (20% validation split), 794 test images.
- **Preprocessing**: Images resized to 32x32, normalized with ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]).
- **Augmentation**: Random horizontal flips and 10° rotations.

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>

## Install dependencies
2. pip install -r requirements.txt

## Mount to google colab
3. python 
from google.colab import drive
drive.mount('/content/drive')
## adjust data path into your working directory/ to match your local setup.

3. cp -r /content/drive/data

4. Run notebook  as it uses GPU or GPu on google colab 
use main.py if running locally(adjust to GPu/CPu)

## Outputs
Outputs: results/results.csv, results/eda_plots/

## Results
Best Model: ReLU + CNN + SVM with configuration [32, 64, 128]_[1, 1, 1] achieves 0.886 accuracy, 0.883 F1-score, and 0.980 ROC-AUC.
Performance Trends: ReLU consistently outperforms Leaky ReLU and ELU across configurations, with CNN + SVM excelling over Softmax and XGBoost. Early stopping (epoch 6) mitigated overfitting in underperforming models (e.g., ReLU + Softmax at 0.446).
Details: See results.csv for full metrics across all tested configurations.

## Files
notebook: notebook.ipynb google colab execution

if you want to run locally: 
main.py.
model.py
train.py
evaluate.py
data_loader.py 

requirements.txt: Dependencies (e.g., PyTorch 1.13.1, scikit-learn 1.2.2).
results/: Output files (results.csv, eda_plots).


## Class Aliases

Black-grass: BG
Charlock: CH
Cleavers: CL
Common Chickweed: CW
Common wheat: WH
Fat Hen: FH
Loose Silky-bent: LS
Maize: MA
Scentless Mayweed: SC
Shepherds Purse: SH
Small-flowered Cranesbill: SF
Sugar beet: SB

README.md: This file.


## Future Improvements

Enhance data augmentation (e.g., vertical flips, color jitter) to address class imbalance.
Tune hyperparameters (e.g., lower learning rate to 0.00001).
Explore deeper CNN architectures or additional filter configurations.
Conduct class-wise performance analysis to identify underpredicted classes.

## Acknowledgments
Tools: PyTorch, scikit-learn, XGBoost, Google Colab.

License

[MIT License] - Feel free to modify; specify your preferred license if different.