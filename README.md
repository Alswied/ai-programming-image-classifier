
# AI Programming Image Classifier

## Project Overview
This project implements an image classification application using **PyTorch** and **transfer learning** to recognize flower species from the [**Oxford 102 Flowers dataset**](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

The project is developed as part of the **Udacity – AI Programming with Python Nanodegree Certification**.  
In **Part 1**, the image classifier is built and trained inside a Jupyter Notebook.  
In **Part 2**, the solution is converted into a **command-line application** for training models and making predictions.

The application allows users to:
- Train a deep learning image classifier from the command line
- Save trained models as checkpoints
- Predict flower names along with their probabilities for new images

---

## Project Creation Date
30.03.2026

## Project Structure
train.py
- Command-line script used to train a deep learning image classifier and save a model checkpoint.

predict.py
- Command-line script used to load a trained checkpoint and predict flower names and probabilities for a given image.

utils.py
- Utility functions for image preprocessing, checkpoint loading, and prediction logic.

cat_to_name.json
- JSON file mapping class indices to actual flower names.

Image Classifier Project.ipynb
- Part 1 development notebook where the classifier is implemented and trained.

Image Classifier Project.html
- Exported HTML version of the Part 1 notebook (required for Udacity submission).

## Training the Model
To train a new model from the command line:

```bash
python train.py flowers
```
## Supported Training Options
- --arch: Choose model architecture (vgg16 or vgg13)
- --learning_rate: Set learning rate
- --hidden_units: Number of hidden units in the classifier
- --dropout: Dropout probability
- --epochs: Number of training epochs
- --gpu: Enable GPU training if available
- --save_dir: Directory to save the checkpoint

**Default values:**
- arch: vgg16
- hidden_units: 512
- dropout: 0.5
- learning_rate: 0.001
- epochs: 5
- batch_size: 64

## Predicting Flower Classes
To predict the flower name and probability for an image:

```bash
python predict.py image.jpg checkpoint.pth --top_k 4 --category_names cat_to_name.json
```
## Supported Prediction Options
- --top_k: Number of top predictions to return
- --category_names: JSON mapping file for flower names
- --gpu: Enable GPU inference if available

**Default values:**
- top_k: 5
- category_names: cat_to_name.json
- GPU disabled by default

## Note:
The dataset is not included in this repository and must be downloaded separately for local training and testing, as required by Udacity submission guidelines.

## Credits

This project is part of the **Udacity AI Programming Nanodegree**.

Starter concepts, dataset references, project structure, and the category mapping file (`cat_to_name.json`) are provided by **Udacity**.

Udacity reference repository:  
https://github.com/udacity/aipnd-project


## License
This project is licensed under the MIT License.
See the LICENSE file for more details.