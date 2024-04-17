# Handwritten Character Recognition

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to develop a machine learning model capable of classifying handwritten characters from images. Using a Convolutional Neural Network (CNN), the model is trained to accurately detect and classify individual characters, which can be utilized in various applications such as digitizing handwritten notes or automating form processing.

## Dataset
The dataset consists of images divided into three categories: training, validation, and testing. Each category contains subfolders labeled with the character the images represent. The dataset is structured as follows:
- **Train Folder**: 60% of the total images, used for training the model.
- **Validation Folder**: 20% of the total images, used for tuning the model's hyperparameters and preventing overfitting.
- **Test Folder**: 20% of the total images, used to evaluate the model's performance on unseen data.

## Sample Dataset
The dataset used in this project includes images of handwritten characters, divided into several classes representing different letters of the alphabet. Below is a sample from the dataset:

| Character | Sample Image |
|-----------|--------------|
| A         | ![Sample A](/character_dataset/Train2/A/1.jpg) |
| B         | ![Sample B](/character_dataset/Train2/B/1.jpg) |
| C         | ![Sample C](/character_dataset/Train2/C/1.jpg) |


## Data Preprocessing
Preprocessing steps include:
- Resizing all images to a standard size of 32x32 pixels to ensure uniformity.
- Converting images to grayscale to reduce complexity.
- Normalizing pixel values to the range [0, 1] for faster convergence during training.

## Model Architecture
The chosen model is a CNN with the following layers:
- Convolutional layers to extract features from the images.
- MaxPooling layers to reduce spatial dimensions.
- Dropout layers to prevent overfitting.
- A Flatten layer to convert the 2D feature maps into a 1D feature vector.
- Dense layers for classification.

![Model Architecture](/model/images/model_architecture.png)

The model was selected due to its proven effectiveness in image recognition tasks, particularly in recognizing patterns in visual input.

## Model Performance and Metrics

### Results
The latest training session yielded the following results:
- Final training accuracy: 96%
- Final validation accuracy: 95%
- Test accuracy on unseen data: 93.8%

### Evaluation Metrics
After training, the model was evaluated on the test dataset. Here are the results:

- **Accuracy**: 94.6%
- **Precision**: 94.7%
- **Recall**: 94.6%
- **F1-Score**: 94.5%

These metrics indicate the model's performance, and effectiveness in recognizing handwritten characters with high reliability, highlighting its strengths and areas for improvement.

### Training and Validation Accuracy
Throughout the training process, we monitored the model's accuracy on both the training and validation datasets. The following graph illustrates the model's performance over epochs:

![Training and Validation Accuracy](/model/images/training_validation_accuracy_graph.png)

![Training and Validation Loss](/model/images/training_validation_loss_graph.png)

As observed, the model's validation accuracy closely follows its training accuracy, indicating good generalization to unseen data.


### Confusion Matrix
The confusion matrix below provides insight into the model's performance across all character classes, highlighting its strengths and areas where improvement is needed.

![Confusion Matrix](/model/images/confusion_matrix.png)


## Conclusion
The CNN model demonstrates promising capabilities in recognizing handwritten characters, with potential applications in automated data entry and digital archiving. Future work will focus on refining the model by exploring more sophisticated architectures and training techniques to further improve accuracy and reliability.

## Contributing
Contributions to this project are welcome! To contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
