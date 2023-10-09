# Deep-Learning-for-Classification
I'm excited to share a project I recently completed as part of the IBM Deep Learning and Reinforcement Learning course. This endeavor was an exploration into the world of Convolutional Neural Networks (CNNs) using TensorFlow and Keras, focusing on the classification of handwritten digits from the MNIST dataset.

Key Highlights:

Custom Data Loading: Leveraged Python's struct module to create custom functions that load MNIST images and labels from binary files. This allowed for a more hands-on approach to data ingestion, giving me a deeper understanding of the dataset's structure.

Data Preprocessing: Implemented normalization techniques to scale pixel values between 0 and 1, ensuring optimal model training. Additionally, reshaped the images to include the grayscale channel, making them compatible with Keras' Conv2D layer.

Model Architectures:

Basic CNN Model: A foundational model to set the benchmark for performance.

Regularized CNN Model: Incorporated dropout layers to reduce overfitting and improve generalization.

Deeper CNN Model: Explored a more complex architecture with additional convolutional layers to capture intricate patterns in the data.

Model Evaluation: After rigorous training, I evaluated each model's performance on a test dataset, gaining insights into their respective accuracies and potential areas of improvement.

Dataset Source: While the MNIST dataset is widely available, I chose to use a version from Kaggle, ensuring that I worked with a dataset that might have slight variations, adding to the challenge.

This project was a significant learning experience, reinforcing my understanding of deep learning concepts and best practices. It was a pleasure to see theoretical knowledge translate into practical application, resulting in models that can accurately classify handwritten digits.

I'm always open to feedback and collaboration. If you're interested in discussing this project further or exploring other deep learning topics, please feel free to connect!

