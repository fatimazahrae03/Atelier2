Atelier2: Image Classification with Deep Learning
This project explores different deep learning architectures for image classification, focusing on the MNIST dataset. The goal is to compare the performance of various models, including CNN, Faster R-CNN, and Vision Transformers (ViT), and evaluate their effectiveness for computer vision tasks.

Objectives
Establish a CNN Architecture using PyTorch to classify the MNIST dataset.

Implement Faster R-CNN for MNIST classification and compare its performance with CNN.

Compare the performance of both models using metrics like Accuracy, F1 Score, Loss, and Training Time.

Fine-tune pretrained models (VGG16, AlexNet) on the MNIST dataset and compare results with CNN and Faster R-CNN models.

Implement Vision Transformer (ViT) from scratch to classify MNIST and compare it with CNN and Faster R-CNN results.

Steps
Part 1: CNN Classifier
CNN Architecture:

Convolutional layers (with kernels, stride, padding)

Pooling layers

Fully connected layers

Optimizers: Adam, SGD

Regularization techniques: Dropout

Model implementation on GPU for faster computation.

Faster R-CNN:

Used a pretrained Faster R-CNN model.

Fine-tuned the model for the MNIST dataset.

Model Evaluation:

Metrics: Accuracy, F1 score, Loss, Training time.

Compared the performance of CNN and Faster R-CNN.

Pretrained Model Fine-Tuning:

Fine-tuned VGG16 and AlexNet on the MNIST dataset.

Compared the results of these models with CNN and Faster R-CNN.

Part 2: Vision Transformer (ViT)
ViT Model:

Built a Vision Transformer (ViT) model from scratch using PyTorch.

Followed a tutorial to ensure correct implementation.

Evaluation:

Evaluated the ViT model on the MNIST dataset.

Compared the performance of ViT with CNN and Faster R-CNN in terms of accuracy, F1 score, and training time.

Results
CNN vs Faster R-CNN: The CNN model achieved higher accuracy, but Faster R-CNN performed better on more complex datasets.

Fine-Tuned Pretrained Models: VGG16 and AlexNet performed better than CNN and Faster R-CNN after fine-tuning on MNIST, showing how pretrained models enhance classification tasks.

Vision Transformer: ViT showed competitive results, offering advantages in generalization but requiring more computational resources and training time compared to CNN.

Conclusion
Key Learnings:
Different deep learning architectures offer trade-offs in performance and computational requirements.

CNNs are simpler and faster, while more advanced models like Faster R-CNN and ViT excel at handling more complex tasks and datasets.

Pretrained models like VGG16 and AlexNet can significantly improve accuracy when fine-tuned for new datasets.

Insights:
ViT is promising for high-accuracy tasks but demands more computational power compared to CNNs.

While CNNs are ideal for simple tasks (like MNIST classification), architectures like ViT provide a more robust solution for larger, more complex datasets.

Summary
In this lab, I learned how to implement and evaluate different neural network architectures for computer vision tasks using PyTorch. I explored CNNs for image classification, Faster R-CNN for object detection, and Vision Transformers (ViT) for advanced classification. The project allowed me to compare model performance in terms of accuracy, training time, and other metrics. Overall, I gained deeper insights into the applications of various neural network architectures in computer vision.
