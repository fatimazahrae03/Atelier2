# Atelier2: Image Classification with Deep Learning

This project explores various deep learning architectures for image classification, focusing on the **MNIST** dataset. The goal is to compare the performance of multiple models, including **CNN**, **Faster R-CNN**, and **Vision Transformers (ViT)**, and evaluate their effectiveness for computer vision tasks.

---

## Objectives

- **CNN Architecture**: Establish a CNN model using PyTorch to classify the MNIST dataset.
- **Faster R-CNN**: Implement Faster R-CNN for MNIST classification and compare its performance with CNN.
- **Model Comparison**: Compare the performance of CNN and Faster R-CNN using metrics like **Accuracy**, **F1 Score**, **Loss**, and **Training Time**.
- **Fine-tuning Pretrained Models**: Fine-tune **VGG16** and **AlexNet** on the MNIST dataset and compare results with CNN and Faster R-CNN.
- **Vision Transformer (ViT)**: Implement Vision Transformer (ViT) from scratch to classify MNIST and compare its performance with CNN and Faster R-CNN.

---

## Steps

### Part 1: CNN Classifier

#### CNN Architecture:
- **Convolutional Layers**: Used kernels, stride, and padding for feature extraction.
- **Pooling Layers**: Applied max pooling for dimensionality reduction.
- **Fully Connected Layers**: Flattened the output and connected it to dense layers.
- **Optimizers**: Used **Adam** and **SGD** for model training.
- **Regularization**: Applied **Dropout** to avoid overfitting.
- **GPU Computation**: Implemented the model to run on GPU for faster training.

#### Faster R-CNN:
- Used a **pretrained Faster R-CNN model** and fine-tuned it for the MNIST dataset.

#### Model Evaluation:
- **Metrics**: Accuracy, **F1 score**, Loss, and **Training time**.
- **Comparison**: Compared CNN and Faster R-CNN models' performance.

#### Pretrained Model Fine-Tuning:
- Fine-tuned **VGG16** and **AlexNet** pretrained models on the MNIST dataset.
- Compared the results of these models with CNN and Faster R-CNN.

  
| **Modèle**      | **Loss d'Entraînement** | **Précision** | **Temps d'Entraînement** |
|-----------------|-------------------------|---------------|  ------------------------|
| **CNN**         | 0.0229                  | 0.9927        |  3 minutes               |
| **Faster R-CNN**| 0.0173                  | 0.9897        |  1 minute                |
| **VGG16**       | 0.0283                  | 0.9919        |  2939.66 secondes        |
| **AlexNet**     | 0.0233                  | 0.9932        |  507.70 secondes         |

---

### Part 2: Vision Transformer (ViT)

#### ViT Model:
- Built a **Vision Transformer (ViT)** model from scratch using PyTorch.
- Followed a detailed tutorial to ensure correct implementation.

#### Evaluation:
- Evaluated the **ViT model** on the MNIST dataset.
- Compared ViT's performance with CNN and Faster R-CNN using metrics such as **Accuracy**, **F1 score**, and **Training time**.

---

## Results

### CNN vs Faster R-CNN:
- The **CNN model** achieved higher accuracy on the MNIST dataset, as it is a simpler task. However, **Faster R-CNN** performed better on more complex datasets, showing its effectiveness in object detection.

### Fine-Tuned Pretrained Models:
- **VGG16** and **AlexNet** outperformed the CNN and Faster R-CNN models after fine-tuning on the MNIST dataset, highlighting the power of **transfer learning** for image classification tasks.

### Vision Transformer (ViT):
- **ViT** showed **competitive results**, offering advantages in generalization. However, it required more **computational resources** and **training time** compared to CNN.

---

## Conclusion

### Key Learnings:
- **CNNs** are faster and simpler, making them ideal for small datasets like MNIST.
- **Faster R-CNN** and **ViT** excel in handling more complex tasks, especially with larger datasets.
- Pretrained models like **VGG16** and **AlexNet** significantly boost performance when fine-tuned on a new dataset.
  
### Insights:
- **ViT** provides promising results, especially for high-accuracy tasks, but demands more computational power than CNN.
- While CNNs are ideal for simple tasks, **ViT** and **Faster R-CNN** offer robust solutions for larger, more complex datasets.

---

## Summary

In this lab, I learned how to implement and evaluate different neural network architectures for computer vision tasks using **PyTorch**. I explored **CNNs** for image classification, **Faster R-CNN** for object detection, and **Vision Transformers (ViT)** for advanced classification. The project allowed me to compare model performance across **accuracy**, **training time**, and other important metrics. Overall, this lab provided valuable insights into the practical applications of neural network architectures in **computer vision**.

