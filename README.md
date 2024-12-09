# Mammogram Analysis Model
A deep learning framework for automated mammogram analysis leveraging ResNet-50, CSA (Cross-Architectural Self-Supervision), and unsupervised mask generation for annotation-efficient segmentation and classification.

# Overview
Mammogram analysis often requires high-quality annotations for tasks like segmentation and classification. This project combines self-supervised learning and transfer learning to reduce reliance on large annotated datasets while maintaining robust performance.

The framework integrates:

- ResNet-50: A powerful backbone for feature extraction.
- Cross-Architectural Self-Supervision (CSA): Aligns embeddings from CNN and Transformer architectures to improve representation learning.
- Unsupervised Mask Generation: Creates pseudo-masks using clustering or motion-based methods to pretrain the segmentation model.
- Shared Backbone for CC and MLO Views: Utilizes a single backbone for craniocaudal (CC) and mediolateral oblique (MLO) mammogram views, enabling efficient multitask learning.
# Key Features
- Annotation-Efficient Segmentation:
Uses unsupervised pretext tasks to generate pseudo-masks as segmentation targets.
- Transfer Learning:
Pretrained ResNet-50 serves as a shared backbone, fine-tuned for both CC and MLO views.
