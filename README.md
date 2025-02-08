# **MLS Take-Home Assessment - Transaction Category Classification**
## **Author:** Chau Jia Yi
## **Date:** 11/02/2025

This project focuses on classifying **bank transaction descriptions** into predefined **categories** using an **Artificial Neural Network (ANN)** with **FastText embeddings**. The pipeline includes **data preprocessing, model training, inference, and evaluation**.  

---

## 📂 **Project Structure**
### **1️⃣ dataset/**
- **Contains preprocessed transaction datasets** used for training and inference.

### **2️⃣ models/**
- **Saved trained models** for:
  - **ANN_20e_1e-3lr_4l_classifier.pth** → Trained ANN model for classification.
  - **fasttext_model.bin** → Custom-trained **FastText model** for generating transaction description embeddings.

### **3️⃣ source/**
- **Contains all Jupyter notebooks** used for different stages of development:
  - `data_preprocessing.ipynb` → Cleans and preprocesses transaction data.
  - `model_training.ipynb` → Trains the ANN model using structured data and FastText embeddings.
  - `model_inference.ipynb` → Runs inference on unseen transactions.
  - `preprocess_train_pipeline.ipynb` → End-to-end pipeline combining preprocessing, training, and inference.

### **4️⃣ results/**
- Stores **evaluation results** from model training and testing, including:
  1.  **Loss and accuracy plots** (train vs. test).
  2.  **Confusion matrix visualizations**.
  3.  **Classification report (precision, recall, and F1-score).**

### **5️⃣ util/**
- Contains utility functions for reusability.

### **6️⃣ requirements.txt**
- Lists all **required Python packages** needed to run the project.

---

## 🚀 **Quickstart**
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 **Key Features**

> 📌 *For detailed explanations, justifications, and methodology, please refer to **report.pdf** in the project folder*.

1. **Data Preprocessing**: 
    - Text normalization, stopword removal, and handling of numeric + alphanumeric patterns.
    - FastText embeddings for textual transaction descriptions.
    - One-hot encoding for categorical data and target class.

2. **Model Training**:

    - Multilayer ANN using structured features + embeddings.
    - Batch Normalization and Dropout for better generalization.
    - Training pipeline with loss and accuracy tracking.

3. **Inference & Evaluation**:

    - Predicts transaction categories for unseen transactions.
    - Visualizes Loss/accuracy curves to monitor training progress.
    - Generates Confusion Matrix & Classification Report for performance analysis.

## 🔮 **Future Development Plans**

> 📌 *For detailed explanations on future ideas, please refer to **report.pdf** in the project folder*.

- Experiment with **different embeddings** (e.g., BERT or Word2Vec) for transaction descriptions.
- Optimize the ANN architecture via **hyperparameter tuning** for better classification accuracy.
- Leveraging **unsupervised deep clustering techniques**, offering the potential to enable new market entries without reliance on labeled data.
- Develop **APIs (backend architecture)** for transaction classification prediction and inference.

## 🙌 **Appreciation**

A huge thank you to the MoneyLion team for providing me the opportunity to try out on this task!

