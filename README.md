# **MLS Take-Home Assessment - Transaction Category Classification**
## **Author:** Chau Jia Yi
## **Date:** 11/02/2025

This project focuses on classifying **bank transaction descriptions** into predefined **categories** using an **Artificial Neural Network (ANN)** with **FastText embeddings**. The pipeline includes **data preprocessing, model training, evaluation, and inference**.  

---

## 📂 **Project Structure**
### **1️⃣ `dataset/`**
- **Contains raw and preprocessed transaction datasets** used for training and inference.
  - `bank_transaction.csv`: Provided dataset
  - `user_profile.csv`: Provided dataset
  - `preprocessed_bank_transaction.csv`: Merged, preprocessed dataset with numerical embeddings
  - `inference.csv`: Unseen instances used for model inference.

### **2️⃣ `models/`**
- **Saved trained models** for:
  - `ann/`: Trained **ANN models** for classification.
  - `fasttext/`: Custom-trained **FastText model** for generating transaction description embeddings.
  - `scaler/`: **StandardScaler** fitted on training set.

### **3️⃣ `source/`**
- **Contains all Jupyter notebooks** used for different stages of development:
  - `data_preprocessing.ipynb` → Cleans and preprocesses transaction data.
  - `model_training.ipynb` → Trains the ANN model using structured data and FastText embeddings.
  - `model_inference.ipynb` → Runs inference on unseen transactions.
  - `preprocess_train_pipeline.ipynb` → End-to-end pipeline combining preprocessing, training, and inference.

### **4️⃣ `html_report/`**

- Contains all four Jupyter notebooks in HTML format.

### **5️⃣ `results/`**
- Stores **evaluation results** from model training and testing, including:
  1.  **Loss and accuracy plots** (train vs. test).
  2.  **Confusion matrix visualizations**.
  3.  **Classification report (precision, recall, and F1-score).**

### **6️⃣ `util/`**
- Contains utility functions for reusability.

### **7️⃣ `requirements.txt`**
- Lists all **required Python packages** needed to run the project.

---

## 🚀 **Quickstart**
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 **Development Steps**

> 📌 *For detailed explanations, justifications, and methodology, please refer to **documentation_chau_jia_yi.pdf** in the project folder*.

1. **Data Preprocessing**: 
    - Data merging, cleaning and normalisation.
    - Text normalization, stopword removal, and handling of numeric + alphanumeric patterns.
    - FastText embeddings for textual transaction descriptions.
    - One-hot encoding for categorical data and target class.

2. **Model Training & Evaluation**:

    - Multilayer ANN using structured features + embeddings.
    - Batch Normalization and Dropout for better generalization.
    - Training pipeline with loss and accuracy tracking.
    - Visualizes Loss/accuracy curves to monitor training progress.
    - Generates Confusion Matrix & Classification Report for performance analysis.

3. **Inference**:

    - Predicts transaction categories for unseen transactions.
    
## **🔍 Why Use an Artificial Neural Network (ANN)?**

Artificial Neural Networks (ANNs) are chosen over traditional ML methods (e.g., Logistic Regression, Decision Trees, Random Forests) due to their ability to:

- **Capture complex, non-linear relationships** in transaction data, which may not be effectively modeled by traditional ML algorithms.
- **Work with high-dimensional feature representations**, such as FastText embeddings for transaction descriptions.
- **Perform well on large datasets** and **handle multi-class classification**.

Whereas DL models like CNNs or LSTMs are commonly used in image or sequential data processing, but since transaction classification is based on structured tabular and textual features, ANNs are cheaper and efficient without excessive computational overhead.




## 🔮 **Future Development Plans**

> 📌 *For detailed explanations on future ideas, please refer to **documentation_chau_jia_yi.pdf** in the project folder*.

- Optimize the ANN architecture via **hyperparameter tuning** for better classification accuracy.
- **Adjust model architecture** by experimenting with deeper neural networks (adding more hidden layers)
- Experiment with **different embeddings** (e.g., BERT or Word2Vec) for transaction descriptions.
- Leveraging **unsupervised deep clustering techniques**, offering the potential to enable new market entries without reliance on labeled data.
- Develop **APIs (backend architecture)** for transaction classification prediction and inference.

## 🙌 **Appreciation**

A huge thank you to the MoneyLion team for providing me the opportunity to try out on this task!

