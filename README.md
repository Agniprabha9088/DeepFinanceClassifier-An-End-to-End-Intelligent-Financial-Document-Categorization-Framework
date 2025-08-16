# 🚀 Introducing DeepFinanceClassifier: An-End-to-End-Intelligent-Financial-Document-Categorization-Framework

## 🌟 Overview

Managing financial documents such as **Corporate Ledgers**, **Cash Movement Statements**, **Profit & Loss Reports**, **Explanatory Notes**, and **Miscellaneous Filings** manually can be tedious, error-prone, and time-intensive.
**DeepFinanceClassifier** solves this challenge by leveraging **Deep Learning** and **Natural Language Processing (NLP)** to **automatically classify financial documents** with remarkable accuracy.

Built with **TensorFlow** and a **Bidirectional Long Short-Term Memory (Bi-LSTM)** architecture, this system delivers **approximately 96% accuracy** while offering a **Streamlit-powered interface** for seamless document analysis. The solution is further **deployed on Hugging Face Spaces**, integrating advanced NLP preprocessing (**HTML parsing**, **tokenization**, **lemmatization**, **Word2Vec embeddings**) and **dataset balancing** (**SMOTETomek**) to deliver real-time, user-friendly financial document classification with high robustness and scalability, making it easily accessible for real-world usage.

---

## 📑 Table of Contents

* [Key Highlights](#-key-highlights)
* [Installation](#-installation)
* [Usage](#-usage)
* [Features](#-features)
* [Model Development](#-model-development)
* [Deployment](#-deployment)
* [Results](#-results)
* [Conclusion](#-conclusion)
* [References](#-references)
* [Contributing](#-contributing)
* [License](#-license)

---

## 🔑 Key Highlights

* **Core Stack**: Python, TensorFlow, Keras, Bi-LSTM RNN
* **NLP Tools**: spaCy, NLTK, Gensim (Word2Vec)
* **Data Processing**: BeautifulSoup, SMOTETomek balancing, NumPy, Pandas
* **Visualization & Utilities**: Matplotlib, WordCloud, SciPy
* **Deployment & Interface**: Streamlit + Hugging Face Spaces

---

## ⚙️ Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/Agniprabha9088/DeepFinanceClassifier.git
cd DeepFinanceClassifier
```

Install packages:

```bash
pip install tensorflow==2.12.0
pip install spacy nltk gensim imblearn numpy pandas matplotlib wordcloud scipy==1.12 streamlit streamlit_extras beautifulsoup4
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
```

⚠️ If you face `ImportError: DLL load failed` with TensorFlow:

```bash
pip uninstall tensorflow
pip install tensorflow==2.12.0 --upgrade
```

---

## ▶️ Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open the app at **[http://localhost:8501](http://localhost:8501)**

3. Upload an **HTML-based financial document** → View predicted class + confidence score.

---

## ✨ Features

### 📂 Dataset

* Documents are categorized into:

  * Balance Sheets
  * Cash Flow Statements
  * Income Statements
  * Notes
  * Others

### 🔧 Data Preprocessing

* **HTML parsing**: Extracted text using *BeautifulSoup*
* **Tokenization & Lemmatization**: NLTK + spaCy
* **Stopword & Noise Removal**: Cleaned redundant tokens and special characters

### 🧠 Word Embeddings

* Trained **Word2Vec** embeddings (300-dim vectors)
* Encoded classes & saved embedding model for reuse

### ⚖️ Class Balancing

* Applied **SMOTETomek** (SMOTE + Tomek Links)
* Balanced under-represented classes → improved generalization

### 📊 Data Preparation

* Converted features & targets into **TensorFlow tensors**
* Used optimized `tf.data` pipelines (cache, shuffle, prefetch) for efficient training

---

## 🏗️ Model Development

* **Architecture**: Bi-LSTM RNN with multiple layers
* **Regularization**: Dropout to reduce overfitting
* **Activations**:

  * `tanh` for hidden layers
  * `sigmoid` for LSTM forget gates
  * `softmax` for multiclass output
* **Optimizer & Loss**:

  * `Adam` optimizer
  * `SparseCategoricalCrossentropy` loss

✅ Achieved **96.2% test accuracy**

---

## 🌍 Deployment

* **Model Export**: Trained model saved for inference
* **App Development**: Streamlit-based UI for document upload & prediction
* **Hugging Face Spaces Deployment**: 

---

## 📊 Results

* Accuracy: **96.2%**
* Robust performance across all five financial document categories
* Easy to use interface → Upload → Classify → Visualize

---

## 📝 Conclusion

**DeepFinanceClassifier** demonstrates how **deep learning + NLP** can streamline **financial document management**.
From **data preprocessing, embeddings, balancing, model training, to deployment**, this project represents a **complete end-to-end pipeline** ready for enterprise-level adoption.

---

## 📚 References

* [spaCy Documentation](https://spacy.io/usage)
* [NLTK Documentation](https://www.nltk.org/)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [Gensim Documentation](https://radimrehurek.com/gensim/)
* [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repo
* Create a new branch
* Submit a pull request 🚀

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---


🔥 **DeepFinanceClassifier** → Automating financial document classification with **AI-powered precision**.

---
