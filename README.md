Emailspam


# Email Spam Classification using LSTM

An end-to-end Deep Learning pipeline built in Python to classify emails as **Spam** or **Ham (Not Spam)**. This project addresses class imbalance through downsampling, utilizes Natural Language Processing (NLP) for text preprocessing, and implements a Recurrent Neural Network (RNN) using an **LSTM (Long Short-Term Memory)** architecture built with TensorFlow/Keras.



# 🚀 Features
1. **Data Downsampling:** Balances the dataset to match the number of spam and ham emails, preventing model bias.
2. **Text Preprocessing Pipeline:** Automatic removal of the 'Subject' header, punctuation, and English stopwords.
3. **Data Visualization:** Generates distribution count plots and insight-driven **WordClouds** for both spam and ham categories.
4. **Deep Learning Model:** An LSTM-based binary classifier utilizing Word Embeddings, dynamic learning rate adjustment (`ReduceLROnPlateau`), and `EarlyStopping` to maximize accuracy without overfitting.


# 🛠️ Tech Stack & Libraries
1. **Framework:** TensorFlow / Keras
2. **Data Manipulation:** NumPy, Pandas
3. **Visualization:** Matplotlib, Seaborn, WordCloud
4. **NLP Processing:** NLTK (Natural Language Toolkit), Python `string` module
5. **Model Evaluation:** Scikit-learn


# 📋 Dataset
The notebook is configured to process the `spam_ham_dataset.csv`. It expects a CSV file containing at least the following columns:
* `text`: The raw email body context (including the subject line).
* `label`: The target text classification (`spam` or `ham`).



# 💻 Installation & Usage

# 1. Clone the Repository
```
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

```

# 2. Install Dependencies

Ensure you have Python installed, then install the required packages:

**Packages**: pip install numpy pandas seaborn matplotlib nltk wordcloud tensorflow scikit-learn



# 3. Running the Project

You can run this project locally by converting the script or directly in the cloud:

* **Google Colab (Recommended):** Upload the `Emailspam.ipynb` notebook to your Google Drive, open it via Google Colab, and run the cells sequentially. You will be prompted to upload your dataset file (`spam_ham_dataset.csv`) directly into the environment.

---

# 🧠 Model Architecture & Training

The model processes the text via the following layers:

1. **Embedding Layer:** Maps tokenized words into a dense 32-dimensional vector space.
2. **LSTM Layer (16 units):** Captures sequential and contextual dependencies within the email text.
3. **Dense Layer (32 units, ReLU):** Adds non-linearity and extracts deeper features.
4. **Output Layer (1 unit, Sigmoid):** Outputs a probability score between 0 and 1 for binary classification.

# Training Optimization:

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Callbacks:**
* `EarlyStopping` (monitors validation accuracy to prevent overfitting)
* `ReduceLROnPlateau` (scales down learning rate when validation loss plateaus)





# 📊 Results & Performance

The script evaluates performance on a separated 20% validation split and outputs:

* **Test Loss & Test Accuracy** metrics.
* **Training History Graph:** A plot visualizing Training Accuracy vs. Validation Accuracy over training epochs to verify optimal model convergence.



# 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute to further improving model metrics (e.g., trying bidirectional LSTMs or BERT fine-tuning).

