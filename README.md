# 📧 Spam Mail Detector using Machine Learning

This project implements a **Spam Mail Detector** that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The project was developed as part of an **AI/ML internship task** and demonstrates a complete end-to-end text classification workflow.

---

## 📌 Problem Statement
Spam messages are unwanted and often harmful.  
The goal of this project is to build a machine learning model that can automatically detect whether a given SMS message is spam or not.

---

## 📂 Dataset
- **SMS Spam Collection Dataset**
- Contains **5,574 SMS messages**
- Labels:
  - `ham` → Normal message
  - `spam` → Unwanted message

---

## ⚙️ Technologies & Libraries Used
- Python
- NumPy
- Pandas
- NLTK
- Scikit-learn
- Matplotlib & Seaborn

---

## 🔍 Project Workflow

1. **Data Loading & Inspection**
   - Checked dataset shape and column structure
   - Removed unnecessary columns

2. **Text Preprocessing**
   - Converted text to lowercase
   - Removed punctuation
   - Removed stopwords
   - Created a cleaned text column

3. **Feature Extraction**
   - Used **TF-IDF Vectorizer**
   - Limited vocabulary to top 3000 important words

4. **Model Training**
   - Trained a **Multinomial Naive Bayes** classifier

5. **Model Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report (Precision, Recall, F1-score)

---

## 📊 Results
- **Accuracy:** ~97%
- High precision and recall for spam detection
- Model performs well on unseen test data

---

## ▶️ How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/SoumyMittal/Spam-Mail-Detector
2. Install required libraries:
   ```bash
   pip install numpy pandas nltk scikit-learn matplotlib seaborn
3. Open the notebook:
   ```bash
   jupyter notebook spam_mail_detector.ipynb
4. Run all cells from top to bottom.

📌 Conclusion

This project demonstrates how NLP and machine learning techniques can be effectively used for spam detection.
The Multinomial Naive Bayes model combined with TF-IDF features provides high accuracy and reliable performance.
