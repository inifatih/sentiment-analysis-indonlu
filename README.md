# 🌟 Sentiment Analysis Bahasa Indonesia dengan IndoNLU & BERT

Proyek ini bertujuan untuk menganalisis sentimen kalimat dalam bahasa Indonesia menggunakan **IndoNLU** dan **BERT Transformer**. Model akan memprediksi apakah suatu kalimat memiliki sentimen **positif** atau **negatif**, serta memberikan persentase probabilitas dari prediksi tersebut.

---

## 📚 Struktur Folder
```
sentiment-analysis-indonlu/
│── sentiment-analysis.ipynb  # Notebook utama untuk sentiment analysis
│── requirements.txt          # Daftar dependensi
└── README.md                 # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Proyek

### **1️⃣ Clone Repository**
```bash
git clone https://github.com/inifatih/sentiment-analysis-indonlu.git
cd sentiment-analysis-indonlu
```

### **2️⃣ Install Dependensi**
Pastikan Python 3.8+ telah terinstal, lalu jalankan:  
```bash
pip install -r requirements.txt
```

### **3️⃣ Download Dataset IndoNLU**
Dataset dapat diunduh langsung dari IndoNLU:  
- [IndoNLU Dataset](https://github.com/indobenchmark/indonlu)  

Jika menggunakan dataset IndoNLU langsung dalam notebook, gunakan:
```python
from datasets import load_dataset

dataset = load_dataset("indonlu", "smsa")
```

---

## 📝 Model yang Digunakan
Model yang digunakan dalam proyek ini adalah **BERT Transformer** yang telah dilatih untuk bahasa Indonesia:
- IndoBERT (`indobenchmark/indobert-base-p1`)
- IndoBERT Lite (`indobenchmark/indobert-lite-base-p2`)
- Multilingual BERT (`bert-base-multilingual-cased`)

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=2)
```

---

## 💪 Cara Menjalankan Notebook
Untuk menjalankan notebook di Jupyter atau Google Colab, jalankan:  
```bash
jupyter notebook
```
Atau buka langsung di Google Colab dengan badge berikut:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inifatih/sentiment-analysis-indonlu/blob/main/sentiment-analysis.ipynb)

---

## 📊 Evaluasi Model
Setelah training, model dievaluasi menggunakan metrik seperti **accuracy**, **F1-score**, dan **confusion matrix**.  
```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 🔮 Hasil Prediksi Contoh
Setelah training, model dapat digunakan untuk mengklasifikasikan kalimat baru dengan persentase probabilitas untuk sentimen **positif** dan **negatif**:

```python
import torch
from torch.nn.functional import softmax

text = "Film ini sangat bagus dan menginspirasi!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
output = model(**inputs)
probabilities = softmax(output.logits, dim=1).detach().numpy()[0]

print(f"Prediksi Sentimen:\n- Positif: {probabilities[1] * 100:.2f}%\n- Negatif: {probabilities[0] * 100:.2f}%")
```

Contoh output:
```
Prediksi Sentimen:
- Positif: 85.74%
- Negatif: 14.26%
```

---

## 📅 Catatan Tambahan
- Model dapat ditingkatkan dengan **fine-tuning IndoBERT** menggunakan dataset tambahan.
- Bisa juga diterapkan untuk **multi-label classification** jika ingin mendeteksi lebih banyak kategori sentimen.

---

🔥 **Selamat mencoba! Jika ada pertanyaan atau saran, silakan buka issue atau pull request di repo ini.** 🚀

