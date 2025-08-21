# Cyla: Web-Based PCOS Risk Detection & Menstrual Health Management

🎓 **Final Year Project (FYP)** 
🔐 A privacy-first, intelligent, tiered web app for early PCOS risk screening and menstrual health education.

## 🌟 Overview
**Cyla** is a free, secure, and accessible web application designed to help women understand their menstrual health and assess their risk for **Polycystic Ovary Syndrome (PCOS)** - a common hormonal disorder affecting 1 in 10 women.

Unlike commercial apps (e.g., Flo, Clue), Cyla does **not collect or sell user data**. Instead, it uses **self-reported symptoms and optional clinical data** to provide:
- Menstrual phase detection
- Rule-based symptom checker
- **ML-powered risk prediction (XGBoost)**
- Educational content
- PDF reports for doctors

This project bridges the gap between **menstrual tracking**, **early diagnosis**, and **health equity**.

## 🎯 Key Features
✅ **Tiered Risk Assessment**  
- Basic: Self-reported symptoms only (no lab tests needed)  
- Advanced: Optional clinical data (LH, FSH, BMI) → higher accuracy

✅ **XGBoost Hybrid Model**  
- Two trained models:  
  - `NC_XGB_model.pkl` – Non-clinical (18 features)  
  - `ALL_XGB_model.pkl` – Full clinical + non-clinical (39 features)  
- Accuracy: **92.3%**, AUC = 0.94

✅ **Educational Hub**  
- Learn about PCOS, hormones, and cycle phases

✅ **PDF Report Generator**  
- Download/share risk results with healthcare providers

✅ **History Tracking**  
- View past risk assessments and track progress

✅ **Privacy-First Design**  
- Data stored locally in MySQL
- Passwords hashed with `bcrypt`
- No third-party tracking

## 🛠️ Tech Stack
| Layer | Technology |
|------|------------|
| **Frontend** | Streamlit (Python) |
| **Backend** | Python 3.10 |
| **ML Model** | XGBoost, Scikit-learn |
| **Database** | MySQL + DigitalOcean |
| **PDF Generation** | FPDF2 |
| **Deployment** | Streamlit Community Cloud |

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cyla-pcos.git
cd cyla-pcos
