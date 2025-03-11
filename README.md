# Locally Weighted Regression (LWR) vs Linear Regression (LR) on the Diabetes Dataset

## 📌 Overview
This project implements and compares **Locally Weighted Regression (LWR)** with **Linear Regression (LR)** using the **Diabetes Dataset**. It visualizes the differences between the two models using **3D plots** and evaluates performance across different bandwidth (`tau`) values.

## 📂 Repository Structure
```
├── diabetes_lwr.py         # Main script for data processing, model training, and visualization
├── README.md               # Project documentation
└── requirements.txt        # Dependencies for running the project
```

## 🔍 Dataset Information
- The **Diabetes Dataset** is a well-known dataset containing physiological measurements of diabetes patients.
- The dataset is loaded using `sklearn.datasets.load_diabetes()`.
- The target variable represents disease progression.

## 🚀 Features Implemented
✅ Data Loading & Preprocessing (Standardization, Feature Selection)  
✅ Exploratory Data Analysis (EDA) with **Seaborn**  
✅ Implementation of **Locally Weighted Regression (LWR)**  
✅ Implementation of **Linear Regression (LR) for Comparison**  
✅ **3D Visualization** of Regression Surfaces  
✅ Performance Evaluation using **RMSE & R² Score**  

## 📊 Model Comparison
### **1️⃣ Locally Weighted Regression (LWR)**
- A **non-parametric regression technique** where each query point has its own model.
- Uses a **Gaussian weighting function** based on distance.
- Controlled by **bandwidth (τ)**: Lower values focus more on nearby points.

### **2️⃣ Linear Regression (LR)**
- A **global model** trained once on all data.
- Assumes a **linear relationship** between features and target.

### **3️⃣ Performance Metrics (Added for Quantitative Comparison)**
- **Root Mean Squared Error (RMSE)**: Measures prediction error.
- **R² Score (Coefficient of Determination)**: Measures goodness of fit.

## 📈 Results Visualization
- The project generates **3D scatter plots** overlayed with prediction surfaces:
  - 🔴 **Red Points** → Actual Data
  - 🔵 **Cyan Surface** → LWR Predictions
  - 🟢 **Green Surface** → LR Predictions

## 🛠️ Installation & Usage
### **🔧 Setup the Environment**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-lwr.git
   cd diabetes-lwr
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python diabetes_lwr.py
   ```

## 📚 Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 🔮 Future Enhancements
- Implement **Kernel Ridge Regression** for further comparison.
- Optimize LWR computation using **vectorized operations**.
- Add **cross-validation** for better hyperparameter selection.

## 📜 License
This project is **open-source** and available under the MIT License.

## 🙌 Contributing
Feel free to **fork this repository**, raise issues, or submit pull requests for improvements!

📧 **Contact**: Reach out at officialsubho10@gmail.com or connect on www.linkedin.com/in/ameyo-jha-6917b3223.

---
Made with ❤️ by **Ameyo Jha**

