# 📊 Vendor Invoice Intelligence System

### Freight Cost Prediction & Invoice Risk Flagging

An end-to-end Machine Learning system designed to help businesses analyze vendor invoices, **predict freight costs**, and **identify potentially risky invoices**.

The project combines **SQL-based feature engineering, exploratory data analysis, statistical testing, regression, classification, and an interactive Streamlit application** into a single machine-learning workflow.

---

## 🚀 Project Overview

Vendor invoices contain valuable information about purchasing, freight costs, vendors, quantities, and inventory. Manually analyzing this information can make it difficult to identify:

* Unexpected freight costs
* Vendor cost patterns
* Potentially risky invoices
* Purchasing anomalies
* Cost leakage and audit risks

The **Vendor Invoice Intelligence System** addresses these problems using two complementary ML models:

1. **Freight Cost Prediction** → Regression
2. **Invoice Risk Flagging** → Classification

The system works with relational data stored in a **SQLite database** and performs invoice-level aggregation and feature engineering using SQL.

---

## 🎯 Business Problem

### Invoice Cost Leakage & Audit Risk

Businesses need to understand whether invoice costs are reasonable and whether certain invoices require additional investigation.

The system focuses on two major business questions:

> **How much freight cost should we expect for an invoice?**

and

> **Is this invoice potentially risky and worth further investigation?**

---

## 💡 Business Objectives

### 1. Freight Cost Prediction

Predict the expected freight cost of an invoice using historical purchasing and vendor information.

This can help businesses:

* Estimate expected freight expenses
* Identify unusually expensive invoices
* Monitor vendor-related costs
* Improve cost analysis
* Support financial planning

### 2. Invoice Risk Flagging

Classify invoices based on their risk behavior.

This can help organizations:

* Prioritize invoices for review
* Identify unusual vendor behavior
* Support audit teams
* Detect potentially problematic transactions
* Reduce manual invoice screening

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │     Business Problem    │
                    │ Invoice Cost Leakage &  │
                    │       Audit Risk        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Business Objectives  │
                    │                         │
                    │ • Freight Cost          │
                    │   Prediction             │
                    │ • Risky Invoice         │
                    │   Flagging               │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │            SQLite Database           │
              │                                     │
              │ • vendor_invoice                    │
              │ • purchases                          │
              │ • purchase_prices                    │
              │ • inventory tables                   │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ SQL Feature Engineering │
                    │                         │
                    │ Invoice-Level           │
                    │ Aggregations             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      EDA & Analysis     │
                    │                         │
                    │ • Cost Patterns         │
                    │ • Risk Behavior         │
                    │ • Statistical Tests     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        Modeling         │
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        ┌────────────────────┐      ┌────────────────────┐
        │ Freight Cost       │      │ Invoice Risk       │
        │ Prediction         │      │ Flagging            │
        │                    │      │                    │
        │ Regression         │      │ Classification     │
        └────────────────────┘      └────────────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Streamlit App        │
                    │ Interactive ML Results   │
                    └─────────────────────────┘
```

---

## 🤖 Dual-Model Architecture

The project uses two ML pipelines because the business problems are different.

### Model 1 — Freight Cost Prediction

**Problem Type:** Regression

**Objective:** Predict the expected freight cost associated with an invoice.

```text
Invoice / Purchase Data
          ↓
SQL Aggregation
          ↓
Feature Engineering
          ↓
Data Preprocessing
          ↓
Regression Model
          ↓
Predicted Freight Cost
```

---

### Model 2 — Invoice Risk Flagging

**Problem Type:** Classification

**Objective:** Determine whether an invoice should be considered potentially risky.

```text
Invoice / Vendor / Purchase Data
              ↓
       SQL Aggregation
              ↓
      Feature Engineering
              ↓
       Data Preprocessing
              ↓
    Classification Model
              ↓
       Risk Prediction
              ↓
     Risk / Review Flag
```

---

# 🗄️ Data Architecture

The project uses a relational **SQLite database** instead of relying only on CSV files.

### Database

```text
inventory.db
```

### Main Data Sources

| Table              | Purpose                          |
| ------------------ | -------------------------------- |
| `vendor_invoice`   | Vendor invoice information       |
| `purchases`        | Purchase transaction information |
| `purchase_prices`  | Purchase price information       |
| `inventory tables` | Inventory-related information    |

The relational structure allows information from multiple business tables to be combined before creating invoice-level ML features.

---

# 🔧 SQL Feature Engineering

One of the key parts of the project is **invoice-level feature engineering**.

Instead of directly feeding raw transactional records into a model, the system aggregates relevant information at the invoice level.

Example feature categories include:

* Vendor information
* Purchase quantities
* Purchase prices
* Invoice values
* Historical cost behavior
* Freight-related information
* Vendor purchase patterns
* Transaction-level aggregations

Conceptually:

```text
Raw Relational Data
        ↓
JOIN
        ↓
GROUP BY Invoice
        ↓
Aggregations
        ↓
Business Features
        ↓
ML Dataset
```

This approach makes the ML dataset more representative of the actual business decision being modeled.

---

# 🔎 Exploratory Data Analysis

EDA is performed to understand the relationship between invoice attributes, costs, vendors, and risk behavior.

### Analysis Areas

* Freight cost distribution
* Invoice cost patterns
* Vendor behavior
* Purchase quantity patterns
* Price variations
* High-cost invoices
* Risk behavior
* Feature relationships
* Statistical significance

### Statistical Analysis

The project also uses **T-tests** to investigate relationships between relevant groups and determine whether observed differences are statistically meaningful.

---

# 🧠 Machine Learning Workflow

```text
Data Collection
      ↓
SQL Data Extraction
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Exploratory Data Analysis
      ↓
Statistical Analysis
      ↓
Train / Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Prediction
      ↓
Streamlit Application
```

---

# 📈 Model Evaluation

## Regression

The freight cost prediction model can be evaluated using regression metrics such as:

* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* R² — R-Squared

Example:

```text
Actual Freight Cost
        │
        ├──────────────┐
        │              │
        ▼              ▼
   ML Prediction    Error
        │
        ▼
   Regression Metrics
```

---

## Classification

The invoice risk model can be evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

For invoice-risk detection, **precision and recall** are particularly important because the business may want to minimize both unnecessary investigations and missed risky invoices.

---

# 🖥️ Streamlit Application

The project includes an interactive **Streamlit application** for demonstrating the ML pipeline.

The application can provide an interface for:

* Loading invoice-related information
* Generating predictions
* Viewing predicted freight costs
* Viewing invoice risk classifications
* Understanding model outputs
* Exploring business-oriented results

The goal is to make the ML models accessible through a simple user interface rather than requiring users to execute notebooks manually.

---

# 🛠️ Technology Stack

### Programming

* Python

### Database

* SQLite
* SQL

### Data Science

* Pandas
* NumPy
* Matplotlib

### Machine Learning

* Scikit-learn
* Regression Models
* Classification Models

### Statistical Analysis

* Statistical Testing
* T-tests

### Application

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/ 
```

```bash
cd vendor-invoice-intelligence
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

```bash
streamlit run app/app.py
```
or 
```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

---

# 📊 Business Value

The system demonstrates how machine learning can be applied to a real-world finance and procurement use case.

### Potential business benefits

* 💰 Identify unusual freight costs
* 🔍 Prioritize invoices for audit
* 📉 Reduce potential cost leakage
* 🏢 Analyze vendor behavior
* ⚡ Automate invoice screening
* 📊 Support procurement decisions
* 🤖 Introduce ML into finance workflows

---

# 🌟 Key Project Highlights

* ✅ End-to-end ML workflow
* ✅ Dual-model architecture
* ✅ Regression + classification
* ✅ Relational SQLite database
* ✅ SQL-based feature engineering
* ✅ Invoice-level aggregation
* ✅ Exploratory data analysis
* ✅ Statistical testing
* ✅ Business-oriented ML problem
* ✅ Interactive Streamlit application
* ✅ Git/GitHub ready project structure

---

# 🧩 Challenges Addressed

### 1. Combining Multiple Data Sources

Invoice-related information may be distributed across multiple relational tables.

**Solution:**
Use SQL joins and invoice-level aggregations to create a unified modeling dataset.

### 2. Different ML Objectives

Freight cost prediction and invoice risk detection are different ML problems.

**Solution:**
Build separate regression and classification pipelines.

### 3. Translating Business Problems into ML Features

Raw database fields do not always directly represent business behavior.

**Solution:**
Create aggregated and business-oriented features at the invoice level.

### 4. Making ML Results Accessible

A notebook-based model is difficult for non-technical users to interact with.

**Solution:**
Expose predictions through a Streamlit application.

---

# 🔮 Future Improvements

Possible future improvements include:

* [ ] Hyperparameter optimization
* [ ] Advanced feature selection
* [ ] Model explainability using SHAP
* [ ] Automated model retraining
* [ ] MLflow experiment tracking
* [ ] Model monitoring
* [ ] REST API using FastAPI
* [ ] PostgreSQL production database
* [ ] Docker deployment
* [ ] Cloud deployment
* [ ] Automated anomaly detection
* [ ] Role-based application access
* [ ] Automated invoice ingestion
* [ ] Dashboard with business KPIs

---

# 📌 Learning Outcomes

This project demonstrates practical experience in:

* Python programming
* SQL
* Relational databases
* Data preprocessing
* Feature engineering
* Exploratory data analysis
* Statistical analysis
* Regression
* Classification
* Model evaluation
* Machine-learning pipelines
* Streamlit application development
* Git/GitHub workflow
* Translating business requirements into ML solutions

---

## ⭐ Project Purpose

This project was developed as a practical demonstration of how **Machine Learning, SQL, and application development can be combined to solve real-world finance and procurement problems**.

If you find this project useful, consider giving the repository a ⭐.

---

## 📜 License

This project is intended for educational and portfolio purposes.

Add an appropriate open-source license if you plan to allow others to reuse the source code.
