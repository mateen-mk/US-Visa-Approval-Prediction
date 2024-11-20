# 🛂 **US Visa Approval Prediction**

Welcome to the **US Visa Approval Prediction** project! 🎉  
This repository provides a machine learning pipeline to predict US visa approval outcomes, featuring robust data processing, model evaluation, and deployment capabilities.

---

## 🌟 **Key Features**

- **End-to-End Machine Learning Workflow** 🚀  
  Covers data ingestion, transformation, validation, training, evaluation, and deployment.
- **Interactive Web Application** 🌐  
  User-friendly web interface for real-time predictions.
- **Cloud Integration** ☁️  
  Utilizes AWS and MySQL for efficient data management.
- **Data Drift Analysis** 📊  
  Monitors data changes to maintain model reliability.
- **Modular Design** 🧩  
  Flexible and easy-to-extend architecture.

---

## 📁 **Current Project Structure**

```plaintext
├── .usvisaenv/               # Environment setup
├── artifact/                 # Generated artifacts (datasets, models, etc.)
│   └── <timestamp>/          # Timestamped folders for versioning
├── config/                   # Configuration files (model, schema)
├── logs/                     # Log files for tracking execution
├── notebooks/                # Jupyter notebooks for EDA and experiments
│   ├── EDA_Images/           # Supporting images for analysis
│   ├── USVisa_data_drift_dashboard.html
│   └── *.ipynb               # Jupyter notebooks
├── src/                      # Core project source code
│   ├── US_Visa_Approval/     # Main package
│   │   ├── components/       # ML pipeline components
│   │   ├── configuration/    # Configuration handlers
│   │   ├── constants/        # Constants and global settings
│   │   ├── data_access/      # Data handling utilities
│   │   ├── entity/           # Data structures and entities
│   │   ├── exception/        # Custom exception handling
│   │   ├── logger/           # Logging utilities
│   │   ├── pipline/          # Training and prediction pipelines
│   │   └── utils/            # General utilities
├── app.py                    # Web app entry point
├── demo.py                   # Script for showcasing functionalities
├── Dockerfile                # Docker configuration
├── LICENSE                   # License file
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── setup.py                  # Package setup file
```

---

## 🛠 **Setup Instructions**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/US-Visa-Approval.git
cd US-Visa-Approval
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv .usvisaenv
source .usvisaenv/bin/activate    # For Linux/MacOS
.usvisaenv\Scripts\activate       # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Web App
```bash
python app.py
```
Access the application at: [http://localhost:5000](http://localhost:5000)

---

## 🚀 **Pipeline Overview**

The project implements a structured machine learning pipeline with the following stages:

1. **Data Ingestion**  
   Collect and preprocess raw data.  
2. **Data Validation**  
   Validate schema and data quality.  
3. **Feature Engineering**  
   Transform raw data into meaningful features.  
4. **Model Training**  
   Train models with advanced algorithms like CatBoost.  
5. **Model Evaluation**  
   Evaluate model performance with metrics.  
6. **Deployment**  
   Deploy the model for real-time predictions.

📍 *Pipeline Diagram Placeholder:*  
![Pipeline Diagram](https://via.placeholder.com/800x400?text=Pipeline+Diagram+Here)

---

## 📝 **Available Notebooks**

| Notebook                      | Description                              |
|-------------------------------|------------------------------------------|
| `01_push_data_to_mysql.ipynb` | Push data to MySQL database.             |
| `02_fetch_data_from_mysql.ipynb` | Fetch and explore data from MySQL.     |
| `03_EDA.ipynb`                | Perform exploratory data analysis.       |
| `04_Feature_Engineering_and_Model_Training.ipynb` | Train and evaluate models. |
| `05_data_drift_evidently.ipynb` | Analyze data drift using Evidently AI. |

---

## 🔧 **Technologies Used**

| Technology        | Purpose                              |
|-------------------|--------------------------------------|
| **Python**        | Core programming language           |
| **Flask**         | Backend framework for the web app   |
| **CatBoost**      | Model training and evaluation       |
| **AWS S3**        | Cloud storage for model artifacts   |
| **MySQL**         | Relational database for data storage|
| **Evidently AI**  | Data drift analysis and visualization|

---

## 🎯 **Future Enhancements**

- Add support for other ML algorithms like XGBoost or LightGBM.  
- Build a dashboard for monitoring model performance.  
- Extend data drift detection for real-time systems.  

---

## 📜 **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 🤝 **Contributing**

Contributions are welcome! Follow these steps to get started:

1. Fork the repository.  
2. Create a new branch for your feature:  
   ```bash
   git checkout -b feature-name
   ```
3. Commit and push your changes.  
4. Submit a pull request.

---

## 📧 **Contact**

For queries or feedback, reach out to:  
📩 Email: your_email@example.com  
🌐 GitHub: [your_username](https://github.com/your_username)  

---

## 🌟 **Acknowledgments**