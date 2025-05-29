# 🛡️ Network Security Using ETL Pipelines

## 🧠 Abstract

This project presents a comprehensive, automated, and scalable network security solution designed to detect and mitigate cyber threats in real time. It integrates an ETL (Extract, Transform, Load) pipeline that collects raw network traffic logs from sources like firewalls, IDS/IPS, and email gateways. These logs are preprocessed to remove inconsistencies and stored in a NoSQL database (MongoDB) for efficient querying. Using structured data, the system applies machine learning models—specifically ensemble methods like Random Forest with AdaBoost—to identify patterns associated with phishing, malware, and unauthorized access attempts. The entire pipeline is orchestrated with tools like FastAPI for rapid deployment, MLflow for experiment tracking, and Docker for scalability and portability. With a focus on modularity, the system supports both real-time threat detection and batch prediction workflows, achieving high accuracy (98.5%) with low false positive rates. This makes it suitable for enterprise-scale cybersecurity operations where adaptability, precision, and performance are essential.


## 🔧 Tech Stack

* **Frontend**: HTML5, CSS3, JavaScript
* **Backend**: Python, FastAPI, Uvicorn
* **Database**: MongoDB Atlas (NoSQL)
* **ML Tools**: Scikit-learn, NumPy, Pandas, MLflow
* **DevOps**: Docker, Dagshub, GitHub
* **Dataset**: Phishing dataset with 30 engineered features


## 🧱 Project Architecture

**Network Devices → ETL Pipeline → MongoDB → ML Model → FastAPI Service → Security Alerts**

* **ETL**: Extracts and cleans logs from firewalls, IDS, and email gateways
* **ML**: Classifies threats using ensemble models (Random Forest + AdaBoost)
* **Deployment**: Dockerized services using FastAPI and tracked via MLflow


## ✨ Key Features

* 🔄 Real-Time ETL pipeline for ingestion and transformation
* 🤖 AI-based threat detection using ensemble ML models
* 🌐 FastAPI RESTful service for real-time predictions
* 📊 MLflow integration for experiment tracking and model monitoring
* 📁 Batch prediction pipeline for historical threat analysis
* ⚙️ Scalable architecture with Docker and cloud-ready components


## ⚙️ How It Works

1. **Ingest Logs**: Collect data from firewalls, IDS, and other network components
2. **Transform**: Preprocess and structure logs into analyzable format using Pandas
3. **Train Model**: Use historical data and optimize with grid/random search
4. **Deploy**: Serve the model using FastAPI and Uvicorn
5. **Monitor**: Evaluate model performance and update using MLflow tracking


## 🚀 Installation

Contact at ([work.ganeshpawar03@gmail.com](mailto:work.ganeshpawar03@gmail.com)) for installation and giude.


## 📈 Results

* **Accuracy**: 98.5%
* **Precision**: 96.2%
* **Recall**: 97.8%
* **False Positive Rate**: 1.4%
* **UI**: Chrome-based interface for user interaction
* **Monitoring**: Metrics tracked using Dagshub and MLflow


## 🤝 Connect

📬 **Share your story** ([work.ganeshpawar03@gmail.com](mailto:work.ganeshpawar03@gmail.com)) if you're using this repo for your mini or course project. I’d love to know how this project helped you!


You Can Connect with on: 

🔗 [LinkedIn](https://www.linkedin.com/in/ganesh-pawar143)
💻 [GitHub](https://github.com/ganesh-1433)
