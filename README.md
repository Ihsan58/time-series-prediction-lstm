LSTM Time Series Prediction Pipeline
An end-to-end pipeline for time series prediction using LSTM (Long Short-Term Memory) neural networks. This project includes data preprocessing, model training, and API deployment to predict future values from sequential data.

🚀 Features
Data Preprocessing:
Clean and preprocess raw time-series data.
Add moving averages (5-day and 20-day) for feature engineering.
Scale features using Min-Max Scaling for compatibility with LSTM.
LSTM Model Training:
Build and train an LSTM model with TensorFlow/Keras.
Use lagged sequences to predict future values.
API for Predictions:
Flask-based REST API for real-time predictions.
Accepts JSON input and returns model predictions.
📂 Project Structure
graphql
Copy code
.
├── data/
│   ├── eur_usd.csv                # Raw dataset
│   ├── processed_data.csv         # Preprocessed dataset
├── src/
│   ├── preprocess_data.py         # Data preprocessing script
│   ├── train_model.py             # Model training script
│   ├── lstm_model.h5              # Trained LSTM model
│   ├── api.py                     # Flask API for predictions
└── README.md                      # Project documentation
📋 Prerequisites
Before running the project, ensure you have the following installed:

Python 3.8+
Libraries:
Copy code
pandas
numpy
scikit-learn
tensorflow
flask
Install all dependencies using:

bash
Copy code
pip install -r requirements.txt
⚙️ Usage
1. Data Preprocessing
Prepare the raw dataset for training:

bash
Copy code
python preprocess_data.py
Input: data/eur_usd.csv (raw dataset).
Output: data/processed_data.csv (cleaned and scaled dataset).
2. Model Training
Train the LSTM model using preprocessed data:

bash
Copy code
python train_model.py
Input: data/processed_data.csv.
Output: src/lstm_model.h5 (trained model).
3. Running the API
Start the Flask API server to serve predictions:

bash
Copy code
python api.py
API URL: http://127.0.0.1:5000/.
4. Making Predictions
Send a POST request to the /predict endpoint with JSON input.
Example Input:

json
Copy code
{
  "features": [
    [0.5, 0.6, 0.7, 0.8, 0.9],
    [0.6, 0.7, 0.8, 0.9, 1.0]
  ]
}
Using curl:

bash
Copy code
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [[0.5, 0.6, 0.7, 0.8, 0.9], [0.6, 0.7, 0.8, 0.9, 1.0]]}'
Example Response:

json
Copy code
{
  "prediction": [[1.23456], [1.34567]]
}
🛠️ Technologies Used
Python: Core programming language.
Pandas & NumPy: Data manipulation and feature engineering.
Scikit-Learn: Data scaling and preprocessing.
TensorFlow/Keras: LSTM model architecture and training.
Flask: API deployment.
🎯 Future Enhancements
Add early stopping and validation during model training.
Experiment with other architectures like GRU or Transformer.
Deploy the API to a cloud platform (e.g., AWS, Heroku, or GCP).
Integrate Docker for containerized deployment.
📧 Contact
Ihsan Ul Haq
Email: uihsan458@gmail.com
GitHub: github.com/ihsan58

📜 License
This project is licensed under the MIT License. See the LICENSE file for details
