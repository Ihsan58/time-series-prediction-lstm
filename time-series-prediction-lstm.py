import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    
    # Drop unnecessary columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Keep key features
    
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    # Feature engineering: Add moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    
    # Drop rows with NaN values (from rolling averages)
    data.dropna(inplace=True)
    
    # Scale numerical features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_features, columns=data.columns, index=data.index)
    
    return scaled_data

if __name__ == "__main__":
    processed_data = preprocess_data('data/eur_usd.csv')
    processed_data.to_csv('data/processed_data.csv')
    print("Processed data saved to data/processed_data.csv")
""",
            "train_model.py": """
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Prepare data for LSTM
def prepare_data(data, n_lags):
    X, y = [], []
    for i in range(len(data) - n_lags):
        X.append(data[i:i + n_lags])
        y.append(data[i + n_lags, 3])  # Predict "Close" price
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Load preprocessed data
    data = pd.read_csv('data/processed_data.csv', index_col=0).values
    
    # Prepare data
    n_lags = 30  # Use last 30 days for prediction
    X, y = prepare_data(data, n_lags)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    # Save model
    model.save('src/lstm_model.h5')
    print("Model saved to src/lstm_model.h5")
""",
            "api.py": """
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Load pre-trained model
model = load_model('src/lstm_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # JSON input
    features = np.array(data['features']).reshape(1, -1, len(data['features'][0]))
    prediction = model.predict(features).tolist()
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
    """
       }
    }
}