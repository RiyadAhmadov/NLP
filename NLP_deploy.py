import uvicorn
from fastapi import FastAPI, Query
import pickle
import pandas as pd
import warnings
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Create a FastAPI app instance
app = FastAPI()

# Load the pickled model using a relative file path
with open('nlp.pkl', "rb") as model_file:
    model = pickle.load(model_file)

# Load TF-IDF vectorizer
with open('tfidfVectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Opinion Sentiment Prediction</title>
    <style>
        body {
            background-image: url('https://assets.aboutamazon.com/dims4/default/b073721/2147483647/strip/true/crop/2357x1179+74+690/resize/1200x600!/quality/90/?url=https%3A%2F%2Famazon-blogs-brightspot.s3.amazonaws.com%2Fab%2F7d%2F7387c5c34035af9dafce465fe433%2Famazon-org-smile-extruded-sq.jpg');
            background-size: cover;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .form-container {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 10px;
            width: 80%;
            max-width: 500px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        label {
            font-size: 18px;
            color: #333;
            display: block;
            margin-bottom: 8px;
            text-align: left;
        }

        input {
            font-size: 16px;
            padding: 10px;
            margin-bottom: 20px;
            width: calc(100% - 20px);
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 5px;
            transition: border-color 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: #007BFF;
        }

        input[type="submit"] {
            background-color: #007BFF;
            color: #fff;
            font-size: 18px;
            padding: 12px 20px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #0056b3;
        }

        h1 {
            font-size: 32px;
            color: #333;
            margin-bottom: 30px;
        }

        h2 {
            font-size: 24px;
            color: #333;
            margin-bottom: 20px;
        }

        p {
            font-size: 18px;
            color: #333;
            margin-bottom: 20px;
        }

        img {
            width: 200px;
            height: auto;
            margin-bottom: 20px;
        }

        .prediction-result {
            font-weight: bold;
            font-size: 20px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/2560px-Amazon_logo.svg.png" alt="Logo">
        <h1>Opinion Sentiment Prediction</h1>
        <form id="prediction-form">
            <label for="name">What is your Name?</label>
            <input type="text" id="name" name="name" required>

            <label for="surname">What is your Surname?</label>
            <input type="text" id="surname" name="surname" required>

            <label for="opinion">Describe your opinion.</label>
            <input type="text" id="opinion" name="opinion" required>

            <input type="submit" value="Predict Sentiment">
        </form>
        <h2>Sentiment Result:</h2>
        <p id="prediction_result" class="prediction-result"></p>
    </div>

    <script>
        const form = document.getElementById('prediction-form');
        const predictionResult = document.getElementById('prediction_result');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const formData = new FormData(form);

            const response = await fetch('/predict/?' + new URLSearchParams(formData).toString());
            const data = await response.json();

            predictionResult.textContent = data['prediction'];
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content=html_content)


@app.get("/predict/")
async def predict(
    name: str = Query(..., description="Name"),
    surname: str = Query(..., description="Surname"),
    opinion: str = Query(..., description="Opinion")
):
    # Vectorize the input opinion
    opinion_vectorized = tfidf_vectorizer.transform([opinion])

    # Make predictions using the pre-trained model
    price_prediction = model.predict(opinion_vectorized)

    price_prediction
    

    # You can use these parameters as input for your model and return the prediction result
    prediction_result = f"Opinion Sentiment Prediction: {'Your opinion is positive !!!' if price_prediction[0] == 1 else 'Your opinion is Negative !!!'}"
    return {"prediction": prediction_result}



# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(
        app,
        host="192.168.1.15", # You must change ip with your pc or device ip.
        port=5002,
        log_level="debug",
    )

# lET'S do run code for see project how to work