from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_result = None

    if request.method == "POST":
        try:
            # Extract values from the form
            user_input = {
                'LV ActivePower (kW)': float(request.form['active_power']),
                'Wind Speed (m/s)': float(request.form['wind_speed']),
                'Theoretical_Power_Curve (KWh)': float(request.form['theoretical_power']),
                'Wind Direction (°)': float(request.form['wind_direction'])
            }

            # Instantiate prediction pipeline
            pipeline = PredictionPipeline()
            prediction_result = pipeline.predict(user_input)

        except Exception as e:
            prediction_result = f"Error occurred during prediction: {str(e)}"

    return render_template("index.html", prediction=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)

