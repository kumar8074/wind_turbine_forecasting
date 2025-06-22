import os
from flask import Flask, render_template, request, send_from_directory
import subprocess
import socket

from src.pipeline.predict_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainingPipeline

app = Flask(__name__)

def is_port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0

def start_mlflow_server():
    mlflow_port = 5000  # or whatever port you're using for MLflow
    if not is_port_open(mlflow_port):
        print(f"Starting MLflow server on port {mlflow_port}...")
        subprocess.Popen(
            ["mlflow", "server", "--host", "0.0.0.0", "--port", str(mlflow_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        print("MLflow server already running.")


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_result = None

    if request.method == "POST":
        try:
            user_input = {
                'LV ActivePower (kW)': float(request.form['active_power']),
                'Wind Speed (m/s)': float(request.form['wind_speed']),
                'Theoretical_Power_Curve (KWh)': float(request.form['theoretical_power']),
                'Wind Direction (°)': float(request.form['wind_direction'])
            }

            pipeline = PredictionPipeline()
            prediction_result = pipeline.predict(user_input)

        except Exception as e:
            prediction_result = f"Error occurred during prediction: {str(e)}"

    return render_template("index.html", prediction=prediction_result)

@app.route("/train", methods=["GET", "POST"])
def train():
    training_done = False
    error_message = None

    if request.method == "POST":
        try:
            start_mlflow_server()  # 👈 Ensure MLflow is up
            train_pipeline= TrainingPipeline()
            train_pipeline.train()
            training_done = True
        except Exception as e:
            error_message = f"Error occurred during training: {str(e)}"

    plots = []
    if training_done:
        viz_path = "artifacts/visualizations"
        plots = [
            os.path.join(viz_path, "training_curves.png"),
            os.path.join(viz_path, "predictions_and_truth.png"),
            os.path.join(viz_path, "pred_vs_true_over_time.png")
        ]

    return render_template("train.html", training_done=training_done, error=error_message, plots=plots)

# Serve static files (plots) from artifacts
@app.route('/artifacts/visualizations/<filename>')
def serve_visualization(filename):
    return send_from_directory('artifacts/visualizations', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)


