from flask import Flask, jsonify, request
import pickle
import numpy as np

app = Flask(__name__)
with open('./xgboost.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/getaqi', methods=['POST'])
def create_book():
    json_data = request.json
    if json_data and 'data' in json_data:
        data_list = json_data['data']

    # data_list = [16.64, 49.97, 4.05, 29.26, 18.8, 10.03, 0.52, 9.84, 28.3, 0.0, 0.0, 0.0]
    latest_data = np.array(data_list)
    latest_data = latest_data.reshape(1, -1)

    prediction=float(model.predict(latest_data)[0])
    response_data = {'aqi': prediction}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

