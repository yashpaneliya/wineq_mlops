from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
templates_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=templates_dir)

def read_params(config_path):
    with open(config_path, "r") as yf:
        config = yaml.safe_load(yf)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir = config["webapp_model_dir"]
    model = joblib.load(model_dir)
    pred = model.predict(data)
    print(pred)
    return pred[0]

def api_response(request):
    try:
        data = np.array([list(request.json.values())])
        res = predict(data)
        res = {"response":res}
        return res
    except Exception as e:
        print(e)
        error = {"error":"Something went wrong!"}
        return error

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if request.form:
                data = dict(request.form).values()
                data = [list(map(float, data))]
                response = predict(data)
                return render_template("index.html", response=response)
            elif request.json:
                response = api_response(request)
                return jsonify(response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!"}
            return render_template("404.html", error=error)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
