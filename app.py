from models import get_model, set_memory_limit, preprocess, get_class, data_uri_to_image # from ./models/__init__.py
from flask_cors import CORS
from flask import Flask, request
import pickle, numpy as np


MEMORY_LIMIT = 1024 * 1
app = Flask(__name__)
CORS(app)

set_memory_limit(1)
disease_model = get_model("./models/crop_disease-VGG19-10_epoch")
crop_model = get_model("./models/crop_predictor-100")
price_model = None
encoders = None

with open("./models/price_predictor.pkl","rb") as f:
    price_model = pickle.load(f)
with open("./models/label_encoders.pkl","rb") as f:
    encoders = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the home page of the server where the magic of Machine Learning model happens"

@app.route("/api/disease", methods=["POST"])
def disease():

    data = request.get_json()

    if data and "image" in data:
        image = data_uri_to_image(data["image"])
        image = preprocess(image)
        prediction = disease_model.predict(image)
        prediction = prediction.tolist()
        label = get_class(prediction)

        return {
            "predictions": prediction,
            "label": label,
        }

    else:
        return {"error": "No image found"}

@app.route("/api/recommend", methods=["POST"])
def recommandation():

    data = request.get_json()

    # temperature	humidity	ph	rainfall	State	District	Market	Grade	Variety	N	P	K

    if data:
        input_data = [
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"],
            encoders["State"].transform([data["State"]])[0],
            encoders["District"].transform([data["District"]])[0],
            encoders["Market"].transform([data["Market"]])[0],
            encoders["Grade"].transform([data["Grade"]])[0],
            encoders["Variety"].transform([data["Variety"]])[0],
            data["N"],
            data["P"],
            data["K"],
        ]

        # print("State : ",encoders["State"].transform([data["State"]]))

        input_array = np.array(input_data,dtype=np.float32).reshape(1, -1)
        print(encoders["label"].transform(["rice"])[0])
        print(encoders["label"].inverse_transform([17])[0])
        prediction = crop_model.predict(input_array)
        prediction = prediction.tolist()
        label_no = np.argmax(prediction)
        print(prediction)
        label_name = encoders["label"].inverse_transform([label_no])[0]

        price_input = input_data
        price_input.append(label_no)
        price_input = np.array(price_input,dtype=np.float32).reshape(1, -1)
        price = price_model.predict(price_input)

        price = price.tolist()[0]

        return {
            "data" : np.array(input_data).tolist(),
            "label": label_name,
            "price": price,
            "raw_prediction": {
                "crop" : prediction,
                "price" : price
            }
        }

    else:
        return {"error": "No data found"}

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)