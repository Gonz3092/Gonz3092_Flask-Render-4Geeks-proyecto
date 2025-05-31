from flask import Flask, request, render_template
from pickle import load

app = Flask(__name__)
model = load(open("/workspaces/Gonz3092_Flask-Render-4Geeks-proyecto/models/tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav", "rb"))
class_dict = {
    "0": "Resultado Negativo",
    "1": "Resultado Positivo"
}

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        # Obtain values from form
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        val5 = float(request.form["val5"])
        val6 = float(request.form["val6"])
        val7 = float(request.form["val7"])
        
        data = [[val1, val2, val3, val4, val5, val6, val7]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)