from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load SVM Model
with open("SVM_model.pkl", "rb") as f:
    model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Student Performance Prediction</title>

<style>
*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Segoe UI',sans-serif;
}

body{
background:linear-gradient(135deg,#0f172a,#1e40af);
min-height:100vh;
display:flex;
justify-content:center;
align-items:center;
padding:20px;
}

.container{
width:100%;
max-width:1000px;
background:#fff;
padding:35px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

h1{
text-align:center;
color:#1e40af;
margin-bottom:10px;
}

p{
text-align:center;
color:#64748b;
margin-bottom:25px;
}

.form-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:15px;
}

input,select{
width:100%;
padding:12px;
border:1px solid #cbd5e1;
border-radius:10px;
font-size:15px;
}

button{
width:100%;
padding:14px;
margin-top:20px;
background:#2563eb;
color:white;
border:none;
border-radius:10px;
font-size:18px;
font-weight:bold;
cursor:pointer;
}

button:hover{
background:#1d4ed8;
}

.result{
margin-top:25px;
padding:18px;
text-align:center;
font-size:22px;
font-weight:bold;
background:#eff6ff;
border-radius:10px;
color:#1e40af;
}

.footer{
margin-top:20px;
text-align:center;
color:#64748b;
}

@media(max-width:768px){
.form-grid{
grid-template-columns:1fr;
}
}
</style>

</head>

<body>

<div class="container">

<h1>🎓 Student Performance Prediction</h1>

<p>Support Vector Machine (SVC) Machine Learning Model</p>

<form method="POST">

<div class="form-grid">

<input type="number" step="0.1" name="Student_Type"
placeholder="Student Type" required>

<input type="number" step="0.1" name="Sleep_Hours"
placeholder="Sleep Hours" required>

<input type="number" step="0.1" name="Study_Hours"
placeholder="Study Hours" required>

<input type="number" step="0.1" name="Social_Media_Hours"
placeholder="Social Media Hours" required>

<input type="number" step="0.1" name="Attendance"
placeholder="Attendance %" required>

<input type="number" step="0.1" name="Exam_Pressure"
placeholder="Exam Pressure" required>

<input type="number" step="0.1" name="Family_Support"
placeholder="Family Support" required>

<input type="number" step="0.1" name="Month"
placeholder="Month" required>

</div>

<button type="submit">Predict Performance</button>

</form>

{% if prediction %}
<div class="result">
{{ prediction }}
</div>
{% endif %}

<div class="footer">
Developed by Pranita | Data Analyst
</div>

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        features = np.array([[
            float(request.form["Student_Type"]),
            float(request.form["Sleep_Hours"]),
            float(request.form["Study_Hours"]),
            float(request.form["Social_Media_Hours"]),
            float(request.form["Attendance"]),
            float(request.form["Exam_Pressure"]),
            float(request.form["Family_Support"]),
            float(request.form["Month"])
        ]])

        pred = model.predict(features)[0]

        if pred == 1:
            prediction = "✅ High Performance Predicted"
        else:
            prediction = "⚠️ Low Performance Predicted"

    return render_template_string(
        HTML,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
