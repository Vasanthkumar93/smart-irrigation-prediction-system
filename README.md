# Smart Irrigation Prediction System Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-brightgreen)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-blue)
![SQLite](https://img.shields.io/badge/Database-SQLite3-green)

An end-to-end Machine Learning web application designed to optimize agricultural water management. The system analyzes critical soil and atmospheric parameters—**Soil Moisture**, **Ambient Temperature**, **Relative Humidity**, and **Recent Rainfall**—to predict whether irrigation is needed, providing actionable guidance to prevent both under-irrigation and water wastage.

Developed as a college final-year project in **Artificial Intelligence & Data Science / Computer Science Engineering**.

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Architecture & Folder Structure](#project-architecture--folder-structure)
3. [Technologies Used](#technologies-used)
4. [Dataset & Agro-Climatic Logic](#dataset--agro-climatic-logic)
5. [Machine Learning Algorithms & Auto-Selection](#machine-learning-algorithms--auto-selection)
6. [Step-by-Step Installation & Execution](#step-by-step-installation--execution)
7. [Sample Test Cases](#sample-test-cases)
8. [REST API Documentation](#rest-api-documentation)
9. [Viva & Project Presentation Q&A](#viva--project-presentation-qa)

---

## 1. Project Overview

Agriculture accounts for over 70% of freshwater consumption globally. Traditional manual irrigation methods rely on estimation or fixed schedules, leading to over-irrigation (soil nutrient leaching, root rot, high power bills) or under-irrigation (crop wilting, stunted yield).

This web application offers:
- **Real-Time Prediction:** Instant classification (Yes/No) based on real-time soil & weather inputs.
- **Algorithm Benchmarking:** Compares **Decision Tree**, **Random Forest**, and **Logistic Regression** on Accuracy, Precision, Recall, and F1-Score.
- **Champion Auto-Selection:** Automatically selects and deploys the best-performing model based on F1-score.
- **Actionable Guidance:** Recommends precise water dosage (mm), optimal timing (dawn/dusk), and irrigation techniques (e.g. drip irrigation).
- **Persistent Logging:** SQLite-backed audit log with CSV export capability.
- **REST API:** Fully functional JSON endpoint (`/api/predict`) ready for IoT microcontrollers (ESP32/Raspberry Pi).

---

## 2. Project Architecture & Folder Structure

```
smart-irrigation/
│
├── app.py                      # Flask web application server, routes, and DB logic
├── train_model.py              # ML training script, evaluation & visual plotting
├── requirements.txt            # Python package dependencies
├── smart_irrigation.db         # Auto-generated SQLite database for prediction history
├── README.md                   # Complete documentation and setup manual
│
├── dataset/
│   ├── irrigation_data.csv     # Curated agricultural dataset (moisture, temp, humidity, rain)
│   └── generate_dataset.py     # Agro-climatic data generation utility (Penman-Monteith logic)
│
├── models/
│   ├── best_model.pkl          # Serialized champion ML model (e.g., Random Forest)
│   ├── scaler.pkl              # Fitted StandardScaler for input normalization
│   └── model_metrics.json      # Comparison scores (Accuracy, Precision, Recall, F1)
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern agricultural green & white UI stylesheet
│   ├── js/
│   │   └── main.js             # Slider-input sync, scenario presets, CSV exporter
│   └── images/
│       ├── model_comparison.png# Bar chart comparing the 3 algorithms
│       ├── feature_importance.png # Importance weights of soil & weather features
│       └── confusion_matrix.png# Confusion matrix heatmap of the champion model
│
└── templates/
    ├── base.html               # Master layout with responsive navbar & footer
    ├── index.html              # Main dashboard overview & quick stats
    ├── predict.html            # Interactive prediction form, sliders, & advisory card
    ├── models.html             # Model scorecard, charts & academic metrics breakdown
    ├── history.html            # SQLite historical prediction log with CSV download
    └── about.html              # Theoretical agronomic background & methodology
```

---

## 3. Technologies Used

- **Programming Language:** Python 3.9+ / 3.10+ / 3.11+
- **Web Framework:** Flask
- **Machine Learning Library:** Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Database:** SQLite3
- **Frontend:** Semantic HTML5, CSS3 (Custom Responsive Agricultural Theme), Modern JavaScript ES6

---

## 4. Dataset & Agro-Climatic Logic

The dataset incorporates real-world agronomic dynamics inspired by the **FAO-56 Penman-Monteith Evapotranspiration Model**:

1. **Soil Moisture (%):** Represents volumetric water content. Between Field Capacity (FC) and the Permanent Wilting Point (PWP), plants absorb water comfortably. When soil moisture drops below critical management thresholds (typically $< 28\%$), irrigation is triggered.
2. **Temperature (°C):** High ambient temperatures elevate vapor pressure deficit (VPD) and intensify plant transpiration.
3. **Relative Humidity (%):** Low humidity accelerates soil evaporation, whereas high humidity ($\ge 75\%$) slows water loss.
4. **Recent Rainfall (mm):** Significant precipitation ($\ge 10\text{ mm}$) saturates the upper soil crust, suppressing the need for supplemental irrigation.

---

## 5. Machine Learning Algorithms & Auto-Selection

Three diverse classification algorithms are implemented and evaluated simultaneously on an 80/20 stratified split:

| Algorithm | Strengths | Handling in System |
| :--- | :--- | :--- |
| **Decision Tree** | Highly interpretable, creates intuitive threshold rules. | Trained directly on physical sensor features. |
| **Random Forest** | Ensemble of 100 decorrelated trees; minimizes variance and resists overfitting. | Consistently achieves top F1-Score & Accuracy. |
| **Logistic Regression** | Linear probabilistic baseline; fast inference. | Scaled using `StandardScaler` to normalize dimensions. |

The script evaluates each model using:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The algorithm with the highest **F1-Score** is automatically persisted as `models/best_model.pkl` and deployed inside `app.py`.

---

## 6. Step-by-Step Installation & Execution

### Prerequisites
- Python 3.8 or higher installed on your computer.
- Visual Studio Code (or any code editor).

### Step 1: Open the Project in VS Code
Open VS Code, select **File > Open Folder...**, and select the folder:
```
C:\Users\user\.gemini\antigravity\scratch\smart-irrigation
```

### Step 2: Open Terminal in VS Code
Press ``Ctrl + ` `` (Backtick) or go to **Terminal > New Terminal**.

### Step 3: Create and Activate Virtual Environment (Recommended)
In the terminal, run:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy warning, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use Command Prompt: `venv\Scripts\activate.bat`)*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
Run:
```bash
pip install -r requirements.txt
```

### Step 5: Train the Machine Learning Models
Execute the training pipeline:
```bash
python train_model.py
```
**Expected Output in Terminal:**
- Displays the loaded dataset records.
- Outputs the comparison metrics table for Decision Tree, Random Forest, and Logistic Regression.
- Announces the champion model (e.g. `Random Forest - 98.7% F1-score`).
- Generates and saves visual charts into `static/images/` and model weights into `models/`.

### Step 6: Launch the Flask Web Application
Run:
```bash
python app.py
```
You will see:
```text
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

### Step 7: Open in Your Browser
Open Chrome, Edge, or Firefox and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. Sample Test Cases

You can test these realistic scenarios either manually via the web form or by clicking the preset buttons:

| Scenario | Soil Moisture | Temperature | Humidity | Rainfall | Expected Prediction | Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Extreme Drought** | `11.5%` | `42.0°C` | `18.0%` | `0.0 mm` | **Irrigation Required (Yes)** | Critical root zone depletion + severe heat stress. |
| **2. Dry Summer Day** | `18.2%` | `36.5°C` | `30.0%` | `0.0 mm` | **Irrigation Required (Yes)** | Soil moisture is below threshold; high daytime evaporation. |
| **3. Moderate Spring** | `42.0%` | `27.5°C` | `62.0%` | `1.5 mm` | **Irrigation Not Needed (No)** | Soil moisture is well within comfortable field capacity. |
| **4. Monsoon Rainy** | `65.0%` | `22.0°C` | `85.0%` | `24.0 mm` | **Irrigation Not Needed (No)** | Heavy rainfall observed; soil saturated; prevent waterlogging. |

---

## 8. REST API Documentation

The system includes a dedicated REST API endpoint for integration with mobile apps or IoT sensor boards.

### Endpoint: `POST /api/predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "soil_moisture": 14.5,
  "temperature": 38.0,
  "humidity": 26.0,
  "rainfall": 0.0
}
```

- **Successful Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "prediction": 1,
    "prediction_label": "Irrigation Required (Yes)",
    "confidence": 98.2,
    "model_used": "Random Forest",
    "inputs": {
      "soil_moisture": 14.5,
      "temperature": 38.0,
      "humidity": 26.0,
      "rainfall": 0.0
    },
    "guidance": {
      "action": "Irrigate Now",
      "recommended_water": "25 - 35 mm",
      "best_timing": "Early Morning (5:00 AM - 7:30 AM) or Late Evening (6:00 PM - 8:00 PM)",
      "recommended_method": "Drip irrigation recommended to conserve water...",
      "weather_note": "Zero precipitation recorded. Ensure consistent soil hydration..."
    }
  }
}
```

---

## 9. Viva & Project Presentation Q&A

### Q1: Why did you choose Random Forest over Logistic Regression for this application?
> **Answer:** Logistic Regression is a linear classifier that assumes a linear decision boundary. However, agricultural moisture dynamics are non-linear; for example, high rainfall overrides low soil moisture, and high temperatures drastically lower the moisture threshold. Random Forest naturally captures these complex, non-linear feature interactions and feature thresholds without requiring manual polynomial feature engineering.

### Q2: What is the significance of the F1-Score in smart irrigation?
> **Answer:** In irrigation, both False Positives (irrigating when unnecessary) and False Negatives (failing to irrigate when the crop is desiccating) carry penalties. False Negatives result in crop death or severe yield reduction, whereas False Positives cause water and power wastage. The F1-score provides the harmonic mean of Precision and Recall, ensuring the selected model does not sacrifice crop safety for water conservation or vice-versa.

### Q3: Why is hardware (sensors/Arduino) not mandatory for this project?
> **Answer:** The primary intellectual contribution of this project is the **Machine Learning model, data engineering pipeline, decision advisory logic, and full-stack software architecture**. In modern precision agriculture, edge microcontrollers simply act as REST API clients transmitting telemetry. Our system provides both a human web interface and a standard JSON REST API (`/api/predict`), proving complete end-to-end functionality ready to pair with physical sensors whenever deployed to the field.

### Q4: How is data persisted and audited?
> **Answer:** The application uses an embedded **SQLite** database (`smart_irrigation.db`). Every prediction performed via the web interface is stored with a timestamp, raw environmental parameters, prediction label, confidence level, and actionable guidance, allowing farmers to export audit reports to CSV.
