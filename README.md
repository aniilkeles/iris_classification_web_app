# 🌸 Iris Flower Classifier — Deep Learning Web Application

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

*A real-time Iris species classification web application powered by a PyTorch deep learning model and served through a FastAPI REST backend.*

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [What I Did](#-what-i-did)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Training Details](#-training-details)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Getting Started](#-getting-started)
- [Preview](#-preview)

---

## 🧩 About the Project

This project demonstrates an **end-to-end machine learning pipeline** — from training a neural network on the classic Iris dataset to deploying it as an interactive web application.

The Iris dataset contains 150 samples across 3 species of Iris flowers. Each sample has 4 features:

| Feature | Description |
|--------|-------------|
| Sepal Length | Length of the sepal (cm) |
| Sepal Width | Width of the sepal (cm) |
| Petal Length | Length of the petal (cm) |
| Petal Width | Width of the petal (cm) |

Based on these 4 measurements, the model predicts which of the 3 Iris species the flower belongs to — **Setosa**, **Versicolor**, or **Virginica** — along with confidence scores for all classes.

---

## ✅ What I Did

### 1. 🤖 Built & Trained a Neural Network with PyTorch
- Designed a 3-layer feedforward neural network (`IrisClassifier`) using `torch.nn.Sequential`
- Used **CrossEntropyLoss** as the loss function for multi-class classification
- Used the **Adam optimizer** with a learning rate of `0.01`
- Trained for **200 epochs** and evaluated on a held-out test set using `torchmetrics`
- Saved the trained model weights using `torch.save(model.state_dict(), "iris_classification_model.pth")`

### 2. 🚀 Built a REST API with FastAPI
- Created a `/predict` POST endpoint that accepts the 4 Iris features as JSON
- Loaded the trained `.pth` model weights at startup using `model.load_state_dict()`
- The endpoint returns the **predicted class**, **confidence score**, and **probabilities for all 3 classes**
- Served the HTML frontend via **Jinja2 templating** and static files via FastAPI's `StaticFiles` mount

### 3. 🎨 Designed a Modern, Responsive Frontend
- Built a two-panel layout (input form + result display) using pure HTML, CSS, and JavaScript
- Applied a **deep navy + teal/cyan aurora theme** with glassmorphism panel effects
- Added animated **probability bar charts** for all 3 classes
- Added **quick example buttons** that auto-fill form values for each species
- Used the **Fetch API** (async/await) to communicate with the FastAPI backend without page reloads

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Machine Learning | **PyTorch 2.x** | Model definition, training, inference |
| Metrics | **TorchMetrics** | Accuracy evaluation during training |
| Backend | **FastAPI** | REST API server |
| ASGI Server | **Uvicorn** | Running the FastAPI app |
| Templating | **Jinja2** | Serving the HTML frontend |
| Validation | **Pydantic** | Request body validation |
| Frontend | **HTML5 / CSS3 / JS** | UI, styling, async requests |
| Fonts | **Google Fonts (Inter)** | Typography |

---

## 🧠 Model Architecture

```
Input Layer  →  4 features (sepal_length, sepal_width, petal_length, petal_width)
                          │
                   Linear(4 → 16)
                        ReLU
                          │
                   Linear(16 → 16)
                        ReLU
                          │
                   Linear(16 → 3)
                          │
Output Layer  →  3 class logits  →  Softmax  →  Probabilities
```

The model uses `nn.Sequential` internally as `self.linear_layer_stack`.

**Parameter count:** ~500 trainable parameters — intentionally lightweight for this tabular dataset.

---

## 📊 Training Details

```python
epochs    = 200
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()
```

- Training loop includes both **train** and **eval** phases per epoch
- **MulticlassAccuracy** from `torchmetrics` used for performance monitoring
- Training progress logged every 20 epochs:

```
Epoch: 0   | Loss: 1.12 | Accuracy: 35.0 | Test Loss: 1.10 | Test Accuracy: 34.0
Epoch: 20  | Loss: 0.74 | Accuracy: 73.3 | Test Loss: 0.71 | Test Accuracy: 76.0
...
Epoch: 180 | Loss: 0.07 | Accuracy: 98.1 | Test Loss: 0.09 | Test Accuracy: 97.4
```

---

## 🗂️ Project Structure

```
IrisClassifierWebApplication/
│
├── main.py                        # FastAPI application (routes, model loading, inference)
├── iris_classification_model.pth  # Trained PyTorch model weights (state_dict)
├── requirements.txt               # Python package dependencies
├── README.md                      # This file
│
├── templates/
│   └── index.html                 # Main UI page (served via Jinja2)
│
└── static/
    ├── style.css                  # UI styles (dark aurora theme, glassmorphism)
    └── main.js                    # Async form handling & result rendering
```

---

## 🌐 API Reference

### `GET /`
Returns the main HTML page.

---

### `POST /predict`
Predicts the Iris species from flower measurements.

**Request Body:**
```json
{
  "sepal_length": 5.7,
  "sepal_width": 2.9,
  "petal_length": 4.2,
  "petal_width": 1.3
}
```

**Response:**
```json
{
  "class": "Iris-Versicolor",
  "class_index": 1,
  "confidence": 99.8,
  "all_probs": {
    "Iris-Setosa": 0.0,
    "Iris-Versicolor": 99.8,
    "Iris-Virginica": 0.2
  }
}
```

> 📖 Full interactive API documentation available at **`/docs`** (Swagger UI) after running the app.

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/IrisClassifierWebApplication.git
cd IrisClassifierWebApplication
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your Model Weights
Place your trained `iris_classification_model.pth` file in the **root directory** of the project.

### 5. Run the Application
```bash
python main.py
```
Or with auto-reload for development:
```bash
uvicorn main:app --reload
```

### 6. Open in Browser
```
http://localhost:8000
```

---

## 📸 Preview

> Add your app screenshot below:

![Iris Classifier App Preview](assets/iris_app_preview.png)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ using PyTorch & FastAPI
</div>
