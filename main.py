from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import torch.nn as nn
import os

app = FastAPI(title="Iris Classifier")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# --------- KULLANICI İÇİN NOT ---------
# Kendi model sınıfınızı ('IrisClassifier') buraya yapıştırın veya import edin.
# Modelinizin init metodundaki özelliklerin eğitim anındakiyle aynı olduğundan emin olun.
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(4, 16),   # layer index 0
            nn.ReLU(),          # layer index 1
            nn.Linear(16, 16),  # layer index 2
            nn.ReLU(),          # layer index 3
            nn.Linear(16, 3)    # layer index 4
        )
        
    def forward(self, x):
        return self.linear_layer_stack(x)

# Sınıf İsimleri
class_names = {
    0: 'Iris-Setosa',
    1: 'Iris-Versicolor',
    2: 'Iris-Virginica'
}

model = None

# Model yükleme fonksiyonu
def load_model():
    global model
    model_path = "iris_classification_model.pth" # Eğitilmiş modelinizin yolunu buraya yazın
    model = IrisClassifier()
    try:
        # Pytorch model state_dict'ini yükleme
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print("Model başarıyla yüklendi!")
        else:
            print(f"Uyarı: '{model_path}' bulunamadı. Lütfen modelinizi ana dizine ekleyin. Şimdilik rastgele ağırlıklarla çalışıyor.")
            model.eval()
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")

# Uygulama başlarken modeli yükle
load_model()

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(features: Features):
    if model is None:
        return {"error": "Model yüklenemedi."}
    
    # Girdi verilerini tensor'a dönüştürme
    input_data = torch.tensor([[
        features.sepal_length, 
        features.sepal_width, 
        features.petal_length, 
        features.petal_width
    ]], dtype=torch.float32)
    
    with torch.inference_mode():
        logits = model(input_data)
        probs = torch.softmax(logits, dim=1)
        pred_index = probs.argmax(dim=1).item()
        confidence = probs[0][pred_index].item() * 100

    all_probs = {
        class_names[i]: round(probs[0][i].item() * 100, 1)
        for i in range(3)
    }

    return {
        "class": class_names[pred_index],
        "class_index": pred_index,
        "confidence": round(confidence, 1),
        "all_probs": all_probs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
