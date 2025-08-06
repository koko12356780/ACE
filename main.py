import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import torch
import pickle
from model import AceAssistantModel
from tokenizer import AceTokenizer

# ✅ Use real API key (from .env or hardcoded fallback)
API_KEY = os.getenv("API_KEY", "RszsawokRu794V9EhmjnfgrkoT7oAxpHgvJ9_xzY8Lk")

# ✅ Load tokenizer
with open("ace_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load model
model = AceAssistantModel(vocab_size=30522)
model.load_state_dict(torch.load("ace_assistant_model.pt", map_location=torch.device("cpu")))
model.eval()

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Input format
class InputText(BaseModel):
    text: str

# ✅ Secure predict endpoint
@app.post("/predict")
def predict(input_data: InputText, x_api_key: str = Header(...)):
    # Verify API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Encode input
    input_ids = tokenizer.encode(input_data.text)
    input_tensor = torch.tensor([input_ids])

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
        decoded_output = tokenizer.decode(predicted_ids)

    return {
        "input": input_data.text,
        "predicted_ids": predicted_ids,
        "decoded_output": decoded_output
    }

# ✅ Local run entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
