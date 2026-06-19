import joblib
import torch
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)

enc_cause = LabelEncoder().fit([
    "Accident", "Breakdown", "Flood", "Fire", "Roadwork",
    "Signal Failure", "VIP Movement", "Unknown"
])
enc_veh = LabelEncoder().fit([
    "Car", "Truck", "Bus", "Bike", "Auto", "Heavy Vehicle", "Unknown"
])

sample_coords = np.array([
    [12.8000, 77.4500], [13.2000, 77.8000],
    [12.9716, 77.5946], [13.0000, 77.6000],
    [12.8500, 77.5000], [13.1000, 77.7500]
])
scaler = StandardScaler().fit(sample_coords)

embedding_sizes = [
    (len(enc_cause.classes_), 8),
    (len(enc_veh.classes_), 6),
]

preprocessors = {
    "encoders": {
        "event_cause": enc_cause,
        "veh_type": enc_veh,
    },
    "scaler": scaler,
    "embedding_sizes": embedding_sizes,
}

joblib.dump(preprocessors, "preprocessors.pkl")
print("preprocessors.pkl saved")

sys.path.insert(0, ".")
from main import V8TitanNet

model = V8TitanNet(embedding_sizes, num_time=4, num_bool=1)
torch.save(model.state_dict(), "v8_dl_model.pth")
print("v8_dl_model.pth saved")
print("Done. Run: uvicorn main:app --reload")
