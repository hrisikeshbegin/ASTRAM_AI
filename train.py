import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import joblib

DATA_PATH = "astram_event_data.csv"
EPOCHS = 25
BATCH_SIZE = 64
LR = 0.001

print("Loading data...")
df = pd.read_csv(DATA_PATH)

df = df[df['latitude'] > 0]
df = df[df['longitude'] > 0]
df = df[df['priority'].notna()]
df['description'] = df['description'].fillna("no description")
df['event_cause'] = df['event_cause'].fillna("others")
df['veh_type'] = df['veh_type'].fillna("others")

df['priority_label'] = (df['priority'] == 'High').astype(float)
df['closure_label'] = (df['requires_road_closure'] == True).astype(float)

severity_map = {'High': 80.0, 'Low': 30.0}
df['severity_target'] = df['priority'].map(severity_map).fillna(50.0)

print(f"Training rows: {len(df)}")
print(f"High priority: {int(df['priority_label'].sum())}, Low: {int((1 - df['priority_label']).sum())}")
print(f"Road closures: {int(df['closure_label'].sum())}")

print("Encoding features...")
enc_cause = LabelEncoder().fit(df['event_cause'].astype(str))
enc_veh = LabelEncoder().fit(df['veh_type'].astype(str))

df['cause_enc'] = enc_cause.transform(df['event_cause'].astype(str))
df['veh_enc'] = enc_veh.transform(df['veh_type'].astype(str))

scaler = StandardScaler().fit(df[['latitude', 'longitude']])
spatial_scaled = scaler.transform(df[['latitude', 'longitude']])

print("Loading sentence transformer and encoding descriptions...")
nlp = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = df['description'].astype(str).tolist()
text_embeddings = nlp.encode(descriptions, show_progress_bar=True, batch_size=128)

df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
df['hour'] = df['start_datetime'].dt.hour.fillna(12)
df['weekday'] = df['start_datetime'].dt.weekday.fillna(0)

time_feats = np.stack([
    np.sin(2 * np.pi * df['hour'] / 24),
    np.cos(2 * np.pi * df['hour'] / 24),
    np.sin(2 * np.pi * df['weekday'] / 7),
    np.cos(2 * np.pi * df['weekday'] / 7)
], axis=1)

bool_feats = (df['weekday'] >= 5).astype(float).values.reshape(-1, 1)

cat_feats = df[['cause_enc', 'veh_enc']].values

embedding_sizes = [
    (len(enc_cause.classes_), 8),
    (len(enc_veh.classes_), 6),
]


class FourierSpatialEncoding(nn.Module):
    def __init__(self, input_dim=2, num_frequencies=16):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, num_frequencies) * 10, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GatedResidualNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        gate, out = self.fc2(h).chunk(2, dim=-1)
        return self.norm((out * torch.sigmoid(gate)) + x)


class V8TitanNet(nn.Module):
    def __init__(self, embedding_sizes, num_time, num_bool, nlp_dim=384, fourier_freqs=16, hidden_dim=128):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(nc, ed) for nc, ed in embedding_sizes])
        n_emb = sum(ed for _, ed in embedding_sizes)
        self.spatial_encoder = FourierSpatialEncoding(input_dim=2, num_frequencies=fourier_freqs)
        n_spatial = fourier_freqs * 2
        self.tabular_proj = nn.Linear(n_emb + n_spatial + num_time + num_bool, hidden_dim)
        self.nlp_proj = nn.Linear(nlp_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.deep = nn.Sequential(
            GatedResidualNetwork(hidden_dim),
            nn.Dropout(0.3),
            GatedResidualNetwork(hidden_dim)
        )
        self.severity_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.duration_mu = nn.Linear(hidden_dim, 1)
        self.duration_sigma = nn.Linear(hidden_dim, 1)
        self.closure_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_cat, x_spatial, x_time, x_bool, x_text):
        x_emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, 1)
        x_spat_encoded = self.spatial_encoder(x_spatial)
        x_tab_all = torch.cat([x_emb, x_spat_encoded, x_time, x_bool], 1)
        x_tab = F.gelu(self.tabular_proj(x_tab_all))
        x_txt = F.gelu(self.nlp_proj(x_text))
        x_stacked = torch.stack([x_tab, x_txt], dim=1)
        attn_out, _ = self.attention(x_stacked, x_stacked, x_stacked)
        x_fused = attn_out.mean(dim=1)
        x_out = self.deep(x_fused)
        severity = torch.sigmoid(self.severity_head(x_out)) * 100
        mu = self.duration_mu(x_out)
        sigma = F.softplus(self.duration_sigma(x_out)) + 1e-6
        closure_logits = self.closure_head(x_out)
        return severity, mu, sigma, closure_logits


class AstramDataset(Dataset):
    def __init__(self, idx):
        self.cat = torch.tensor(cat_feats[idx], dtype=torch.long)
        self.spatial = torch.tensor(spatial_scaled[idx], dtype=torch.float32)
        self.time = torch.tensor(time_feats[idx], dtype=torch.float32)
        self.bool = torch.tensor(bool_feats[idx], dtype=torch.float32)
        self.text = torch.tensor(text_embeddings[idx], dtype=torch.float32)
        self.severity = torch.tensor(df['severity_target'].values[idx], dtype=torch.float32)
        self.closure = torch.tensor(df['closure_label'].values[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, i):
        return (self.cat[i], self.spatial[i], self.time[i], self.bool[i], self.text[i],
                self.severity[i], self.closure[i])


all_idx = np.arange(len(df))
train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42)

train_loader = DataLoader(AstramDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(AstramDataset(val_idx), batch_size=BATCH_SIZE)

model = V8TitanNet(embedding_sizes, num_time=4, num_bool=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

pos_weight = torch.tensor([(df['closure_label'] == 0).sum() / max(1, (df['closure_label'] == 1).sum())], dtype=torch.float32)
closure_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
severity_loss_fn = nn.MSELoss()

print("Training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for cat, spatial, time, bool_f, text, sev, clos in train_loader:
        optimizer.zero_grad()
        sev_pred, mu, sigma, clos_logits = model(cat, spatial, time, bool_f, text)
        loss_sev = severity_loss_fn(sev_pred.squeeze(), sev)
        loss_clos = closure_loss_fn(clos_logits.squeeze(), clos)
        loss = loss_sev * 0.01 + loss_clos
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct_clos = 0
    total = 0
    with torch.no_grad():
        for cat, spatial, time, bool_f, text, sev, clos in val_loader:
            sev_pred, mu, sigma, clos_logits = model(cat, spatial, time, bool_f, text)
            clos_pred = (torch.sigmoid(clos_logits.squeeze()) > 0.5).float()
            correct_clos += (clos_pred == clos).sum().item()
            total += len(clos)
    acc = correct_clos / total
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss/len(train_loader):.4f}  val_closure_acc={acc:.3f}")

print("Saving model and preprocessors...")
preprocessors = {
    "encoders": {"event_cause": enc_cause, "veh_type": enc_veh},
    "scaler": scaler,
    "embedding_sizes": embedding_sizes,
}
joblib.dump(preprocessors, "preprocessors.pkl")
torch.save(model.state_dict(), "v8_dl_model.pth")
print("Done. preprocessors.pkl and v8_dl_model.pth saved.")
