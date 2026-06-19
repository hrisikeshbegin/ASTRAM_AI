import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger
import joblib
from sentence_transformers import SentenceTransformer
import random

app = FastAPI(
    title="ASTRAM-AI V8.2 TITAN",
    description="Zero-Touch Multi-Modal Probabilistic Grid",
    version="8.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

random.seed(42)
GLOBAL_STATIONS = [f"BLR_Node_{str(i).zfill(4)}" for i in range(1, 1001)]
BASE_COSTS = {station: random.randint(50, 300) for station in GLOBAL_STATIONS}

try:
    preprocessors = joblib.load("preprocessors.pkl")
    encoders = preprocessors['encoders']
    scaler = preprocessors['scaler']
    embedding_sizes = preprocessors['embedding_sizes']
except Exception as e:
    print(f"Warning: {e}")

nlp_model = None


def get_nlp_model():
    global nlp_model
    if nlp_model is None:
        try:
            nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"SentenceTransformer load warning: {e}")
            nlp_model = None
    return nlp_model


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


try:
    dl_model = V8TitanNet(embedding_sizes, num_time=4, num_bool=1)
    dl_model.load_state_dict(torch.load("v8_dl_model.pth", weights_only=True))
    dl_model.eval()
except Exception as e:
    print(f"Model load warning: {e}")


class AgenticLLMNode:
    def analyze(self, text: str):
        text = text.lower()
        context = {
            "sop_flag": "Standard",
            "hazmat_risk": False,
            "sentiment_vector": 1.0,
            "inferred_priority": "Low",
            "force_road_closure": False
        }
        if "fire" in text or "spill" in text or "chemical" in text or "hazmat" in text:
            context.update({"sop_flag": "HAZMAT_PROTOCOL", "hazmat_risk": True, "sentiment_vector": 2.5, "inferred_priority": "High", "force_road_closure": True})
        elif "fatal" in text or "crash" in text or "injur" in text or "accident" in text:
            context.update({"sop_flag": "CODE_RED_MEDICAL", "sentiment_vector": 3.0, "inferred_priority": "High", "force_road_closure": True})
        elif "water" in text or "flood" in text or "heavy" in text:
            context.update({"sop_flag": "CIVIC_INFRA_ALERT", "sentiment_vector": 1.5, "inferred_priority": "High", "force_road_closure": True})
        elif "block" in text or "stuck" in text or "jam" in text:
            context.update({"inferred_priority": "Medium", "sentiment_vector": 1.2})
        return context


llm_node = AgenticLLMNode()


class TrafficEventInput(BaseModel):
    event_type: str
    event_cause: str
    latitude: float
    longitude: float
    veh_type: str
    description: str


class BayesianPrediction(BaseModel):
    mean_impact_score: float
    mean_duration_mins: float
    confidence_interval: str
    epistemic_uncertainty: float
    ai_predicted_road_closure: bool


class PredictionResponse(BaseModel):
    bayesian_forecast: BayesianPrediction
    llm_reasoning_engine: dict
    rl_agentic_deployment: dict


@app.post("/api/v8/cognitive_grid", response_model=PredictionResponse)
async def predict_event(event: TrafficEventInput):
    try:
        llm_ctx = llm_node.analyze(event.description)

        nlp = get_nlp_model()
        if nlp is None:
            text_embedding = np.zeros((1, 384))
        else:
            text_embedding = nlp.encode([event.description])
        t_text = torch.tensor(text_embedding, dtype=torch.float32)

        def safe_enc(col, val):
            try:
                return encoders[col].transform([val])[0]
            except:
                return 0

        c_feats = [safe_enc('event_cause', event.event_cause), safe_enc('veh_type', event.veh_type)]
        spatial_feats = scaler.transform([[event.latitude, event.longitude]])[0].tolist()

        now = datetime.datetime.now(datetime.timezone.utc)
        time_feats = [
            np.sin(2 * np.pi * now.hour / 24),
            np.cos(2 * np.pi * now.hour / 24),
            np.sin(2 * np.pi * now.weekday() / 7),
            np.cos(2 * np.pi * now.weekday() / 7)
        ]
        b_feats = [1.0 if now.weekday() >= 5 else 0.0]

        t_cat = torch.tensor([c_feats], dtype=torch.long)
        t_spatial = torch.tensor([spatial_feats], dtype=torch.float32)
        t_time = torch.tensor([time_feats], dtype=torch.float32)
        t_bool = torch.tensor([b_feats], dtype=torch.float32)

        with torch.no_grad():
            sev_tensor, mu_tensor, sigma_tensor, clos_logits = dl_model(t_cat, t_spatial, t_time, t_bool, t_text)
            base_sev = sev_tensor.item()
            base_dur = np.expm1(mu_tensor.item())
            model_uncertainty = sigma_tensor.item()
            closure_prob = torch.sigmoid(clos_logits).item()
            predicted_closure = True if closure_prob > 0.5 else False
            if llm_ctx["force_road_closure"]:
                predicted_closure = True

        final_sev = min(100.0, base_sev * llm_ctx["sentiment_vector"])
        final_dur = max(5.0, base_dur + (final_sev * 0.3))

        prob = LpProblem("City_Wide_RL_Optimizer", LpMinimize)
        x = LpVariable.dicts("Officers", GLOBAL_STATIONS, lowBound=0, cat=LpInteger)

        dynamic_costs = {}
        for station in GLOBAL_STATIONS:
            distance_penalty = (hash(station + str(event.latitude) + str(event.longitude)) % 150)
            dynamic_costs[station] = BASE_COSTS[station] + distance_penalty

        prob += lpSum([dynamic_costs[i] * x[i] for i in GLOBAL_STATIONS])

        if llm_ctx["inferred_priority"] == "High":
            deployment_divisor = 8
        elif llm_ctx["inferred_priority"] == "Medium":
            deployment_divisor = 15
        else:
            deployment_divisor = 25

        total_officers_needed = max(1, int(final_sev / deployment_divisor))
        prob += lpSum([x[i] for i in GLOBAL_STATIONS]) >= total_officers_needed
        prob.solve()

        deployed = 0
        active_dispatchers = []
        for station in GLOBAL_STATIONS:
            if x[station].varValue > 0:
                deployed += x[station].varValue
                active_dispatchers.append(f"{station} ({int(x[station].varValue)} units)")

        primary_dispatch = " & ".join(active_dispatchers[:2])
        if len(active_dispatchers) > 2:
            primary_dispatch += f" + {len(active_dispatchers) - 2} other nodes"

        return {
            "bayesian_forecast": {
                "mean_impact_score": round(final_sev, 2),
                "mean_duration_mins": round(final_dur, 1),
                "confidence_interval": f"± {round(model_uncertainty * 1.96, 2)} Log-Mins (95% CI)",
                "epistemic_uncertainty": round(model_uncertainty, 4),
                "ai_predicted_road_closure": predicted_closure
            },
            "llm_reasoning_engine": llm_ctx,
            "rl_agentic_deployment": {
                "dispatch_node": primary_dispatch,
                "officers_assigned": int(deployed),
                "barricades": 20 if predicted_closure else int(final_sev / 20),
                "special_units": "Hazmat Team" if llm_ctx["hazmat_risk"] else "Standard Units"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
