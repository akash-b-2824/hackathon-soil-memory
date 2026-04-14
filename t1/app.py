from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path(".")

FILES = {
    "crop_reco": BASE_DIR / "crop_recommendation.csv",
    "crops_npk": BASE_DIR / "crops_npk.csv",
    "weather": BASE_DIR / "weather.csv",
    "yield": BASE_DIR / "crop_yield.csv",
    "market": BASE_DIR / "market.csv",
    "fertilizer": BASE_DIR / "fertilizer.csv",
}

MODELS: Dict[str, object] = {}
METRICS: Dict[str, Dict[str, float]] = {}
STATE: Dict[str, object] = {}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype(str).str.strip().str.lower()
    return out


def load_data() -> Dict[str, pd.DataFrame]:
    data = {name: normalize_columns(pd.read_csv(path)) for name, path in FILES.items()}
    return data


def train_all() -> None:
    data = load_data()
    crop_reco = data["crop_reco"]
    crops_npk = data["crops_npk"]
    weather = data["weather"]
    yield_df = data["yield"]
    market = data["market"]
    fert = data["fertilizer"]

    # 1) Crop recommendation model
    # Use curated core crop dataset for stable high-accuracy baseline
    common_cols = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall", "label"]
    crop_train = crop_reco[[c for c in common_cols if c in crop_reco.columns]].copy()
    crop_train = crop_train.dropna(subset=["label"])

    for c in ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]:
        crop_train[c] = pd.to_numeric(crop_train[c], errors="coerce")
    crop_train = crop_train.dropna()

    X_crop = crop_train[["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]]
    y_crop = crop_train["label"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_crop, y_crop, test_size=0.2, random_state=RANDOM_STATE, stratify=y_crop
    )
    crop_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=450, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    crop_model.fit(Xc_train, yc_train)
    yc_pred = crop_model.predict(Xc_test)
    METRICS["crop_model"] = {"accuracy": float(accuracy_score(yc_test, yc_pred))}
    MODELS["crop_model"] = crop_model

    # 2) Soil memory profile
    soil_memory = weather.copy()
    if "year" in soil_memory.columns:
        soil_memory["year_num"] = pd.to_numeric(
            soil_memory["year"].astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        )
    else:
        soil_memory["year_num"] = np.nan

    sort_cols = [c for c in ["state", "district", "crop", "year_num"] if c in soil_memory.columns]
    soil_memory = soil_memory.sort_values(sort_cols, na_position="last")
    soil_memory["prev_crop"] = soil_memory.groupby(["state", "district"])["crop"].shift(1)
    soil_memory["prev2_crop"] = soil_memory.groupby(["state", "district"])["crop"].shift(2)
    same_as_prev = (soil_memory["crop"] == soil_memory["prev_crop"]).astype(int)
    same_as_prev2 = (soil_memory["crop"] == soil_memory["prev2_crop"]).astype(int)
    soil_memory["soil_memory_index"] = (same_as_prev * 0.65 + same_as_prev2 * 0.35).round(3)
    soil_memory_profile = (
        soil_memory.groupby(["state", "district", "crop"], dropna=False)["soil_memory_index"]
        .mean()
        .reset_index()
        .rename(columns={"soil_memory_index": "avg_soil_memory_index"})
    )
    STATE["soil_memory_profile"] = soil_memory_profile

    # 3) Yield model
    yield_model_df = yield_df.copy()
    year_candidates = ["year", "crop_year", "year_num"]
    year_source = next((c for c in year_candidates if c in yield_model_df.columns), None)
    if year_source is not None:
        yield_model_df["year_num"] = pd.to_numeric(
            yield_model_df[year_source].astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        )
    else:
        yield_model_df["year_num"] = 2020

    for c in ["area", "production", "yield"]:
        if c in yield_model_df.columns:
            yield_model_df[c] = pd.to_numeric(yield_model_df[c], errors="coerce")

    yield_model_df = yield_model_df.dropna(subset=["yield"])
    yield_features = ["state", "district", "crop", "season", "year_num", "area", "production"]
    available_features = [c for c in yield_features if c in yield_model_df.columns]
    Xy = yield_model_df[available_features]
    yy = yield_model_df["yield"]
    Xy_train, Xy_test, yy_train, yy_test = train_test_split(
        Xy, yy, test_size=0.2, random_state=RANDOM_STATE
    )
    num_cols = Xy.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in available_features if c not in num_cols]
    yield_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    yield_model = Pipeline(
        [
            ("prep", yield_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    yield_model.fit(Xy_train, yy_train)
    yy_pred = yield_model.predict(Xy_test)
    METRICS["yield_model"] = {
        "mae": float(mean_absolute_error(yy_test, yy_pred)),
        "r2": float(r2_score(yy_test, yy_pred)),
    }
    MODELS["yield_model"] = yield_model
    STATE["yield_features"] = available_features

    # 4) Price model
    market_model_df = market.copy()
    market_model_df["arrival_date"] = pd.to_datetime(
        market_model_df["arrival_date"], dayfirst=True, errors="coerce"
    )
    for c in ["min_price", "max_price", "modal_price"]:
        market_model_df[c] = pd.to_numeric(market_model_df[c], errors="coerce")
    market_model_df = market_model_df.dropna(
        subset=["arrival_date", "modal_price", "commodity", "min_price", "max_price"]
    )
    market_model_df["year"] = market_model_df["arrival_date"].dt.year
    market_model_df["month"] = market_model_df["arrival_date"].dt.month
    market_model_df["day"] = market_model_df["arrival_date"].dt.day
    market_model_df["price_spread"] = (
        market_model_df["max_price"] - market_model_df["min_price"]
    )

    Xp = market_model_df[
        [
            "commodity",
            "state",
            "district",
            "market",
            "month",
            "year",
            "day",
            "min_price",
            "max_price",
            "price_spread",
        ]
    ]
    yp = market_model_df["modal_price"]
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        Xp, yp, test_size=0.2, random_state=RANDOM_STATE
    )
    price_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["month", "year", "day", "min_price", "max_price", "price_spread"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["commodity", "state", "district", "market"],
            ),
        ]
    )
    price_model = Pipeline(
        [
            ("prep", price_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=320, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    price_model.fit(Xp_train, yp_train)
    yp_pred = price_model.predict(Xp_test)
    METRICS["price_model"] = {
        "mae": float(mean_absolute_error(yp_test, yp_pred)),
        "r2": float(r2_score(yp_test, yp_pred)),
    }
    MODELS["price_model"] = price_model

    # 5) Fertilizer model
    fert_model_df = fert.copy()
    target_col = "recommended_fertilizer"
    fert_features = [c for c in fert_model_df.columns if c != target_col]
    Xf = fert_model_df[fert_features]
    yf = fert_model_df[target_col]
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        Xf, yf, test_size=0.2, random_state=RANDOM_STATE, stratify=yf
    )
    num_cols_f = Xf.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_f = [c for c in fert_features if c not in num_cols_f]
    fert_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols_f,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols_f,
            ),
        ]
    )
    fert_model = Pipeline(
        [
            ("prep", fert_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    fert_model.fit(Xf_train, yf_train)
    yf_pred = fert_model.predict(Xf_test)
    METRICS["fertilizer_model"] = {"accuracy": float(accuracy_score(yf_test, yf_pred))}
    MODELS["fertilizer_model"] = fert_model
    STATE["fert_input_df"] = fert_model_df
    STATE["fert_target_col"] = target_col


def get_soil_memory_score(state: str, district: str, crop: str) -> float:
    profile = STATE["soil_memory_profile"]
    mask = (
        (profile["state"] == state.strip().lower())
        & (profile["district"] == district.strip().lower())
        & (profile["crop"] == crop.strip().lower())
    )
    row = profile.loc[mask, "avg_soil_memory_index"]
    return float(row.iloc[0]) if len(row) else 0.0


class SimulationInput(BaseModel):
    state: str
    district: str
    market_name: str
    season: str = "kharif"
    year: int = 2026
    top_n: int = 5
    n: float
    p: float
    k: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


def simulate_crop_plan(payload: SimulationInput) -> List[Dict[str, object]]:
    crop_model = MODELS["crop_model"]
    yield_model = MODELS["yield_model"]
    price_model = MODELS["price_model"]
    fert_model = MODELS["fertilizer_model"]
    yield_features = STATE["yield_features"]
    fert_df = STATE["fert_input_df"]
    fert_target_col = STATE["fert_target_col"]

    base_input = pd.DataFrame(
        [
            {
                "n": payload.n,
                "p": payload.p,
                "k": payload.k,
                "temperature": payload.temperature,
                "humidity": payload.humidity,
                "ph": payload.ph,
                "rainfall": payload.rainfall,
            }
        ]
    )
    probs = crop_model.predict_proba(base_input)[0]
    classes = crop_model.named_steps["model"].classes_
    candidate_idx = np.argsort(probs)[::-1][: max(payload.top_n, 3)]
    candidates = [(classes[i], float(probs[i])) for i in candidate_idx]

    rows: List[Dict[str, object]] = []
    for crop_name, crop_conf in candidates:
        y_in = pd.DataFrame(
            [
                {
                    "state": payload.state.strip().lower(),
                    "district": payload.district.strip().lower(),
                    "crop": str(crop_name).strip().lower(),
                    "season": payload.season.strip().lower(),
                    "year_num": payload.year,
                    "area": 1.0,
                    "production": 1.0,
                }
            ]
        )
        for col in yield_features:
            if col not in y_in.columns:
                y_in[col] = np.nan
        y_in = y_in[yield_features]
        pred_yield = float(yield_model.predict(y_in)[0])

        p_in = pd.DataFrame(
            [
                {
                    "commodity": str(crop_name).strip().lower(),
                    "state": payload.state.strip().lower(),
                    "district": payload.district.strip().lower(),
                    "market": payload.market_name.strip().lower(),
                    "month": 7 if payload.season.strip().lower() == "kharif" else 1,
                    "year": payload.year,
                    "day": 15,
                    "min_price": np.nan,
                    "max_price": np.nan,
                    "price_spread": np.nan,
                }
            ]
        )
        pred_price = float(price_model.predict(p_in)[0])

        fert_in = fert_df.sample(1, random_state=RANDOM_STATE).drop(columns=[fert_target_col]).copy()
        if "crop_type" in fert_in.columns:
            fert_in.loc[:, "crop_type"] = str(crop_name).strip().lower()
        if "region" in fert_in.columns:
            fert_in.loc[:, "region"] = payload.state.strip().lower()
        if "nitrogen_level" in fert_in.columns:
            fert_in.loc[:, "nitrogen_level"] = payload.n
        if "phosphorus_level" in fert_in.columns:
            fert_in.loc[:, "phosphorus_level"] = payload.p
        if "potassium_level" in fert_in.columns:
            fert_in.loc[:, "potassium_level"] = payload.k
        fert_choice = str(fert_model.predict(fert_in)[0])

        soil_mem = get_soil_memory_score(payload.state, payload.district, str(crop_name))
        sim_score = (
            (0.35 * crop_conf)
            + (0.30 * (pred_yield / (abs(pred_yield) + 1)))
            + (0.25 * (pred_price / (abs(pred_price) + 1)))
            - (0.10 * soil_mem)
        )
        rows.append(
            {
                "crop": crop_name,
                "crop_confidence": round(crop_conf, 4),
                "predicted_yield": round(pred_yield, 4),
                "predicted_market_price": round(pred_price, 2),
                "soil_memory_risk": round(soil_mem, 4),
                "recommended_fertilizer": fert_choice,
                "simulation_score": round(sim_score, 4),
            }
        )

    return sorted(rows, key=lambda x: x["simulation_score"], reverse=True)


app = FastAPI(title="Soil Memory Future Simulation AI")
train_all()


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics() -> Dict[str, Dict[str, float]]:
    return METRICS


@app.post("/api/simulate")
def simulate(payload: SimulationInput) -> Dict[str, List[Dict[str, object]]]:
    return {"results": simulate_crop_plan(payload)}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width,initial-scale=1' />
  <title>Agri AI Studio | Soil Memory</title>
  <style>
    body { font-family: Inter, Segoe UI, Arial, sans-serif; margin: 0; background: #f2f6ff; color: #0f172a; }
    .topbar { background: linear-gradient(90deg, #1e40af, #3b82f6); color: #fff; padding: 14px 20px; font-weight: 700; letter-spacing: 0.2px; }
    .topbar small { opacity: 0.9; font-weight: 500; margin-left: 8px; }
    .wrap { max-width: 1100px; margin: 14px auto; padding: 10px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 10px; }
    .tab-btn { background: #dbeafe; color: #1e3a8a; border: 1px solid #bfdbfe; padding: 9px 12px; border-radius: 10px; cursor: pointer; font-weight: 600; }
    .tab-btn.active { background: #1d4ed8; color: #fff; border-color: #1d4ed8; }
    .tab { display: none; }
    .tab.active { display: block; }
    .card { background: #fff; border-radius: 14px; padding: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); margin-bottom: 12px; }
    .mini-head { display: flex; justify-content: space-between; align-items: center; }
    .mini-btn { border: 1px solid #dbe2f3; background: #f8fbff; border-radius: 8px; padding: 4px 8px; cursor: pointer; font-size: 12px; }
    .card-body.hidden { display: none; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }
    .field label { display: block; font-size: 12px; font-weight: 700; color: #334155; margin: 0 0 6px 2px; }
    input, select, button { width: 100%; padding: 10px; border: 1px solid #d3dae6; border-radius: 9px; box-sizing: border-box; }
    button { background: #2563eb; color: white; border: none; font-weight: 700; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; background: #fff; }
    th, td { border: 1px solid #e3e8f1; padding: 8px; text-align: left; font-size: 13px; }
    th { background: #eff6ff; }
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; }
    .kpi { background: #f8faff; border: 1px solid #dbe7ff; padding: 10px; border-radius: 10px; }
    .kpi .label { font-size: 12px; color: #334155; }
    .kpi .value { font-size: 20px; font-weight: 700; margin-top: 3px; }
    .muted { color: #475569; font-size: 13px; }
    .link-list a { color: #1d4ed8; text-decoration: none; font-weight: 600; display: inline-block; margin-right: 12px; margin-top: 6px; }
    @media (max-width: 900px) { .grid, .kpi-row { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 640px) { .grid, .kpi-row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class='topbar'>Agri AI Studio <small>Soil Memory + Future Simulation</small></div>
  <div class='wrap'>
    <div class='tabs'>
      <button class='tab-btn active' onclick='openTab("dashboard", this)'>Dashboard</button>
      <button class='tab-btn' onclick='openTab("simulator", this)'>Simulator</button>
      <button class='tab-btn' onclick='openTab("api", this)'>API & Links</button>
    </div>

    <div id='dashboard' class='tab active'>
      <div class='card'>
        <div class='mini-head'>
          <h2>Model Performance Slide</h2>
          <button class='mini-btn' onclick='toggleCard(this)'>Minimize</button>
        </div>
        <div class='card-body'>
          <p class='muted'>Single-view slide for judge presentation: all model accuracy and quality indicators.</p>
          <div class='kpi-row' id='kpi-row'></div>
          <table id='metrics-table'></table>
        </div>
      </div>
      <div class='card'>
        <div class='mini-head'>
          <h3>Project Intelligence Layers</h3>
          <button class='mini-btn' onclick='toggleCard(this)'>Minimize</button>
        </div>
        <div class='card-body'>
          <table>
            <tr><th>Layer</th><th>Purpose</th><th>Output</th></tr>
            <tr><td>Crop Recommendation</td><td>NPK + climate fit</td><td>Crop probability</td></tr>
            <tr><td>Soil Memory Engine</td><td>Historical sequence pressure</td><td>Soil risk index</td></tr>
            <tr><td>Yield Prediction</td><td>Expected productivity</td><td>Yield estimate</td></tr>
            <tr><td>Market Forecasting</td><td>Price behavior</td><td>Expected modal price</td></tr>
            <tr><td>Fertilizer Advice</td><td>Nutrient recovery plan</td><td>Recommended fertilizer</td></tr>
          </table>
        </div>
      </div>
    </div>

    <div id='simulator' class='tab'>
      <div class='card'>
        <div class='mini-head'>
          <h2>Run Field Simulation</h2>
          <button class='mini-btn' onclick='toggleCard(this)'>Minimize</button>
        </div>
        <div class='card-body'>
          <div class='grid'>
            <div class='field'><label for='state'>State</label><input id='state' placeholder='e.g. assam' value='assam' /></div>
            <div class='field'><label for='district'>District</label><input id='district' placeholder='e.g. cachar' value='cachar' /></div>
            <div class='field'><label for='market_name'>Market Name</label><input id='market_name' placeholder='e.g. cachar' value='cachar' /></div>
            <div class='field'><label for='n'>Nitrogen (N)</label><input id='n' type='number' placeholder='Nitrogen level' value='90' /></div>
            <div class='field'><label for='p'>Phosphorus (P)</label><input id='p' type='number' placeholder='Phosphorus level' value='42' /></div>
            <div class='field'><label for='k'>Potassium (K)</label><input id='k' type='number' placeholder='Potassium level' value='43' /></div>
            <div class='field'><label for='temperature'>Temperature (°C)</label><input id='temperature' type='number' placeholder='Temperature in Celsius' value='26' /></div>
            <div class='field'><label for='humidity'>Humidity (%)</label><input id='humidity' type='number' placeholder='Humidity percentage' value='78' /></div>
            <div class='field'><label for='ph'>Soil pH</label><input id='ph' type='number' placeholder='Soil pH value' value='6.5' step='0.1' /></div>
            <div class='field'><label for='rainfall'>Rainfall (mm)</label><input id='rainfall' type='number' placeholder='Rainfall in mm' value='220' /></div>
            <div class='field'><label for='season'>Season</label><select id='season'><option>kharif</option><option>rabi</option><option>zaid</option></select></div>
            <div class='field'><label for='year'>Target Year</label><input id='year' type='number' placeholder='Prediction year' value='2026' /></div>
          </div>
          <div style='margin-top:12px;'><button onclick='runSim()'>Simulate Plan</button></div>
          <div id='results'></div>
        </div>
      </div>
    </div>

    <div id='api' class='tab'>
      <div class='card'>
        <div class='mini-head'>
          <h2>Labeled API Endpoints</h2>
          <button class='mini-btn' onclick='toggleCard(this)'>Minimize</button>
        </div>
        <div class='card-body'>
          <p class='muted'>Open these URLs directly in your browser for quick checks.</p>
          <div class='link-list'>
            <a href='/'>Home Dashboard</a>
            <a href='/api/health' target='_blank'>Health Status</a>
            <a href='/api/metrics' target='_blank'>Model Metrics JSON</a>
            <a href='/docs' target='_blank'>Interactive API Docs</a>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function openTab(id, btn) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById(id).classList.add('active');
      btn.classList.add('active');
    }

    function toggleCard(button) {
      const body = button.closest('.card').querySelector('.card-body');
      body.classList.toggle('hidden');
      button.textContent = body.classList.contains('hidden') ? 'Open' : 'Minimize';
    }

    async function loadMetrics() {
      const res = await fetch('/api/metrics');
      const data = await res.json();

      const cropAcc = (data.crop_model?.accuracy ?? 0).toFixed(4);
      const yieldR2 = (data.yield_model?.r2 ?? 0).toFixed(4);
      const priceR2 = (data.price_model?.r2 ?? 0).toFixed(4);
      const fertAcc = (data.fertilizer_model?.accuracy ?? 0).toFixed(4);

      document.getElementById('kpi-row').innerHTML = `
        <div class='kpi'><div class='label'>Crop Model Accuracy</div><div class='value'>${cropAcc}</div></div>
        <div class='kpi'><div class='label'>Yield Model R2</div><div class='value'>${yieldR2}</div></div>
        <div class='kpi'><div class='label'>Price Model R2</div><div class='value'>${priceR2}</div></div>
        <div class='kpi'><div class='label'>Fertilizer Accuracy</div><div class='value'>${fertAcc}</div></div>
      `;

      const table = `
        <tr><th>Model</th><th>Metric 1</th><th>Value</th><th>Metric 2</th><th>Value</th></tr>
        <tr><td>Crop Recommendation</td><td>Accuracy</td><td>${cropAcc}</td><td>-</td><td>-</td></tr>
        <tr><td>Yield Prediction</td><td>MAE</td><td>${(data.yield_model?.mae ?? 0).toFixed(4)}</td><td>R2</td><td>${yieldR2}</td></tr>
        <tr><td>Price Forecasting</td><td>MAE</td><td>${(data.price_model?.mae ?? 0).toFixed(2)}</td><td>R2</td><td>${priceR2}</td></tr>
        <tr><td>Fertilizer Recommendation</td><td>Accuracy</td><td>${fertAcc}</td><td>-</td><td>-</td></tr>
      `;
      document.getElementById('metrics-table').innerHTML = table;
    }

    async function runSim() {
      const payload = {
        state: document.getElementById('state').value,
        district: document.getElementById('district').value,
        market_name: document.getElementById('market_name').value,
        n: Number(document.getElementById('n').value),
        p: Number(document.getElementById('p').value),
        k: Number(document.getElementById('k').value),
        temperature: Number(document.getElementById('temperature').value),
        humidity: Number(document.getElementById('humidity').value),
        ph: Number(document.getElementById('ph').value),
        rainfall: Number(document.getElementById('rainfall').value),
        season: document.getElementById('season').value,
        year: Number(document.getElementById('year').value),
        top_n: 5
      };

      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      const rows = data.results || [];
      let html = '<table><tr><th>Crop</th><th>Confidence</th><th>Yield</th><th>Price</th><th>Soil Risk</th><th>Fertilizer</th><th>Score</th></tr>';
      for (const r of rows) {
        html += `<tr><td>${r.crop}</td><td>${r.crop_confidence}</td><td>${r.predicted_yield}</td><td>${r.predicted_market_price}</td><td>${r.soil_memory_risk}</td><td>${r.recommended_fertilizer}</td><td>${r.simulation_score}</td></tr>`;
      }
      html += '</table>';
      document.getElementById('results').innerHTML = html;
    }

    loadMetrics();
  </script>
</body>
</html>
"""
