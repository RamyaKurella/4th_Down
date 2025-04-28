import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder

# ──────────── Configuration ─────────────
API_URL = "https://api.collegefootballdata.com"
CACHE = Path(".cache")
CACHE.mkdir(exist_ok=True)

def _auth_header():
    key = st.secrets.get("api", {}).get("cfb_api_key") or os.getenv("CFB_API_KEY")
    if not key:
        st.error("🔑 Missing CFB API key. Set in .streamlit/secrets.toml or CFB_API_KEY env var.")
        st.stop()
    return {"Authorization": f"Bearer {key}"}

@st.cache_data(show_spinner="📥 Fetching drives…")
def fetch_drives(year: int, week: int) -> pd.DataFrame:
    url = f"{API_URL}/drives?year={year}&week={week}&classification=fbs"
    raw = pd.json_normalize(requests.get(url, headers=_auth_header(), timeout=10).json())
    return preprocess_drives(raw)

def preprocess_drives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_to_goal"] = 100 - df.get("start_yards_to_goal", 0)
    df["points_scored_offense"] = (
        df.get("end_offense_score", 0) - df.get("start_offense_score", 0)
    ).clip(lower=0)
    df["points_scored_defense"] = (
        df.get("end_defense_score", 0) - df.get("start_defense_score", 0)
    ).clip(lower=0)
    df["yards_gained_per_play"] = df.get("yards", 0) / df.get("plays", 1).replace(0, 1)
    df["score_diff_start"] = df.get("start_offense_score", 0) - df.get("start_defense_score", 0)
    df["yards_gained"] = df.get("end_yardline", 0) - df.get("start_yardline", 0)

    df.loc[
        (df.end_yardline < 0) & (df.drive_result == "Uncategorized"), "end_yardline"
    ] = 0
    df.loc[
        (df.end_yards_to_goal < 0) & (df.drive_result == "Uncategorized"), "end_yards_to_goal"
    ] = 0

    try:
        import ast

        def extract_time(t):
            d = ast.literal_eval(t) if isinstance(t, str) else {}
            return d.get("minutes", 0), d.get("seconds", 0)

        if "start_time" in df.columns:
            df[["start_time_minutes", "start_time_seconds"]] = df["start_time"].apply(
                lambda x: pd.Series(extract_time(x))
            )
            df["start_time_in_seconds"] = df.start_time_minutes * 60 + df.start_time_seconds

        if "end_time" in df.columns:
            df[["end_time_minutes", "end_time_seconds"]] = df["end_time"].apply(
                lambda x: pd.Series(extract_time(x))
            )
            df["end_time_in_seconds"] = df.end_time_minutes * 60 + df.end_time_seconds
    except Exception:
        pass

    le = LabelEncoder()
    matchup = df.get("offense", "") + "_" + df.get("defense", "")
    df["offense_defense_matchup"] = matchup
    df["offense_defense_matchup_encoded"] = le.fit_transform(matchup)
    df["offense_idx"] = le.fit_transform(df.get("offense", []))
    df["defense_idx"] = le.fit_transform(df.get("defense", []))
    df["offense_conference_idx"] = le.fit_transform(df.get("offense_conference", []))
    df["defense_conference_idx"] = le.fit_transform(df.get("defense_conference", []))
    return df

@st.cache_data(show_spinner="📥 Fetching plays…")
def fetch_plays(year: int, week: int) -> pd.DataFrame:
    url = f"{API_URL}/plays?year={year}&week={week}&classification=fbs"
    raw = pd.json_normalize(requests.get(url, headers=_auth_header(), timeout=10).json())
    return preprocess_plays(raw)

def preprocess_plays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df.down == 4]
    drop = ["Punt", "Kickoff", "Field Goal Good", "Field Goal Missed"]
    df = df[~df.play_type.isin(drop)]

    if "yards" in df.columns:
        df["yards_gained"] = df["yards"]

    if "is_success" in df.columns:
        df["success"] = df["is_success"].astype(int)
    elif "playSuccess" in df.columns:
        df["success"] = df["playSuccess"].astype(int)
    elif "yards_gained" in df.columns and "distance" in df.columns:
        df["success"] = (df["yards_gained"] >= df["distance"]).astype(int)
    else:
        df["success"] = 0

    df["distance_to_goal"] = 100 - df.get("yard_line", 0)
    df["red_zone"] = (df.get("yards_to_goal", df.get("distance", 0)) <= 20).astype(int)
    df["down_yards_interaction"] = df["down"] * df.get("yards_to_goal", df.get("distance", 0))
    df["distance_ratio"] = df["distance_to_goal"] / (df.get("yards_to_goal", df.get("distance", 1)) + 1e-5)
    df["score_diff_start"] = df.get("offense_score", 0) - df.get("defense_score", 0)
    df["close_score"] = (df["score_diff_start"].abs() < 10).astype(int)

    le = LabelEncoder()
    combo = df.get("offense", "") + "_" + df.get("defense", "")
    df["offense_defense_matchup"] = combo
    df["offense_idx"] = le.fit_transform(df.get("offense", []))
    df["defense_idx"] = le.fit_transform(df.get("defense", []))
    df["offense_conference_idx"] = le.fit_transform(df.get("offense_conference", []))
    df["defense_conference_idx"] = le.fit_transform(df.get("defense_conference", []))
    return df

def save_csv(df: pd.DataFrame, name: str) -> Path:
    out = CACHE / name
    df.to_csv(out, index=False)
    return out

@st.cache_data(show_spinner="🔧 Computing team coefficients…")
def compute_team_coefficients(drives: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    off_counts = drives["offense_idx"].value_counts()
    def_counts = drives["defense_idx"].value_counts()
    valid_off = off_counts[off_counts >= threshold].index
    valid_def = def_counts[def_counts >= threshold].index

    df = drives[
        drives.offense_idx.isin(valid_off) & drives.defense_idx.isin(valid_def)
    ].copy()
    if df.empty:
        st.error(f"No teams with ≥{threshold} drives.")
        return pd.DataFrame()

    raw_feats = ["start_period", "score_diff_start", "is_home_offense", "distance_to_goal", "goal_to_go"]
    feats = [f for f in raw_feats if f in df.columns]

    off_enc = pd.get_dummies(df["offense_idx"].astype(str), prefix="offense")
    X_off = pd.concat([off_enc, df[feats]], axis=1) if feats else off_enc
    y_off = df["points_scored_offense"]
    model_off = Ridge(alpha=1.0, random_state=42)
    model_off.fit(X_off, y_off)
    off_s = pd.Series(model_off.coef_[: X_off.shape[1]], index=X_off.columns)

    def_enc = pd.get_dummies(df["defense_idx"].astype(str), prefix="defense")
    X_def = pd.concat([def_enc, df[feats]], axis=1) if feats else def_enc
    y_def = df["points_scored_defense"]
    model_def = Ridge(alpha=1.0, random_state=42)
    model_def.fit(X_def, y_def)
    def_s = pd.Series(model_def.coef_[: X_def.shape[1]], index=X_def.columns)

    off_coeff = off_s.filter(regex=r"^offense_\d+$")
    def_coeff = def_s.filter(regex=r"^defense_\d+$")
    team_idxs = sorted(
        {int(c.split("_")[1]) for c in off_coeff.index}
        | {int(c.split("_")[1]) for c in def_coeff.index}
    )

    name_map = (
        df[["offense_idx", "offense"]]
        .drop_duplicates()
        .set_index("offense_idx")["offense"]
        .to_dict()
    )
    records = []
    for tid in team_idxs:
        records.append({
            "team_idx": tid,
            "team_name": name_map.get(tid, ""),
            "train_offense_coeff": float(off_coeff.get(f"offense_{tid}", 0.0)),
            "train_defense_coeff": float(def_coeff.get(f"defense_{tid}", 0.0)),
        })
    return pd.DataFrame(records)

@st.cache_data(show_spinner="🔧 Training 4th-Down (no coeff)…")
def train_no_coeff_model(plays: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if "offense_idx" in plays.columns:
        frames.append(pd.get_dummies(plays["offense_idx"], prefix="off"))
    if "defense_idx" in plays.columns:
        frames.append(pd.get_dummies(plays["defense_idx"], prefix="def"))
    for c in ("yards_to_goal", "distance_to_goal", "distance"):
        if c in plays.columns:
            frames.append(plays[[c]])
    if not frames:
        st.error("No features for 4th-down model.")
        return plays.assign(success_prob=0.0)
    X = pd.concat(frames, axis=1)

    y = plays.get("success", pd.Series(0, index=plays.index)).fillna(0).astype(int)
    if y.nunique() < 2:
        p = float(y.mean())
        st.warning(f"Target constant ({p}). Returning constant prob.")
        return plays.assign(success_prob=p)

    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X, y)
    df_out = plays.copy()
    df_out["success_prob"] = clf.predict_proba(X)[:, 1]
    return df_out

@st.cache_data(show_spinner="🔧 Training 4th-Down (with coeff)…")
def train_with_coeff_model(plays: pd.DataFrame, coeffs: pd.DataFrame) -> pd.DataFrame:
    df = plays.copy()
    if "success" not in df.columns:
        df["success"] = (df.get("yards_gained", 0) >= df.get("distance", 0)).astype(int)

    df = df.merge(coeffs, left_on="offense_idx", right_on="team_idx", how="left")
    df = df.dropna(subset=["yards_gained", "distance", "train_offense_coeff", "train_defense_coeff", "success"])
    if df.empty:
        st.error("No data after merging team coefficients.")
        return pd.DataFrame()

    X = df[["yards_gained", "distance", "train_offense_coeff", "train_defense_coeff"]]
    y = df["success"].astype(int)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    df["predicted_prob"] = clf.predict_proba(X)[:, 1]

    team_preds = (
        df.groupby("offense")
          .agg(
              Attempts=("success", "count"),
              ActualRate=("success", "mean"),
              PredRate=("predicted_prob", "mean"),
          )
          .reset_index()
    )
    team_preds["ActualRate"] = (team_preds["ActualRate"] * 100).map("{:.2f}%".format)
    team_preds["PredRate"]   = (team_preds["PredRate"] * 100).map("{:.2f}%".format)
    return team_preds