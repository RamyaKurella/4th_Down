# app.py  – 4-Down Team-Ratings Explorer (wording updated)

import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, requests
from streamlit_lottie import st_lottie
import utils as u

st.set_page_config("Team Ratings", layout="wide")

# ── dark theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{background:#0d1117;color:#c9d1d9;}
h2,h3,strong{color:#e6edf3;}
.stButton>button{background:#238636;color:#fff;border:none;
  border-radius:8px;padding:0.45rem 1.3rem;font-weight:600;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_lottie(url:str):
    try:
        r=requests.get(url,timeout=10);r.raise_for_status();return r.json()
    except: return None

# ── header ───────────────────────────────────────────────────────
c1,c2=st.columns([1,5])
with c1:
    ani=load_lottie("https://assets7.lottiefiles.com/packages/lf20_x62chJ.json")
    if ani: st_lottie(ani,height=90,width=90)
with c2:
    st.markdown("## 🏈 Team Ratings Explorer")

st.markdown("*Pipeline → Data • Team Ratings (w/o coeffs) • Ridge • Team Ratings (with coeffs)*")

# ── session slots ────────────────────────────────────────────────
for k in ("drives","plays","coeffs","ratings_no","ratings_with"):
    st.session_state.setdefault(k, pd.DataFrame())

tab_data,tab_no,tab_coeff,tab_with = st.tabs(
    ["📥 Data",
     "📊 Team Ratings — Without Coeffs",
     "🧮 Coefficients",
     "⭐ Team Ratings — With Coeffs"])

# ╔════════  Data  ════════╗
with tab_data:
    st.subheader("Fetch / Inspect Raw Data")

    years = st.multiselect("Years",  range(2000, 2025),
                           default=st.session_state.get("years", [2022]))
    weeks = st.multiselect("Weeks",  range(1, 16),
                           default=st.session_state.get("weeks", [1]))

    if st.button("🚀 Fetch Data"):
        if not years or not weeks:
            st.warning("Select at least one year AND one week."); st.stop()

        with st.spinner("Downloading drives & plays …"):
            st.session_state.drives = pd.concat(
                [u.fetch_drives(y, w) for y in years for w in weeks],
                ignore_index=True)
            st.session_state.plays  = pd.concat(
                [u.fetch_plays (y, w) for y in years for w in weeks],
                ignore_index=True)

        # reset downstream objects
        st.session_state.update(dict(
            years=years, weeks=weeks,
            coeffs=pd.DataFrame(),
            ratings_no=pd.DataFrame(),
            ratings_with=pd.DataFrame()
        ))

        st.success(f"Fetched **{len(st.session_state.drives):,} drives** "
                   f"and **{len(st.session_state.plays):,} plays** ✔️")

    # ───────────────────── show tables & downloads ──────────────────
    if not st.session_state.drives.empty:
        with st.expander(f"📄 Drives sample ({len(st.session_state.drives):,} rows)"):
            st.dataframe(st.session_state.drives.head(), use_container_width=True)
            st.download_button("⬇ Download ALL drives (.csv)",
                               st.session_state.drives.to_csv(index=False),
                               "drives_raw.csv", "text/csv")

    if not st.session_state.plays.empty:
        with st.expander(f"📄 Plays sample ({len(st.session_state.plays):,} rows)"):
            st.dataframe(st.session_state.plays.head(), use_container_width=True)
            st.download_button("⬇ Download ALL plays (.csv)",
                               st.session_state.plays.to_csv(index=False),
                               "plays_raw.csv", "text/csv")

# ╔════════  Team Ratings WITHOUT coefficients  ════════╗
with tab_no:
    st.subheader("Team Ratings — **Without Team Coefficients**")
    if st.session_state.plays.empty:
        st.info("➡️ Fetch data first.")
    else:
        if st.button("⚙️ Run Ratings (Without Coeffs)"):
            valid=(st.session_state.drives.offense_idx.value_counts().loc[lambda s:s>=30].index)
            play_ok=st.session_state.plays[st.session_state.plays.offense_idx.isin(valid)]
            if play_ok.empty:
                st.error("No teams with ≥30 drives."); st.stop()
            preds = u.train_no_coeff_model(play_ok)
            st.session_state.ratings_no = (
                preds.groupby("offense")
                     .agg(Attempts=("success","count"),
                          Actual  =("success","mean"),
                          Pred    =("success_prob","mean"))
                     .reset_index()
                     .assign(Actual=lambda d:d.Actual*100,
                             Pred  =lambda d:d.Pred  *100)
                     .sort_values("Actual",ascending=False))
            st.success("Ratings computed ✔️")

        if not st.session_state.ratings_no.empty:
            df = st.session_state.ratings_no
            st.dataframe(df.round(2), use_container_width=True)
            st.download_button("⬇ CSV", df.to_csv(index=False),
                               "team_ratings_without_coeffs.csv","text/csv")

            # plot
            top=df.head(20); x=np.arange(len(top)); w=0.35
            fig,ax=plt.subplots(figsize=(10,4))
            ax.bar(x-w/2, top.Actual,w,label="Actual")
            ax.bar(x+w/2, top.Pred  ,w,label="Predicted")
            ax.set_xticks(x); ax.set_xticklabels(top.offense,rotation=90,fontsize=8)
            ax.set_ylabel("%"); ax.set_ylim(0,100)
            ax.set_title("Top 20 — Team Ratings Without Coefficients"); ax.legend()
            st.pyplot(fig)

# ╔════════  Coefficients  ════════╗
with tab_coeff:
    st.subheader("Ridge-Based Offense / Defense Coefficients")
    if st.session_state.drives.empty:
        st.info("➡️ Fetch data first.")
    else:
        if st.button("⚙️ Compute Coefficients"):
            st.session_state.coeffs = u.compute_team_coefficients(st.session_state.drives,30)
            if st.session_state.coeffs.empty:
                st.error("Not enough drives."); st.stop()
            st.success("Coefficients ready!")
        if not st.session_state.coeffs.empty:
            st.dataframe(st.session_state.coeffs, use_container_width=True)
            st.download_button("⬇ CSV",
                               st.session_state.coeffs.to_csv(index=False),
                               "team_coefficients.csv","text/csv")

# ╔════════  Team Ratings WITH coefficients  ════════╗
with tab_with:
    st.subheader("Team Ratings — **With Team Coefficients**")
    if st.session_state.coeffs.empty or st.session_state.plays.empty:
        st.info("➡️ Need data *and* coefficients.")
    else:
        if st.button("⚙️ Run Ratings (With Coeffs)"):
            preds_w = u.train_with_coeff_model(st.session_state.plays, st.session_state.coeffs)
            if preds_w.empty: st.error("No rows."); st.stop()

            if {'success','success_prob'}.issubset(preds_w.columns):
                agg=(preds_w.groupby("offense")
                     .agg(Actual=("success","mean"),
                          Pred  =("success_prob","mean"))
                     .reset_index()
                     .assign(Actual=lambda d:d.Actual*100,
                             Pred  =lambda d:d.Pred  *100))
            else:
                agg=(preds_w
                     .assign(Actual=lambda d:d.ActualRate.str.rstrip('%').astype(float),
                             Pred  =lambda d:d.PredRate  .str.rstrip('%').astype(float))
                     [["offense","Actual","Pred"]])
            st.session_state.ratings_with = agg.sort_values("Actual",ascending=False)
            st.success("Ratings computed ✔️")

        if not st.session_state.ratings_with.empty:
            df = st.session_state.ratings_with
            st.dataframe(df.round(2), use_container_width=True)
            st.download_button("⬇ CSV", df.to_csv(index=False),
                               "team_ratings_with_coeffs.csv","text/csv")

            # plot
            top=df.head(20); x=np.arange(len(top)); w=0.35
            fig2,ax2=plt.subplots(figsize=(10,4))
            ax2.bar(x-w/2, top.Actual,w,label="Actual")
            ax2.bar(x+w/2, top.Pred  ,w,label="Predicted")
            ax2.set_xticks(x); ax2.set_xticklabels(top.offense,rotation=90,fontsize=8)
            ax2.set_ylabel("%"); ax2.set_ylim(0,100)
            ax2.set_title("Top 20 — Team Ratings With Coefficients"); ax2.legend()
            st.pyplot(fig2)