import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u

st.set_page_config(page_title="4th-Down Without Coefficients", layout="centered")

st.markdown(
    """
    <style>
      .home-container { margin-bottom: 1rem; }
      .home-container a {
        background-color: #FF5722;
        color: #FFF !important;
        text-decoration: none;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: bold;
      }
    </style>
    <div class="home-container">
      <a href="/">🏠 Home</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("4th-Down Success • Without Team Coefficients")
tabs = st.tabs(["🗂 Gather Data", "⚙️ Run Model", "📊 Visualize"])

with tabs[0]:
    years = st.multiselect(
        "Select Season Year(s)",
        options=list(range(2000, 2025)),
        default=[2022],
        key="no_years"
    )
    weeks = st.multiselect(
        "Select Week(s)",
        options=list(range(1, 16)),
        default=[1],
        key="no_weeks"
    )
    if st.button("📥 Gather Data", key="gather_no"):
        drives = [u.fetch_drives(y, w) for y in years for w in weeks]
        plays  = [u.fetch_plays(y, w)   for y in years for w in weeks]

        df_drives = pd.concat(drives, ignore_index=True)
        df_plays  = pd.concat(plays,  ignore_index=True)

        st.session_state.drives_no = df_drives
        st.session_state.plays_no  = df_plays

        st.success(f"Fetched {len(df_drives):,} drives & {len(df_plays):,} plays.")
        st.download_button("⬇️ Download Drives CSV",
                           df_drives.to_csv(index=False),
                           "drives_no.csv","text/csv")
        st.download_button("⬇️ Download Plays CSV",
                           df_plays.to_csv(index=False),
                           "plays_no.csv","text/csv")

        st.subheader("Collected Drives")
        st.dataframe(df_drives)
        st.subheader("Collected Plays")
        st.dataframe(df_plays)

with tabs[1]:
    if "drives_no" in st.session_state and "plays_no" in st.session_state:
        threshold = 30
        counts    = st.session_state.drives_no["offense_idx"].value_counts()
        valid_off = counts[counts >= threshold].index
        st.write(f"Filtering to teams with ≥{threshold} drives.")
        
        plays_filt = st.session_state.plays_no[
            st.session_state.plays_no["offense_idx"].isin(valid_off)
        ]
        if plays_filt.empty:
            st.error("No plays left after filtering — try more seasons/weeks.")
        else:
            st.write(f"{len(plays_filt):,} plays remain after filter.")
            if st.button("▶️ Train 4th-Down Model", key="run_no"):
                dfp = u.train_no_coeff_model(plays_filt)
                st.session_state.df_no_pred = dfp
                st.success("Model trained!")
                st.download_button("⬇️ Download Predictions CSV",
                                   dfp.to_csv(index=False),
                                   "preds_no_coeff.csv","text/csv")
                st.subheader("Predictions Dataset")
                st.dataframe(dfp)
    else:
        st.info("Please gather data in Step 1 first.")

with tabs[2]:
    if "df_no_pred" not in st.session_state:
        st.info("Please run the model in Step 2 first.")
    else:
        dfv = st.session_state.df_no_pred
        team = (
            dfv.groupby("offense")
               .agg(Actual=("success","mean"),
                    Predicted=("success_prob","mean"))
               .reset_index()
        )
        team["Actual"]    *= 100
        team["Predicted"] *= 100

        st.subheader("Team Conversion: Actual vs Predicted")
        fig2, ax2 = plt.subplots(figsize=(8,4))
        x = np.arange(len(team))
        ax2.plot(x, team["Actual"],    marker="o", linestyle="-",  label="Actual")
        ax2.plot(x, team["Predicted"], marker="o", linestyle="--", label="Predicted")
        ax2.set_xticks(x)
        ax2.set_xticklabels(team["offense"], rotation=90, fontsize=7)
        ax2.set_ylabel("Conversion Rate (%)")
        ax2.legend()
        st.pyplot(fig2)