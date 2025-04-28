
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u

st.set_page_config(page_title="4th-Down With Coefficients", layout="centered")

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

st.title("4th-Down Success • With Team Coefficients")
tabs = st.tabs(["🗂 Gather Data", "🔧 Compute & Train", "📊 Visualize"])

with tabs[0]:
    years = st.multiselect(
        "Select Season Year(s)",
        options=list(range(2000, 2025)),
        default=[2022],
        key="with_years"
    )
    weeks = st.multiselect(
        "Select Week(s)",
        options=list(range(1, 16)),
        default=[1],
        key="with_weeks"
    )
    if st.button("📥 Gather Data", key="gather_with"):
        drives = [u.fetch_drives(y, w) for y in years for w in weeks]
        plays  = [u.fetch_plays(y, w)   for y in years for w in weeks]

        df_drives = pd.concat(drives, ignore_index=True)
        df_plays  = pd.concat(plays,  ignore_index=True)

        st.session_state.drives     = df_drives
        st.session_state.plays_with = df_plays

        st.success(f"Fetched {len(df_drives):,} drives & {len(df_plays):,} plays.")
        st.download_button("⬇️ Download Drives CSV", df_drives.to_csv(index=False), "drives.csv", "text/csv")
        st.download_button("⬇️ Download Plays CSV",  df_plays.to_csv(index=False),  "plays.csv",  "text/csv")

        st.subheader("Collected Drives")
        st.dataframe(df_drives)
        st.subheader("Collected Plays")
        st.dataframe(df_plays)

with tabs[1]:
    if "drives" in st.session_state and "plays_with" in st.session_state:
        min_plays = 30
        counts    = st.session_state.drives["offense_idx"].value_counts()
        eligible  = counts[counts >= min_plays]

        if eligible.empty:
            st.error(f"No teams have ≥{min_plays} drives. Please select more data.")
        else:
            if st.button("▶️ Compute Coeffs & Train", key="run_with"):
                coeffs  = u.compute_team_coefficients(st.session_state.drives, min_plays)
                df_pred = u.train_with_coeff_model(st.session_state.plays_with, coeffs)

                st.session_state.coeffs       = coeffs
                st.session_state.df_with_pred = df_pred

                st.success("Model trained with team coefficients!")
                st.download_button("⬇️ Download Team Coeffs CSV",   coeffs.to_csv(index=False),   "team_coeffs.csv",   "text/csv")
                st.download_button("⬇️ Download Predictions CSV", df_pred.to_csv(index=False), "predictions.csv",   "text/csv")

                st.subheader("Team Coefficients")
                st.dataframe(coeffs)
                st.subheader("Raw Predictions")
                st.dataframe(df_pred)
    else:
        st.info("Please gather data in Step 1 above.")

with tabs[2]:
    if "df_with_pred" not in st.session_state:
        st.info("Please run the model in Step 2 above.")
    else:
        dfv = st.session_state.df_with_pred

        # play-level vs team-level detection
        if "success_prob" in dfv.columns or "predicted_prob" in dfv.columns:
            prob_col = "success_prob" if "success_prob" in dfv.columns else "predicted_prob"

            st.subheader("Predicted Success Probability Distribution")
            fig1, ax1 = plt.subplots()
            ax1.hist(dfv[prob_col], bins=20)
            ax1.set_xlabel("Predicted Success Probability")
            ax1.set_ylabel("Number of Plays")
            st.pyplot(fig1)

            plot_df = (
                dfv.groupby("offense")
                   .agg(
                     Actual    = ("success",    "mean"),
                     Predicted = (prob_col,     "mean")
                   )
                   .reset_index()
            )
            plot_df["Actual"]    *= 100
            plot_df["Predicted"] *= 100
        else:
            plot_df = dfv.copy()
            if "ActualRate" in plot_df.columns:
                plot_df["Actual"]    = plot_df["ActualRate"].str.rstrip("%").astype(float)
                plot_df["Predicted"] = plot_df["PredRate"].str.rstrip("%").astype(float)

        st.subheader("Team Conversion: Actual vs Predicted")
        fig2, ax2 = plt.subplots(figsize=(8,4))
        x = np.arange(len(plot_df))
        ax2.plot(x, plot_df["Actual"],    marker="o", linestyle="-",  label="Actual")
        ax2.plot(x, plot_df["Predicted"], marker="o", linestyle="--", label="Predicted")
        ax2.set_xticks(x)
        ax2.set_xticklabels(plot_df["offense"], rotation=90, fontsize=7)
        ax2.set_ylabel("Conversion Rate (%)")
        ax2.legend()
        st.pyplot(fig2)