import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u

st.set_page_config(page_title="Compare 4th-Down", layout="wide")

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

st.title("4th-Down Conversion: No Coeff vs With Coeff")

years = st.multiselect("Select Season Year(s)", list(range(2000, 2025)), default=[2022])
weeks = st.multiselect("Select Week(s)",             list(range(1, 16)),    default=[1])

if st.button("🚀 Generate Comparison"):

    drives_list, plays_list = [], []
    for yr in years:
        for wk in weeks:
            drives_list.append(u.fetch_drives(yr, wk))
            plays_list.append(u.fetch_plays(yr, wk))
    df_drives = pd.concat(drives_list, ignore_index=True)
    df_plays  = pd.concat(plays_list,  ignore_index=True)
    st.write(f"**Fetched** {len(df_drives):,} drives • {len(df_plays):,} plays")

    df_no = u.train_no_coeff_model(df_plays)
    team_no = (
        df_no.groupby("offense")
             .agg(
                 Attempts=("success","count"),
                 ActualRate=("success","mean"),
                 PredRate=("success_prob","mean")
             )
             .reset_index()
    )
    team_no["Actual"]    = team_no["ActualRate"] * 100
    team_no["Predicted"] = team_no["PredRate"]   * 100
    team_no = team_no.sort_values("Actual", ascending=False).reset_index(drop=True)

    coeffs  = u.compute_team_coefficients(df_drives)
    df_with = u.train_with_coeff_model(df_plays, coeffs)
    df_with = df_with.assign(
        Actual    = df_with["ActualRate"].str.rstrip("%").astype(float),
        Predicted = df_with["PredRate"].  str.rstrip("%").astype(float)
    ).sort_values("Actual", ascending=False).reset_index(drop=True)

    x_no   = np.arange(len(team_no))
    x_with = np.arange(len(df_with))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5), sharey=True)

    ax1.plot(x_no,   team_no["Actual"],    marker="o", linestyle="-",  label="Actual")
    ax1.plot(x_no,   team_no["Predicted"], marker="o", linestyle="--", label="Predicted")
    ax1.set_xticks(x_no)
    ax1.set_xticklabels(team_no["offense"], rotation=90, fontsize=8)
    ax1.set_title("WITHOUT Team Coeffs")
    ax1.set_ylabel("Conversion Rate (%)")
    ax1.legend()

    ax2.plot(x_with, df_with["Actual"],    marker="o", linestyle="-",  label="Actual")
    ax2.plot(x_with, df_with["Predicted"], marker="o", linestyle="--", label="Predicted")
    ax2.set_xticks(x_with)
    ax2.set_xticklabels(df_with["offense"], rotation=90, fontsize=8)
    ax2.set_title("WITH Team Coeffs")
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ranked: Without Coeffs")
        st.dataframe(
            team_no[["offense","Actual","Predicted"]]
            .rename(columns={"offense":"Team"})
            .style.format({"Actual":"{:.2f}%","Predicted":"{:.2f}%"})
        )
    with col2:
        st.subheader("Ranked: With Coeffs")
        st.dataframe(
            df_with[["offense","Actual","Predicted"]]
            .rename(columns={"offense":"Team"})
            .style.format({"Actual":"{:.2f}%","Predicted":"{:.2f}%"})
        )