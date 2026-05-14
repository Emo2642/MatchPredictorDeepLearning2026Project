import os
import datetime as dt

import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

import match_predictor as mp


st.set_page_config(page_title="Match Predictor", layout="wide")


@st.cache_resource
def prepare_pipeline(epochs: int, batch_size: int, force_retrain: bool):
    df = mp.load_data()
    feature_df, y_class, y_goals = mp.build_feature_matrix(df)
    n_timesteps = min(5, max(1, len(df) // 20))
    data = mp.preprocess_data(feature_df, y_class, y_goals, n_timesteps=n_timesteps)

    model_path = "dnn_model.h5"
    lstm_path = "lstm_model.h5"
    if (not force_retrain) and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        model, _ = mp.train_dnn(data, epochs=epochs, batch_size=batch_size)
        model.save(model_path)

    if (not force_retrain) and os.path.exists(lstm_path):
        lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
    else:
        lstm_model, _ = mp.train_lstm(data, epochs=epochs, batch_size=batch_size)
        lstm_model.save(lstm_path)

    return df, feature_df, data, model, lstm_model


st.title("Football Match Prediction")

st.sidebar.header("Training")
use_defaults = st.sidebar.checkbox("Use default training settings", value=True)
if use_defaults:
    epochs = 50
    batch_size = 32
else:
    epochs = st.sidebar.slider("Epochs", min_value=5, max_value=200, value=50, step=5)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
force_retrain = st.sidebar.checkbox("Force retrain models", value=False)
if st.sidebar.button("Delete saved models"):
    for path in ["dnn_model.h5", "lstm_model.h5"]:
        if os.path.exists(path):
            os.remove(path)
    st.cache_resource.clear()
    st.success("Saved models deleted. Please reload to retrain.")

with st.spinner("Loading data and preparing model..."):
    df, feature_df, data, model, lstm_model = prepare_pipeline(
        epochs, batch_size, force_retrain
    )

team_source = df
if "Division" in df.columns:
    divisions = sorted(df["Division"].dropna().astype(str).unique().tolist())
    selected_divisions = st.multiselect("League (Division)", divisions, default=divisions)
    if selected_divisions:
        team_source = df[df["Division"].astype(str).isin(selected_divisions)]

teams = sorted(
    set(team_source["home_team_api_id"].astype(str)).union(
        set(team_source["away_team_api_id"].astype(str))
    )
)

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=0)
with col2:
    away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)

match_date = st.date_input(
    "Match Date",
    value=dt.date.today(),
    min_value=dt.date(1990, 1, 1),
    max_value=dt.date(2035, 12, 31),
)

if st.button("Predict"):
    if home_team == away_team:
        st.warning("Home and Away teams must be different.")
    else:
        elo_df = None
        if mp.USE_ELO_RATINGS and os.path.exists(mp.ELO_RATINGS_PATH):
            try:
                elo_df = mp.load_elo_ratings(mp.ELO_RATINGS_PATH)
            except ValueError as exc:
                st.warning(str(exc))

        feature_row = mp.build_single_feature_row(
            df,
            feature_df.columns.tolist(),
            home_team,
            away_team,
            pd.Timestamp(match_date),
            elo_df=elo_df,
        )
        X_single = data["scaler"].transform(feature_row.values.astype(float))

        dnn_preds = model.predict(X_single, verbose=0)
        dnn_prob = dnn_preds[0][0]
        dnn_goals = dnn_preds[1][0]

        Xl_single = np.repeat(
            X_single[np.newaxis, :, :], data["n_timesteps"], axis=1
        )
        lstm_preds = lstm_model.predict(Xl_single, verbose=0)
        lstm_prob = lstm_preds[0][0]
        lstm_goals = lstm_preds[1][0]

        top_dnn = mp.CLASS_NAMES[int(dnn_prob.argmax())]
        top_lstm = mp.CLASS_NAMES[int(lstm_prob.argmax())]

        st.subheader("Summary")
        st.write(f"DNN top result: {top_dnn}")
        st.write(f"LSTM top result: {top_lstm}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("DNN Probabilities")
            for label, p in zip(mp.CLASS_NAMES, dnn_prob):
                st.write(f"{label}: {p:.3f}")
            st.subheader("DNN Expected Goals")
            st.write(f"Home: {dnn_goals[0]:.2f}")
            st.write(f"Away: {dnn_goals[1]:.2f}")

        with col_b:
            st.subheader("LSTM Probabilities")
            for label, p in zip(mp.CLASS_NAMES, lstm_prob):
                st.write(f"{label}: {p:.3f}")
            st.subheader("LSTM Expected Goals")
            st.write(f"Home: {lstm_goals[0]:.2f}")
            st.write(f"Away: {lstm_goals[1]:.2f}")
