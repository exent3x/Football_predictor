import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import poisson
from datetime import datetime, timedelta

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

API_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "5363d07a1ecc477cbfe776b6f3a2cdaf")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_TOKEN}

COMPETITIONS = {
    "PL": "English Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "UEFA Champions League"
}

DAYS_BACK = 180
HALF_LIFE_DAYS = 45

def fetch_recent_matches(competition_code, days_back=DAYS_BACK):
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")

    params = {
        "status": "FINISHED",
        "dateFrom": date_from,
        "dateTo": date_to
    }

    all_matches = []

    while url:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params if url == f"{BASE_URL}/competitions/{competition_code}/matches" else None
        )
        response.raise_for_status()
        data = response.json()

        for m in data["matches"]:
            if m["score"]["fullTime"]["home"] is not None:
                all_matches.append({
                    "home_team": m["homeTeam"]["shortName"],
                    "away_team": m["awayTeam"]["shortName"],
                    "home_goals": m["score"]["fullTime"]["home"],
                    "away_goals": m["score"]["fullTime"]["away"],
                    "date": m["utcDate"][:10]
                })

        url = data.get("links", {}).get("next", {}).get("href")
        time.sleep(0.5)

    df = pd.DataFrame(all_matches)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        today = pd.Timestamp.now().normalize()
        df["days_ago"] = (today - df["date"]).dt.days
        df["weight"] = 0.5 ** (df["days_ago"] / HALF_LIFE_DAYS)
        df["weight"] = df["weight"] / df["weight"].mean()

    return df

def prepare_model_data(df):
    home = df[["home_team", "away_team", "home_goals", "weight"]].copy()
    home.columns = ["team", "opponent", "goals", "weight"]
    home["home"] = 1

    away = df[["away_team", "home_team", "away_goals", "weight"]].copy()
    away.columns = ["team", "opponent", "goals", "weight"]
    away["home"] = 0

    return pd.concat([home, away], ignore_index=True)

def train_poisson_model(df):
    model_df = prepare_model_data(df)
    formula = "goals ~ home + C(team) + C(opponent)"

    model = smf.glm(
        formula,
        data=model_df,
        family=sm.families.Poisson()
    ).fit(freq_weights=model_df["weight"])

    print("\n=== Poisson GLM: параметры (с весами) ===")
    print(model.summary2().tables[1][["Coef.", "P>|z|"]])
    return model

def poisson_probs(model, home_team, away_team):
    params = model.params

    home_adv = params.get("home", 0)
    intercept = params.get("Intercept", 0)

    att_h = params.get(f"C(team)[T.{home_team}]", 0)
    att_a = params.get(f"C(team)[T.{away_team}]", 0)
    def_h = params.get(f"C(opponent)[T.{home_team}]", 0)
    def_a = params.get(f"C(opponent)[T.{away_team}]", 0)

    log_lam_h = intercept + home_adv + att_h + def_a
    log_lam_a = intercept + att_a + def_h

    lam_h = np.exp(log_lam_h)
    lam_a = np.exp(log_lam_a)

    max_g = 10
    home_win = draw = away_win = 0.0

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p

    total = home_win + draw + away_win
    probs = np.array([home_win, draw, away_win]) / total

    return probs, lam_h, lam_a

def make_features(df):
    X = pd.get_dummies(
        df[["home_team", "away_team"]],
        columns=["home_team", "away_team"],
        prefix=["home", "away"]
    )
    return X

def make_target(df):
    return np.where(
        df.home_goals > df.away_goals, 0,
        np.where(df.home_goals == df.away_goals, 1, 2)
    )

def train_ml_models(df):
    X = make_features(df)
    y = make_target(df)
    weights = df["weight"].values

    if len(np.unique(y)) < 2:
        raise ValueError("Слишком мало классов в исходе матчей для обучения ML.")

    stratify = y if min(np.bincount(y)) >= 2 else None

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))
        ]),
        "rf_clf": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        ),
        "gbc": GradientBoostingClassifier(
            random_state=42
        )
    }

    print("\n=== ML-классификаторы 1X2 (с весами) ===")
    models["logreg"].fit(X_train, y_train, clf__sample_weight=w_train)
    models["rf_clf"].fit(X_train, y_train, sample_weight=w_train)
    models["gbc"].fit(X_train, y_train, sample_weight=w_train)

    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        print(f"{name}: acc={accuracy_score(y_test, pred):.3f}, logloss={log_loss(y_test, proba):.3f}")

    rf_home = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    rf_away = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    rf_home.fit(X_train, df.loc[X_train.index, "home_goals"], sample_weight=w_train)
    rf_away.fit(X_train, df.loc[X_train.index, "away_goals"], sample_weight=w_train)

    return {
        "logreg": models["logreg"],
        "rf_clf": models["rf_clf"],
        "gbc": models["gbc"],
        "rf_home": rf_home,
        "rf_away": rf_away
    }, X.columns

def ml_probs(ml_models, columns, home_team, away_team):
    row = pd.DataFrame([{
        "home_team": home_team,
        "away_team": away_team
    }])

    X_row = make_features(row).reindex(columns=columns, fill_value=0)

    probas = []
    for name in ["logreg", "rf_clf", "gbc"]:
        probas.append(ml_models[name].predict_proba(X_row)[0])
    p_ml = np.mean(probas, axis=0)

    lam_h = ml_models["rf_home"].predict(X_row)[0]
    lam_a = ml_models["rf_away"].predict(X_row)[0]

    return p_ml, lam_h, lam_a

def show_teams(competition_code):
    df = fetch_recent_matches(competition_code, DAYS_BACK)
    if df.empty:
        print("Нет матчей.")
        return
    teams = sorted(set(df.home_team) | set(df.away_team))
    print(f"Команды в {competition_code}:")
    for t in teams:
        print(f"  - {t}")

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1].lower() == "teams":
        show_teams(sys.argv[2].upper())
        sys.exit(0)

    if len(sys.argv) != 4:
        print('Использование: python football_predictor.py "Barça" "Real Betis" PD')
        print('Список команд: python football_predictor.py teams PD')
        print(f'Доступные турниры: {", ".join(COMPETITIONS.keys())}')
        sys.exit(1)

    home_team_input = sys.argv[1]
    away_team_input = sys.argv[2]
    competition_code = sys.argv[3].upper()

    if competition_code not in COMPETITIONS:
        print(f"Неизвестный турнир: {competition_code}")
        print(f'Доступные турниры: {", ".join(COMPETITIONS.keys())}')
        sys.exit(1)

    print(f"Загрузка данных {COMPETITIONS[competition_code]} за {DAYS_BACK} дней...")
    df_matches = fetch_recent_matches(competition_code, DAYS_BACK)

    if df_matches.empty:
        sys.exit("Нет завершённых матчей за выбранный период.")

    print(f"Собрано матчей: {len(df_matches)}")
    print(f"Диапазон дат: {df_matches['date'].min().date()} – {df_matches['date'].max().date()}")
    print(f"Веса: min={df_matches['weight'].min():.3f}, max={df_matches['weight'].max():.3f}")

    teams_available = set(df_matches.home_team) | set(df_matches.away_team)
    if home_team_input not in teams_available:
        print(f"⚠ Команда '{home_team_input}' не найдена в данных.")
        print(f"Доступные: {sorted(teams_available)}")
        sys.exit(1)
    if away_team_input not in teams_available:
        print(f"⚠ Команда '{away_team_input}' не найдена в данных.")
        print(f"Доступные: {sorted(teams_available)}")
        sys.exit(1)

    poisson_model = train_poisson_model(df_matches)
    ml_models, ml_columns = train_ml_models(df_matches)

    p_poisson, lam_h_p, lam_a_p = poisson_probs(
        poisson_model, home_team_input, away_team_input
    )

    p_ml, lam_h_ml, lam_a_ml = ml_probs(
        ml_models, ml_columns, home_team_input, away_team_input
    )

    w_poisson = 0.5
    w_ml = 0.5
    p_final = w_poisson * p_poisson + w_ml * p_ml
    p_final = p_final / p_final.sum()

    print(f"\n=== Прогноз [{competition_code}]: {home_team_input} vs {away_team_input} ===")

    print("\nPoisson GLM (с весами):")
    print(f"  Ожидаемые голы: {lam_h_p:.2f} – {lam_a_p:.2f}")
    print(f"  1: {p_poisson[0]:.1%} | X: {p_poisson[1]:.1%} | 2: {p_poisson[2]:.1%}")

    print("\nML (усреднение LogReg + RF + GBC, с весами):")
    print(f"  Ожидаемые голы: {lam_h_ml:.2f} – {lam_a_ml:.2f}")
    print(f"  1: {p_ml[0]:.1%} | X: {p_ml[1]:.1%} | 2: {p_ml[2]:.1%}")

    print("\nФинальный ансамбль 50/50:")
    print(f"  1: {p_final[0]:.1%} | X: {p_final[1]:.1%} | 2: {p_final[2]:.1%}")