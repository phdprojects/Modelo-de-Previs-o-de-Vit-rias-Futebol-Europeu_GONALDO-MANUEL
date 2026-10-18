# Databricks notebook source
# DBTITLE 1,Modelo de Previsão de Vitórias
# MAGIC %md
# MAGIC # Modelo de Previsão de Vitórias Futebol Europeu | GONALDO MANUEL 23092026
# MAGIC
# MAGIC Modelo de classificação multiclasse (Vitória Casa / Empate / Vitória Fora) baseado em **features pré-match**: odds das casas de apostas + forma recente das equipas (rolling averages dos últimos 5 jogos).
# MAGIC
# MAGIC **Atenção à fuga de dados (data leakage):** apenas se usam features disponíveis **antes** do jogo odds pré-match e estatísticas históricas. Nenhuma estatística do próprio jogo (golos, remates, cartões) é usada como feature.

# COMMAND ----------

# DBTITLE 1,Imports e Carregamento de Dados (4 Épocas)
# MAGIC %pip install xgboost
# MAGIC
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC from pyspark.sql import functions as F
# MAGIC import mlflow
# MAGIC import mlflow.sklearn
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.compose import ColumnTransformer
# MAGIC from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
# MAGIC from sklearn.impute import SimpleImputer
# MAGIC from sklearn.ensemble import RandomForestClassifier
# MAGIC from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# MAGIC from xgboost import XGBClassifier
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC
# MAGIC # Carregar dados da camada Bronze (4 épocas: 2020/21 a 2023/24)
# MAGIC # Selecionar apenas as colunas necessárias para feature engineering
# MAGIC df = spark.table("bronze_european_football_master").select(
# MAGIC     'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
# MAGIC     'AvgH', 'AvgD', 'AvgA', 'league_name', 'season'
# MAGIC )
# MAGIC df_pd = df.toPandas()
# MAGIC print(f"Total de jogos: {len(df_pd)}")
# MAGIC print(f"Épocas: {df_pd['season'].unique()}")
# MAGIC print(f"\nDistribuição do alvo (FTR):")
# MAGIC print(df_pd['FTR'].value_counts())
# MAGIC print(f"\nColunas selecionadas: {len(df_pd.columns)} (de 109 disponíveis)")

# COMMAND ----------

# DBTITLE 1,Feature Engineering — Forma + H2H + Casa/Fora
# --- FEATURE ENGINEERING: Features pré-match avançadas (sem data leakage) ---
# 1. Odds das casas de apostas (disponíveis antes do jogo)
# 2. Forma geral recente (rolling averages dos últimos 5 jogos)
# 3. Forma específica casa/fora (rolling averages separados)
# 4. Head-to-Head (histórico de confrontos diretos)

# Converter Data para datetime e ordenar
df_pd['Date'] = pd.to_datetime(df_pd['Date'], dayfirst=True, errors='coerce')
df_pd = df_pd.sort_values('Date').reset_index(drop=True)

# === 1. Forma geral (rolling averages dos últimos 5 jogos) ===
home_stats = pd.DataFrame({
    'Date': df_pd['Date'], 'team': df_pd['HomeTeam'],
    'goals_scored': df_pd['FTHG'], 'goals_conceded': df_pd['FTAG'],
    'points': df_pd['FTR'].map({'H': 3, 'D': 1, 'A': 0}),
    'is_home': 1, 'match_idx': df_pd.index
})
away_stats = pd.DataFrame({
    'Date': df_pd['Date'], 'team': df_pd['AwayTeam'],
    'goals_scored': df_pd['FTAG'], 'goals_conceded': df_pd['FTHG'],
    'points': df_pd['FTR'].map({'A': 3, 'D': 1, 'H': 0}),
    'is_home': 0, 'match_idx': df_pd.index
})

team_history = pd.concat([home_stats, away_stats]).sort_values(['team', 'Date']).reset_index(drop=True)

# Forma geral (últimos 5 jogos, excluindo o atual)
for col, name in [('goals_scored', 'form_goals_scored'), ('goals_conceded', 'form_goals_conceded'), ('points', 'form_points')]:
    team_history[name] = team_history.groupby('team')[col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

# Forma específica casa/fora (últimos 5 jogos, apenas em casa ou apenas fora)
for is_home_val, suffix in [(1, 'home'), (0, 'away')]:
    mask = team_history['is_home'] == is_home_val
    for col, base in [('goals_scored', 'goals_scored'), ('goals_conceded', 'goals_conceded'), ('points', 'points')]:
        team_history.loc[mask, f'{suffix}_spec_{base}'] = (
            team_history[mask].groupby('team')[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

# Merge forma geral de volta ao df principal
def merge_form(team_stats_df, prefix):
    form_df = team_stats_df[['match_idx', 'team']].merge(
        team_history[['match_idx', 'team', 'form_goals_scored', 'form_goals_conceded', 'form_points',
                      'home_spec_goals_scored', 'home_spec_goals_conceded', 'home_spec_points',
                      'away_spec_goals_scored', 'away_spec_goals_conceded', 'away_spec_points']],
        on=['match_idx', 'team'], how='left'
    )
    form_df.columns = ['match_idx', 'team'] + [f'{prefix}_{c}' for c in
        ['form_goals_scored', 'form_goals_conceded', 'form_points',
         'home_spec_goals_scored', 'home_spec_goals_conceded', 'home_spec_points',
         'away_spec_goals_scored', 'away_spec_goals_conceded', 'away_spec_points']]
    return form_df.drop(columns=['team'])

home_form_final = merge_form(home_stats, 'home')
away_form_final = merge_form(away_stats, 'away')

df_pd['match_idx'] = df_pd.index
df_model = df_pd.merge(home_form_final, on='match_idx', how='left')
df_model = df_model.merge(away_form_final, on='match_idx', how='left')

# === 2. Head-to-Head (confrontos diretos) ===
# Para cada jogo, calcular pontos do HomeTeam nos últimos 5 H2H vs este adversário
df_pd_sorted = df_pd.sort_values('Date').reset_index(drop=True)
h2h_records = []

for idx, row in df_pd_sorted.iterrows():
    home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
    # Jogos anteriores entre as mesmas duas equipas (qualquer ordem)
    prev = df_pd_sorted[
        (df_pd_sorted['Date'] < date) &
        (((df_pd_sorted['HomeTeam'] == home) & (df_pd_sorted['AwayTeam'] == away)) |
         ((df_pd_sorted['HomeTeam'] == away) & (df_pd_sorted['AwayTeam'] == home)))
    ].tail(5)
    
    if len(prev) == 0:
        h2h_records.append({'match_idx': idx, 'h2h_home_points': np.nan, 'h2h_avg_goals': np.nan, 'h2h_games': 0})
    else:
        home_points = []
        goals = []
        pts_home = {'H': 3, 'D': 1, 'A': 0}
        pts_away = {'H': 0, 'D': 1, 'A': 3}
        for _, p in prev.iterrows():
            if p['HomeTeam'] == home:
                home_points.append(pts_home[p['FTR']])
                goals.append(p['FTHG'] + p['FTAG'])
            else:
                home_points.append(pts_away[p['FTR']])
                goals.append(p['FTHG'] + p['FTAG'])
        h2h_records.append({
            'match_idx': idx,
            'h2h_home_points': np.mean(home_points),
            'h2h_avg_goals': np.mean(goals),
            'h2h_games': len(prev)
        })

h2h_df = pd.DataFrame(h2h_records)
df_model = df_model.merge(h2h_df, on='match_idx', how='left')

# === 3. Seleção final de features pré-match ===
feature_cols = [
    # Odds pré-match
    'AvgH', 'AvgD', 'AvgA',
    # Forma geral
    'home_form_goals_scored', 'home_form_goals_conceded', 'home_form_points',
    'away_form_goals_scored', 'away_form_goals_conceded', 'away_form_points',
    # Forma específica casa/fora
    'home_home_spec_goals_scored', 'home_home_spec_goals_conceded', 'home_home_spec_points',
    'away_away_spec_goals_scored', 'away_away_spec_goals_conceded', 'away_away_spec_points',
    # Head-to-Head
    'h2h_home_points', 'h2h_avg_goals', 'h2h_games',
    # Categorical
    'league_name', 'season'
]

target_col = 'FTR'

print(f"Features selecionadas ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  - {f}")
print(f"\nDataset final: {len(df_model)} jogos")
display(df_model[feature_cols + [target_col]].head(10))

# COMMAND ----------

# DBTITLE 1,Treino RandomForest vs XGBoost (4 Épocas)
# --- SPLIT TEMPORAL + TREINO RandomForest vs XGBoost COM MLFLOW ---
# Split temporal: 80% mais antigo para treino, 20% mais recente para teste
df_model = df_model.dropna(subset=[target_col]).sort_values('Date').reset_index(drop=True)
# Imputar NaNs nas features numéricas (manter jogos mesmo sem histórico inicial)
for col in feature_cols:
    if col not in ['league_name', 'season'] and df_model[col].isna().any():
        df_model[col] = df_model[col].fillna(df_model[col].median())

split_idx = int(len(df_model) * 0.8)
X_train = df_model.loc[:split_idx-1, feature_cols]
y_train = df_model.loc[:split_idx-1, target_col]
X_test = df_model.loc[split_idx:, feature_cols]
y_test = df_model.loc[split_idx:, target_col]

print(f"Treino: {len(X_train)} jogos ({df_model.loc[:split_idx-1, 'Date'].min().date()} a {df_model.loc[:split_idx-1, 'Date'].max().date()})")
print(f"Teste:  {len(X_test)} jogos ({df_model.loc[split_idx:, 'Date'].min().date()} a {df_model.loc[split_idx:, 'Date'].max().date()})")
print(f"\nDistribuição treino:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\nDistribuição teste:\n{y_test.value_counts(normalize=True).round(3)}")

# Pipeline de preprocessing
numeric_features = [c for c in feature_cols if c not in ['league_name', 'season']]
categorical_features = ['league_name', 'season']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
])

# Label encoding para XGBoost (requer labels numéricos)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

mlflow.set_experiment("/Users/gonaldomichael@gmail.com/football_match_prediction")

results = {}

# === MODELO 1: RandomForest ===
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'))
])

with mlflow.start_run(run_name="rf_v2_4seasons_h2h"):
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))
    mlflow.log_metric("accuracy", acc_rf)
    from mlflow.models import infer_signature
    signature = infer_signature(X_train.head(100), rf_pipeline.predict(X_train.head(100)))
    mlflow.sklearn.log_model(rf_pipeline, name="rf_model", signature=signature, input_example=X_train.head(3))

results['RandomForest'] = {'pred': y_pred_rf, 'acc': acc_rf, 'pipeline': rf_pipeline}
print(f"✅ RandomForest: {acc_rf:.3f}")

# === MODELO 2: XGBoost ===
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, 
                                  eval_metric='mlogloss', use_label_encoder=False))
])

with mlflow.start_run(run_name="xgb_v1_4seasons_h2h"):
    xgb_pipeline.fit(X_train, y_train_enc)
    y_pred_xgb_enc = xgb_pipeline.predict(X_test)
    y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))
    mlflow.log_metric("accuracy", acc_xgb)
    signature = infer_signature(X_train.head(100), xgb_pipeline.predict(X_train.head(100)))
    mlflow.sklearn.log_model(xgb_pipeline, name="xgb_model", signature=signature, input_example=X_train.head(3))

results['XGBoost'] = {'pred': y_pred_xgb, 'acc': acc_xgb, 'pipeline': xgb_pipeline}
print(f"✅ XGBoost: {acc_xgb:.3f}")

print(f"\n{'='*50}")
print(f"COMPARAÇÃO DE MODELOS")
print(f"{'='*50}")
for name, r in results.items():
    print(f"  {name:15s}: {r['acc']:.3f} ({r['acc']:.1%})")

# COMMAND ----------

# DBTITLE 1,Avaliação Comparativa RF vs XGBoost
# --- AVALIAÇÃO COMPARATIVA: RandomForest vs XGBoost ---
target_names = ['Empate (D)', 'Vitória Casa (H)', 'Vitória Fora (A)']

for name, r in results.items():
    print(f"\n{'='*60}")
    print(f"RELATÓRIO — {name} (Accuracy: {r['acc']:.1%})")
    print(f"{'='*60}")
    print(classification_report(y_test, r['pred'], target_names=target_names))

# Confusion Matrices lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['pred'], labels=['H', 'D', 'A'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Casa (H)', 'Empate (D)', 'Fora (A)'], 
                yticklabels=['Casa (H)', 'Empate (D)', 'Fora (A)'], ax=ax)
    ax.set_xlabel('Previsão', fontweight='bold')
    ax.set_ylabel('Real', fontweight='bold')
    ax.set_title(f'{name} — {r["acc"]:.1%}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Feature Importance comparada
feature_names = numeric_features + list(results['RandomForest']['pipeline']
    .named_steps['preprocessor'].named_transformers_['cat']
    .named_steps['encoder'].get_feature_names_out(categorical_features))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, (name, r) in zip(axes, results.items()):
    importances = r['pipeline'].named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=True).tail(15)
    sns.barplot(data=feat_imp, x='importance', y='feature', hue='feature', palette='viridis', ax=ax, legend=False)
    ax.set_title(f'{name} — Top 15 Features', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importância')
plt.tight_layout()
plt.show()

# Tabela comparativa de feature importance
print("\nTop 10 Features — Comparação:")
rf_imp = pd.DataFrame({'feature': feature_names, 'RF': results['RandomForest']['pipeline'].named_steps['classifier'].feature_importances_})
xgb_imp = pd.DataFrame({'feature': feature_names, 'XGB': results['XGBoost']['pipeline'].named_steps['classifier'].feature_importances_})
compare_imp = rf_imp.merge(xgb_imp, on='feature')
compare_imp['media'] = (compare_imp['RF'] + compare_imp['XGB']) / 2
compare_imp = compare_imp.sort_values('media', ascending=False).head(10)
display(compare_imp)

# COMMAND ----------

# DBTITLE 1,Registo no Unity Catalog
# --- REGISTO DO MODELO NO UNITY CATALOG ---
# O RandomForest foi o modelo vencedor (53.4% accuracy) — vamos registá-lo

from mlflow import MlflowClient

# 1. Encontrar a run do RandomForest
experiment_name = "/Users/gonaldomichael@gmail.com/football_match_prediction"
runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string="tags.mlflow.runName = 'rf_v2_4seasons_h2h'",
    order_by=["attributes.start_time DESC"],
    max_results=1
)
run_id = runs.iloc[0]['run_id']
model_uri = f"runs:/{run_id}/rf_model"
print(f"Run ID: {run_id}")
print(f"Model URI: {model_uri}")

# 2. Validar o artefacto (signature + predict)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
print(f"Signature: {pyfunc_model.metadata.signature is not None}")

representative_input = X_test.head(3)
pred_test = pyfunc_model.predict(representative_input)
print(f"\nPrevisão de validação (3 jogos):")
print(pred_test)

# 3. Registar no Unity Catalog
registered_model_name = "workspace.default.football_match_predictor"
mlflow.set_registry_uri("databricks-uc")

registered_version = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name,
    await_registration_for=300,
)
model_version = registered_version.version

print(f"\n✅ Modelo registado no Unity Catalog!")
print(f"   Nome: {registered_model_name}")
print(f"   Versão: {model_version}")
print(f"   URI: models:/{registered_model_name}/{model_version}")

# 4. Definir alias 'champion' para esta versão
registry_client = MlflowClient(registry_uri="databricks-uc")
registry_client.set_registered_model_alias(
    name=registered_model_name,
    alias="champion",
    version=model_version,
)
print(f"   Alias: champion -> versão {model_version}")