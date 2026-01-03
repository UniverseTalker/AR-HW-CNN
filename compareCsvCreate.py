import pandas as pd

df_ml = pd.read_csv("ml_models_results.csv")
df_cnn = pd.read_csv("cnn_results.csv")  

df_all = pd.concat([df_ml, df_cnn], ignore_index=True)
df_all.to_csv("model_comparison.csv", index=False)
df_all