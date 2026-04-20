import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # для сохранения модели

# 1. Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('/root/.cache/kagglehub/datasets/aayushmishra1512/fifa-2021-complete-player-data/versions/1/FIFA-21 Complete.csv', sep=';')

# 2. Удаляем ненужные колонки
df = df.drop(['player_id', 'name', 'position', 'potential'], axis=1)  # potential исключаем

# 3. Target encoding для team и nationality
def target_encoding_safe(X_train, X_test, y_train, categorical_col, smooth=20):
    temp_df = X_train.copy()
    temp_df['target'] = y_train.values
    category_stats = temp_df.groupby(categorical_col)['target'].agg(['mean', 'count'])
    global_mean = y_train.mean()
    category_stats['encoded'] = (category_stats['mean'] * category_stats['count'] + 
                                  global_mean * smooth) / (category_stats['count'] + smooth)
    X_train[f'{categorical_col}_encoded'] = X_train[categorical_col].map(category_stats['encoded'])
    X_test[f'{categorical_col}_encoded'] = X_test[categorical_col].map(category_stats['encoded'])
    X_test[f'{categorical_col}_encoded'].fillna(global_mean, inplace=True)
    return X_train, X_test

# 4. Разделяем на признаки и целевую переменную
X = df[['age', 'hits', 'team', 'nationality']]
y = df['overall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Кодируем категориальные признаки
X_train, X_test = target_encoding_safe(X_train, X_test, y_train, 'team', smooth=50)
X_train, X_test = target_encoding_safe(X_train, X_test, y_train, 'nationality', smooth=20)

# 6. Удаляем оригинальные категориальные колонки
X_train = X_train.drop(['team', 'nationality'], axis=1)
X_test = X_test.drop(['team', 'nationality'], axis=1)

# 7. Обучаем модель
print("Обучение модели...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Оцениваем
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE модели: {rmse:.3f}")

# 9. Сохраняем модель
joblib.dump(model, 'fifa_model.pkl')
print("Модель сохранена в fifa_model.pkl")

# 10. Важность признаков (опционально)
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nВажность признаков:")
print(importances)