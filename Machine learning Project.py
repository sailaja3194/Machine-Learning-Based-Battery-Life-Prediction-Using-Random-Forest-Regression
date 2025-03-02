import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
data = pd.DataFrame({
    'charge_cycles': np.random.randint(100, 1000, 1000),
    'avg_temperature': np.random.uniform(15, 30, 1000),  
    'avg_speed': np.random.uniform(10, 120, 1000),  
    'regenerative_braking': np.random.uniform(0, 1, 1000), 
    'battery_life_remaining': np.random.uniform(50, 500, 1000) 
})

sns.pairplot(data)
plt.show()

features = ['charge_cycles', 'avg_temperature', 'avg_speed', 'regenerative_braking']
target = 'battery_life_remaining'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

based on above program give me  Books and Tutorials ,Research Papers and Projects
