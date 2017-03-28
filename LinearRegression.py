import pandas as pd
from sklearn.linear_model import LinearRegression


bmi_life_model = LinearRegression()
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
y_values = bmi_life_data[['Life expectancy']]
x_values = bmi_life_data[['BMI']]
bmi_life_model.fit(x_values, y_values)


laos_life_exp = bmi_life_model.predict(21.07931)


print(laos_life_exp)