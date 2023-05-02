import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import plotly.express as px

# Load the data
life_satisfaction = pd.read_csv("BLI_02052023142645388.csv", thousands=',')
gdp_per_capita = pd.read_csv("WEO_Data.csv", thousands=',', na_values="n/a")

#print(life_satisfaction)

life_satisfaction = life_satisfaction[~life_satisfaction.Inequality.str.contains("Total") == False]
life_satisfaction = life_satisfaction.drop(columns=["LOCATION", "INDICATOR", "Indicator", "MEASURE", "Measure", "INEQUALITY", "Inequality", "Unit Code", "Unit", "PowerCode Code", "PowerCode", "Reference Period Code", "Reference Period", "Flag Codes", "Flags"], index=27)
life_satisfaction = life_satisfaction.sort_values('Country')
#print(life_satisfaction)

#print(gdp_per_capita)

gdp_per_capita = gdp_per_capita[gdp_per_capita.Country.isin(life_satisfaction.Country)]
#print(gdp_per_capita)
gdp_per_capita = gdp_per_capita.drop(columns=["Subject Descriptor", "Units", "Scale", "Country/Series-specific Notes", "Estimates Start After"])
#print(gdp_per_capita)
#print(life_satisfaction)

new_df = life_satisfaction.merge(gdp_per_capita)
new_df = new_df.rename(columns={"2020": "GDP per Capita", "Value": "Life Satisfaction"})
#print(new_df)
X = np.c_[new_df["GDP per Capita"]]
y = np.c_[new_df["Life Satisfaction"]]


fig = px.scatter(new_df,y="GDP per Capita",x="Life Satisfaction", hover_data=["Country"])
fig.show()

model = sklearn.linear_model.LinearRegression()

model.fit(X, y)

X_new = [[28559.985]] # Cyprus gdp per capita
print(model.predict(X_new)) # Its actual life satisfaction is 6.22