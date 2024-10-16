# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
import pandas as pd\
import numpy as np\
import matplotlib.pyplot as plt\
from statsmodels.tsa.ar_model import AutoReg\
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('/content/student_performance.csv')\
df = pd.DataFrame(data)

df.set_index('StudentID', inplace=True)\
plt.plot(df.index, df['FinalGrade'], marker='o')\
plt.title('Final Grades Over Student IDs')\
plt.xlabel('Student ID')\
plt.ylabel('Final Grade')\
plt.grid(True)\
plt.show()\
ar_model = AutoReg(df['FinalGrade'], lags=2).fit()\
ar_predictions = ar_model.predict(start=len(df), end=len(df) + 4)

print("AR Model Predictions for next 5 students:\n", ar_predictions)\
plt.plot(df.index, df['FinalGrade'], label='Original Final Grades', marker='o')\
plt.plot(range(len(df) + 1, len(df) + 6), ar_predictions, label='AR Forecasted Grades', marker='x')\
plt.title('Auto-Regressive Model Forecast')\
plt.xlabel('Student ID')\
plt.ylabel('Final Grade')\
plt.legend()\
plt.grid(True)\
plt.show()\
es_model = ExponentialSmoothing(df['FinalGrade'], trend='add', seasonal=None).fit()

es_predictions = es_model.forecast(steps=5)\
print("Exponential Smoothing Predictions for next 5 students:\n", es_predictions)

plt.plot(df.index, df['FinalGrade'], label='Original Final Grades', marker='o')\
plt.plot(df.index, es_model.fittedvalues, label='ES Fitted Values', linestyle='--')\
plt.plot(range(len(df) + 1, len(df) + 6), es_predictions, label='ES Forecasted Grades', marker='x')\
plt.title('Exponential Smoothing Forecast')\
plt.xlabel('Student ID')\
plt.ylabel('Final Grade')\
plt.legend()\
plt.grid(True)\
plt.show()
### OUTPUT:

GIVEN DATA

![Screenshot 2024-10-16 102110](https://github.com/user-attachments/assets/c79d923c-8bc3-4235-842f-41d47a8be319)

PREDICTION

![Screenshot 2024-10-16 102121](https://github.com/user-attachments/assets/20682145-dd1f-4f07-9de5-25731f090b7d)

FINIAL PREDICTION

![Screenshot 2024-10-16 102158](https://github.com/user-attachments/assets/85897f85-4c2f-4844-b354-ed142a8b8758)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
