import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson
from task1 import KolmSmir

def linear(file:str, xval:str, xmult, yval:str):
    data_prev = p.read_csv(f"D:\\STUDY\\LVL3\\Интеллектуальный анализ данных\\lr1\\{file}")

    Q1 = np.percentile(data_prev[yval], 25)  # Первый квартиль (25%)
    Q3 = np.percentile(data_prev[yval], 75)  # Третий квартиль (75%)
    IQR = Q3 - Q1

    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR

    # фильтрация данных без выбросов для sepal width
    data = data_prev[(data_prev[yval] >= lb) & (data_prev[yval] <= ub)]

    x = data[[xval]] #вход для парной
    x_mult = data[xmult] #вход для множественной
    y = data[yval]

    x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=0.2, random_state=1) #20 процентов на тест, 80 процентов на обучение. random_state - сид
    x_tr_m, x_test_m = train_test_split(x_mult, test_size=0.2, random_state=1)

    model = LinearRegression() #создается экземпляр модели линейной регрессии
    model.fit(x_tr, y_tr) #обучение линейной модели, где y_tr - выходные параметры, которые мы пытаемся предсказать
    y_pred = model.predict(x_test) #попытка предсказания на данных, которые модель еще не видела

    model_m = LinearRegression() #аналогично для множественной регрессии
    model_m.fit(x_tr_m, y_tr)
    y_pred_m = model_m.predict(x_test_m)

    mse = mean_squared_error(y_test, y_pred) #вычисление mean square error между тестовыми выходными параметрами и тем, что предсказала модель
    r2 = r2_score(y_test, y_pred) #вычисление коэффициента детерминации парной регрессии
    """является мерой того, какую долю вариации выходной переменной модель может объяснить. если равен единице, то идеально справляется
    с предсказаниями. если r2 = n/100, то n вариаций в выходной переменной определяется входными переменными"""

    mse_m = mean_squared_error(y_test, y_pred_m)
    r2_m = r2_score(y_test, y_pred_m)

    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
    adjusted_r2_m = 1 - (1 - r2_m) * (len(y_test) - 1) / (len(y_test) - x_test_m.shape[1] - 1)
    """скорректированный r² помогает избежать переобучения, т.к. если в модели добавляются лишние входные переменные, R² может 
    увеличиваться даже если эти переменные не являются значимыми. скорректированный r² при такой ситуации не изменится или изменится
    незначительно"""
    std_error = np.sqrt(mse) #стандартная ошибка. чем ниже, тем точнее модель
    std_error_m = np.sqrt(mse_m)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_test, y_test, color='blue', label='Тестовая выброка') #точечный график
    plt.scatter(x_test, y_pred, color='red', label='Предсказанные значения')
    plt.title('Парная регрессия')
    plt.xlabel(f"{xval}")
    plt.ylabel(f"{yval}")
    plt.legend()

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred #регрессионные остатки
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='black', linestyle='--') #линия на нуле позволяет понять, какие значения недооценены, а какие переоценены
    plt.title('Регрессионные остатки парной регрессии')
    plt.xlabel('Значения регрессии')
    plt.ylabel('Остатки')
    plt.tight_layout() #автовыравнивание
    plt.show()

    dw_stat = durbin_watson(residuals)
    print(f"Значение статистики Дурбина-Ватсона: {dw_stat}")
    residuals_df = p.DataFrame(residuals.tolist(), columns=['Residuals'])
    residuals_df.to_csv('residuals1.csv', index=False)

    print("Для парной регрессии:")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    print(f"Скорректированное R2: {adjusted_r2}")
    print(f"Стандартная ошибка: {std_error}")

    print("\nДля множественной регрессии:")
    print(f"MSE: {mse_m}")
    print(f"R2: {r2_m}")
    print(f"Скорректированное R2: {adjusted_r2_m}")
    print(f"Стандартная ошибка: {std_error_m}")
