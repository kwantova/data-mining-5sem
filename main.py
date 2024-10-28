import task1 as t1
import task2 as t2

# Бейсболисты
# Диаграмы Тьюки
t1.tukey("Height(inches)", "sample.csv")
t1.tukey("Weight(pounds)", "sample.csv")
t1.tukey("Age", "sample.csv")

# Тепловые карты
t1.make_heatmap(["Height(inches)", "Weight(pounds)", "Age"], "sample.csv")
t1.pheatmap(["Height(inches)", "Weight(pounds)", "Age"], "sample.csv")
output = t1.KolmSmir(["Height(inches)", "Weight(pounds)", "Age"], "sample.csv")
[print(now) for now in output]

# Регрессия
t2.linear("sample.csv", xval="Height(inches)", xmult=["Height(inches)", "Age"], yval="Weight(pounds)")
output = t2.KolmSmir(["Residuals"], "residuals1.csv")
print("Критерий Колмогорова-Смирнова по отношению к остаткам:")
[print(now) for now in output]

# Ирисы
# Тьюки
t1.tukey("sepal length (cm)", "iris.csv")
t1.tukey("sepal width (cm)", "iris.csv")
t1.tukey("petal length (cm)", "iris.csv")
t1.tukey("petal width (cm)", "iris.csv")

# Тепловые карты
t1.make_heatmap(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], "iris.csv")
t1.pheatmap(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], "iris.csv")
output = t1.KolmSmir(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], "iris.csv")
[print(now) for now in output]

# Регрессия
t2.linear("iris.csv", xval='petal length (cm)', xmult=['petal length (cm)', 'petal width (cm)', 'sepal length (cm)'], yval='sepal width (cm)')
output = t1.KolmSmir(["Residuals"], "residuals1.csv")
print("Критерий Колмогорова-Смирнова по отношению к остаткам:")
[print(now) for now in output]

# Chemical process
t1.make_heatmap(["Parameter 1", "Parameter 2"], "outputcp.csv")