import pandas as p
import matplotlib.pyplot as plt
import seaborn as s
import pingouin as pg
import numpy as np
from scipy import stats

def tukey(field:str, file:str):
    try:
        data = p.read_csv(f"D:\\STUDY\\LVL3\\Интеллектуальный анализ данных\\lr1\\{file}")
        height_data = data[field]

        plt.figure(figsize=(8, 6))
        plt.boxplot(height_data, vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        #patch_artist - включить возможность изменения внешнего вида
        #boxprops = dict(facecolor='lightgreen') - изменить параметры внешнего вида ящика

        plt.title(f'Диаграмма Тьюки для {field}', fontsize=12)
        plt.show()
    except Exception as e:
        print (f"Ошибка: {e}")
        return
def make_heatmap(fields:list, file:str):
    try:
        data = p.read_csv(f"D:\\STUDY\\LVL3\\Интеллектуальный анализ данных\\lr1\\{file}")
        cormat = data[fields].corr()

        plt.figure(figsize=(10, 8))
        s.heatmap(cormat, annot=True, cmap='coolwarm', fmt='.2f', square=True)

        plt.title('Корреляционная матрица', fontsize=16)
        plt.show()
    except Exception as e:
        print(f"Ошибка: {e}")

def pheatmap(fields:list, file:str):
    try:
        data = p.read_csv(f"D:\\STUDY\\LVL3\\Интеллектуальный анализ данных\\lr1\\{file}")
        paircorr = data[fields].corr()
        partcorr = pg.pcorr(data[fields])

        fig, ax = plt.subplots(1, 2, figsize=(14, 6)) #две тепловых карты на одном поле

        s.heatmap(paircorr, annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax[0])
        ax[0].set_title('Парные коэффициенты корреляции', fontsize=16)

        s.heatmap(partcorr, annot=True, cmap='coolwarm', fmt='.2f',
                    square=True, ax=ax[1])
        ax[1].set_title('Частные коэффициенты корреляции', fontsize=16)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Ошибка: {e}")

def KolmSmir(fields:list, file:str):
    try:
        data = p.read_csv(f"D:\\STUDY\\LVL3\\Интеллектуальный анализ данных\\lr1\\{file}")
        N = data.shape[0]
        D_CRIT = 1.36 / np.sqrt(N)  # критическое значение

        results = []
        for field in fields:
            ks, _ = stats.kstest(data[field], 'norm', args=(np.mean(data[field]), np.std(data[field], ddof=1)))
            result = {
                "Признак": field,
                "KS-статистика": ks,
                "Критическое значение": D_CRIT,
                "Результат": "Нулевая гипотеза не отвергается" if ks < D_CRIT else "Нулевая гипотеза отвергается"
            }
            results.append(result)
        return results
    except Exception as e:
        print(f"Ошибка: {e}")