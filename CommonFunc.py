import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import statistics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def draw_must_box(axs, axs2, df, column_name):
    ar = df[column_name]
    axs.boxplot(ar)
    axs.title.set_text(column_name)
    N, bins, patches = axs2.hist(ar, bins=50, orientation="horizontal")
    axs2.axhline(statistics.median(ar), color='k', linestyle='dashed', linewidth=1)
    axs2.title.set_text(column_name)
    
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()
    
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)


def printDict(dictionary)->str:
    string = ''
    for k, v in dictionary.items():
        string += k.__str__() + '   ' + v.__str__() + '\n'
    return string


def getMetrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'*****************')
    print(f'Метрики качества:')
    print(f'*****************')
    print(f'MAE: {mae:.10f}')
    print(f'MSE: {mse:.10f}')
    print(f'RMSE: {rmse:.10f}')
    print(f'R²: {r2:.10f}')
    
    # Визуализация предсказанных и фактических значений
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Фактические значения')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Предсказанные значения')
    plt.xlabel('Наблюдение')
    plt.ylabel('Значение')
    plt.title('Фактические и предсказанные значения')
    plt.legend()
    plt.show()

    return (mae, mse, rmse, r2)


def getImportances(importances, feature_names):
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Важность признаков')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.show()


def getRemains(y_test, y_pred):
    # 13. График остатков
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='orange')
    plt.title('Распределение остатков')
    plt.xlabel('Остаток (Target - Предсказание)')
    plt.ylabel('Частота')
    plt.show()
    
    # 14. Остатки vs Предсказанные значения
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='green', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('Остатки vs Предсказанные значения')
    plt.show()
