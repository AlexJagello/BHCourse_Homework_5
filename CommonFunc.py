import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import statistics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest



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


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def choseAnomalyNotAnomaly(isAnomaly: bool, arr: list, arr_y: list) -> list:
    newlist=[]
    
    for i in range(0,len(arr)):
        if arr_y[i] == 1 and isAnomaly==False:
            newlist.append(arr[i])
        if arr_y[i] == -1 and isAnomaly==True:
            newlist.append(arr[i])
    return newlist
    

def AnomalyOneCategoryVisualisation(xx, yy, name, conts):
    new_X = []  
    amount_contamination = 2.0
    for x,y_ in zip(xx,yy):
        new_X.append([x,y_])

    fig, axs = plt.subplots(1,len(conts))
    fig.set_figheight(4)
    fig.set_figwidth(15)
                    
    for i in range(0, len(conts)):
        clf = IsolationForest(contamination=conts[i], random_state=42) 
        clf.fit(new_X)
        y_pred = clf.predict(new_X)
        
        normal_points = choseAnomalyNotAnomaly(False ,new_X, y_pred)
    
        axs[i].scatter([i[0] for i in normal_points], [i[1] for i in normal_points], c='blue', s=20, edgecolor='k', label="Нормальные точки")
        
        anomalies = choseAnomalyNotAnomaly(True ,new_X, y_pred)
        
        axs[i].scatter([i[0] for i in anomalies], [i[1] for i in anomalies], c='red', s=20, edgecolor='k', label="Аномалии")
        axs[i].title.set_text(name +" " + conts[i].__str__())
        axs[i].legend()


def QuartilesAnomaly(data, name):    
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # Определение границ для выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Поиск выбросов
    outliers = np.where((data < lower_bound) | (data > upper_bound))
    
    # Визуализация данных и выбросов
    plt.figure(figsize=(12, 7))
    plt.plot(data, 'bo', label='Данные')
    plt.plot(outliers[0], data[outliers], 'ro', label='Выбросы')
    plt.axhline(Q1, color='orange', linestyle='dashed', linewidth=2, label='Q1 (Первый квартиль)')
    plt.axhline(Q3, color='green', linestyle='dashed', linewidth=2, label='Q3 (Третий квартиль)')
    plt.axhline(lower_bound, color='red', linestyle='dotted', linewidth=2, label='Нижняя граница')
    plt.axhline(upper_bound, color='red', linestyle='dotted', linewidth=2, label='Верхняя граница')
    
    plt.title('метод квартилей / ' + name)
    plt.xlabel('Индекс')
    plt.ylabel(name)
    plt.legend()
    plt.show()
    
    # Вывод найденных выбросов
    print("Найденные выбросы:", data[outliers])


def ZScoreAnomaly(data, name):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    # Определение порога для выбросов
    threshold = 3
    
    # Поиск выбросов
    outliers = np.where(np.abs(z_scores) > threshold)
    
    # Визуализация данных и выбросов
    plt.figure(figsize=(12, 7))
    plt.plot(data, 'bo', label='Данные')
    plt.plot(outliers[0], data[outliers], 'ro', label='Выбросы')
    plt.axhline(mean, color='g', linestyle='dashed', linewidth=2, label='Среднее значение')
    plt.axhline(mean + threshold * std_dev, color='r', linestyle='dotted', linewidth=2, label='Порог выбросов')
    plt.axhline(mean - threshold * std_dev, color='r', linestyle='dotted', linewidth=2)
    
    plt.title('Z-Score / ' + name)
    plt.xlabel('Индекс')
    plt.ylabel(name)
    plt.legend()
    plt.show()
    
    # Вывод найденных выбросов
    print("Найденные выбросы:", data[outliers])


def ShowMedianMinMaxBarPlot(column, df):
    grouped = df.groupby(column)['CO2 Emissions(g/km)'].agg(['median', 'min', 'max'])
    
    sorted_median = grouped.sort_values(by='median', ascending=False)
    
    sorted_median[column] = sorted_median.index
    
    sns.set_theme(rc={'figure.figsize':(15,8)})
    sns.barplot(x=column, y='max', color='#bcbcbc', data=sorted_median, label = "max")
    ax = sns.barplot(x=column, y='median', color='#5ea9ec', data=sorted_median, label = "median")
    sns.barplot(x=column, y='min', color='#7ae990',  data=sorted_median, label = "min")
    ax.bar_label(ax.containers[0],rotation=90)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title('CO2 emission in g/km  vs  ' + column)
    plt.show()