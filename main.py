import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import subprocess
from levenshtein import levenshtein
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# 0. ФУНКЦИИ ДЛЯ ФИЛЬТРАЦИИ (МЕТРИКИ СХОДСТВА)
# ==========================

def jaccard_similarity(list1, list2):
    """Считает коэффициент Жаккара: отношение пересечения множеств к их объединению."""
    # Превращаем списки в множества для удаления дубликатов и быстрой работы
    set1 = set(list1) if isinstance(list1, list) else set()
    set2 = set(list2) if isinstance(list2, list) else set()
    
    # Если оба списка пустые, считаем сходство нулевым
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
        
    intersection = len(set1.intersection(set2)) # Кол-во общих элементов
    union = len(set1.union(set2))               # Кол-во всех уникальных элементов
    
    # Защита от деления на ноль
    return intersection / union if union != 0 else 0.0

def euclidean_distance(row1, row2):
    """Считает евклидово расстояние между двумя фильмами по их нормализованным признакам."""
    return math.sqrt(
        (row1['rating_norm'] - row2['rating_norm'])**2 +
        (row1['runtime_norm'] - row2['runtime_norm'])**2
    )

def numeric_similarity(row1, row2):
    """Преобразует расстояние в сходство (от 0 до 1). 1 - полное совпадение, 0 - максимальное различие."""
    MAX_DIST = math.sqrt(2) # Максимально возможное расстояние в 2D (если разница по обеим осям = 1)
    dist = euclidean_distance(row1, row2)
    return 1 - (dist / MAX_DIST) # Инвертируем шкалу

def combined_similarity(row1, row2, w_numeric=0.4, w_genre=0.3, w_stars=0.3):
    """Итоговая взвешенная оценка похожести двух фильмов (40% цифры, 30% жанры, 30% актеры)."""
    sim_num = numeric_similarity(row1, row2)
    sim_gen = jaccard_similarity(row1['genre_list'], row2['genre_list'])
    sim_strs = jaccard_similarity(row1['stars_list'], row2['stars_list'])
    return w_numeric * sim_num + w_genre * sim_gen + w_stars * sim_strs

# ======================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ДЛЯ КОНТЕНТНОЙ ФИЛЬТРАЦИИ
# ======================================
movies = pd.read_csv('christmas_movies.csv')
print(f"Загружено {len(movies)} фильмов")

df = movies.copy()

def parse_items(cell):
    """Превращает строку с перечислением (например, "Comedy, Drama") в список элементов."""
    if pd.isna(cell):
        return [] # Защита от пустых ячеек (NaN)

    items = [item.strip() for item in cell.split(',')] # Разделяем по запятой и убираем пробелы
    items = [item for item in items if item]            # Убираем пустые строки, если они есть

    return items

# Применяем парсинг к столбцам с жанрами и актерами
df['stars_list'] = df['stars'].apply(parse_items)
df['genre_list'] = df['genre'].apply(parse_items)

# Удаляем фильмы, у которых нет рейтинга или длительности (без них нельзя считать сходство)
df_clean = df.dropna(subset=['imdb_rating', 'runtime']).copy()
df_clean.reset_index(drop=True, inplace=True) # Сбрасываем индексы после удаления строк

# Нормализация рейтинга (приводим к шкале от 0 до 1)
min_rtg = df_clean['imdb_rating'].min()
max_rtg = df_clean['imdb_rating'].max()
df_clean['rating_norm'] = (df_clean['imdb_rating'] - min_rtg) / (max_rtg - min_rtg)

# Нормализация длительности (приводим к шкале от 0 до 1)
min_runtime = df_clean['runtime'].min()
max_runtime = df_clean['runtime'].max()
df_clean['runtime_norm'] = (df_clean['runtime'] - min_runtime) / (max_runtime - min_runtime)

# Сохраняем очищенный датасет
df_clean.to_csv('christmas_movies_clean.csv', index=False)

# Запускаем скрипт для генерации синтетических оценок пользователей
subprocess.run(['python', 'synthetic-creation.py'])

# ======================================
# 2. ТЕПЛОВАЯ КАРТА ФИЛЬМОВ (первые 50)
# ======================================
# Проверяем, есть ли уже готовый файл. Если нет — строим (это долго, поэтому кэшируем)
if not os.path.exists('film_heatmap.png'):
    print("Строим тепловую карту для первых 50 фильмов...")
    sample_df = df_clean.head(50).copy() # Берем только 50 фильмов, чтобы не считать 1000х1000 матрицу
    n = len(sample_df)
    sim_matrix = np.zeros((n, n))
    
    # Заполняем матрицу сходства (только верхний треугольник для оптимизации)
    for i in range(n):
        sim_matrix[i][i] = 1 # Фильм похож сам на себя на 100%
        for j in range(i + 1, n):
            sim_matrix[i][j] = sim_matrix[j][i] = combined_similarity(sample_df.iloc[i], sample_df.iloc[j])
            
    plt.figure(figsize=(12, 10))
    # Рисуем тепловую карту (cmap='coolwarm' — синий для низкого сходства, красный для высокого)
    sns.heatmap(sim_matrix, annot=False, xticklabels=sample_df['title'], yticklabels=sample_df['title'], cmap='coolwarm')
    plt.title('Матрица сходства первых 50 фильмов (Жаккард + Евклид)')
    plt.tight_layout()
    plt.savefig('film_heatmap.png', dpi=150)
    plt.close()
    print("Тепловая карта сохранена как film_heatmap.png\n")
else:
    print("Тепловая карта фильмов уже есть.\n")

# ======================================
# 3. ЗАГРУЗКА ДАННЫХ ДЛЯ КОЛЛАБОРАТИВНОЙ ФИЛЬТРАЦИИ
# ======================================
ratings = pd.read_csv('synthetic_ratings.csv')
print(f"Загружено {len(ratings)} оценок от {ratings['user_id'].nunique()} пользователей")

# Создаем матрицу "Пользователь х Фильм" (строки — юзеры, столбцы — фильмы, значения — оценки)
# fillna(0) означает, что если юзер не смотрел фильм, ставим 0
user_item_matrix = ratings.pivot_table(
    index='user_id',
    columns='movie_name',
    values='rating'
).fillna(0)
all_movies = user_item_matrix.columns.tolist()

# Центрирование данных: вычитаем среднюю оценку пользователя из каждой его оценки.
# Это нужно, чтобы люди с разным "порогом оценки" (кто-то ставит только 5, а кто-то 3) корректно сравнивались.
user_means = user_item_matrix.mean(axis=1)
centered_matrix = user_item_matrix.sub(user_means, axis=0)

# Получаем тип для каждого синтетического пользователя (для графика)
user_type_map = ratings.groupby('user_id')['user_type'].first().to_dict()
user_types_list = [user_type_map.get(uid, 'unknown') for uid in user_item_matrix.index]

# ========================================
# 4. ФУНКЦИЯ КОЛЛАБОРАТИВНЫХ РЕКОМЕНДАЦИЙ
# ========================================
def get_collab_recommendations(user_ratings, top_k=10, top_n=5):
    """Ищет похожих пользователей и рекомендует фильмы, которые они высоко оценили."""
    
    # Создаем вектор оценок для нового пользователя (по умолчанию все 0)
    user_vector = pd.Series(index=all_movies, dtype=float).fillna(0)
    for title, rating in user_ratings.items():
        if title in user_vector.index:
            user_vector[title] = rating
        else:
            print(f"Предупреждение: фильм '{title}' не найден в датасете")

    # Центрируем оценки нового пользователя так же, как и синтетиков
    user_mean = user_vector[user_vector > 0].mean()
    if np.isnan(user_mean): # Если пользователь ничего не оценил
        user_mean = 0
    user_centered = user_vector - user_mean

    # Считаем косинусное сходство между нашим юзером и всеми синтетиками
    sim_scores = cosine_similarity([user_centered], centered_matrix)[0]

    # Берем топ-K самых похожих пользователей (сортируем по убыванию сходства)
    similar_users_idx = np.argsort(sim_scores)[::-1][:top_k]
    sim_values = sim_scores[similar_users_idx]

    # Собираем рекомендации: смотрим, что высоко оценили (>=4) похожие юзеры
    candidates = {}
    for idx, sim in zip(similar_users_idx, sim_values):
        user = user_item_matrix.index[idx]
        user_ratings_raw = user_item_matrix.loc[user]
        
        for movie, rating in user_ratings_raw[user_ratings_raw >= 4].items():
            # Не рекомендуем фильмы, которые наш пользователь уже оценил
            if movie not in user_ratings:
                if movie not in candidates:
                    candidates[movie] = 0
                # Взвешиваем оценку на схожесть пользователей (чем похожее юзер, тем важнее его оценка)
                candidates[movie] += rating * sim

    if not candidates:
        return [], sim_scores # Если ничего не нашли
    
    # Сортируем кандидатов по накопленному весу и берем top_n
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:top_n], sim_scores

# ======================================
# 5. ФУНКЦИЯ КОНТЕНТНЫХ РЕКОМЕНДАЦИЙ
# ======================================
def get_content_recommendations(movie_title, top_n=5):
    """Ищет фильмы, похожие на заданный по жанрам, актерам и параметрам."""
    
    # Ищем фильм по подстроке (чтобы можно было ввести часть названия)
    matches = df_clean[df_clean['title'].str.lower().str.contains(movie_title.lower(), na=False)]
    
    if len(matches) == 0:
        return None, f"Фильм '{movie_title}' не найден."
    
    # Если нашлось несколько вариантов, даем пользователю выбрать
    if len(matches) > 1:
        print("\nНайдено несколько вариантов:")
        for i, row in enumerate(matches.itertuples(), 1):
            print(f"  {i}. {row.title}")
        print("0 — ввести название заново")
        
        # Цикл, пока пользователь не сделает корректный выбор
        while True:
            choice = input("Введите номер или точное название: ").strip()
            if choice == '0':
                return None, "Попробуйте ввести название заново."
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(matches):
                    target = matches.iloc[idx]
                    break
                else:
                    print("Неверный номер. Попробуйте ещё раз.")
                    continue
            else:
                exact = matches[matches['title'].str.lower() == choice.lower()]
                if len(exact) == 1:
                    target = exact.iloc[0]
                    break
                else:
                    print("Название не найдено среди вариантов. Попробуйте ещё раз или введите 0.")
                    continue
    else:
        target = matches.iloc[0] # Если найден ровно один фильм

    # Считаем сходство с каждым другим фильмом в базе
    similarities = []
    for _, row in df_clean.iterrows():
        if row['title'] == target['title']:
            continue # Пропускаем сам фильм
        sim = combined_similarity(target, row, w_numeric=0.4, w_genre=0.3, w_stars=0.3)
        similarities.append((row['title'], sim))

    # Сортируем по убыванию сходства и возвращаем топ-N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return target['title'], similarities[:top_n]

# ======================================
# 6. ФУНКЦИЯ ДЛЯ ПОСТРОЕНИЯ ГРАФИКА СХОДСТВА С ТИПАМИ
# ======================================
def plot_user_type_similarity(sim_scores, user_types):
    """Строит столбчатую диаграмму, показывая, с каким типом пользователей вы наиболее схожи."""
    
    # Группируем сходства по типу пользователя и считаем среднее
    df_sim = pd.DataFrame({'user_type': user_types, 'similarity': sim_scores})
    grouped = df_sim.groupby('user_type', as_index=False)['similarity'].mean()
    grouped = grouped.sort_values('similarity', ascending=False) # Сортируем для красоты
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=grouped, x='user_type', y='similarity', hue='user_type', legend=False)
    plt.title('Сходство ваших предпочтений с профилями синтетических пользователей')
    plt.xlabel('Тип пользователя')
    plt.ylabel('Среднее косинусное сходство')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('user_similarity_to_types.png', dpi=150)
    plt.show()
    print("График сходства сохранён как user_similarity_to_types.png")

# ======================================
# 7. ГЛАВНОЕ МЕНЮ
# ======================================
while True:
    print("\n" + "="*50)
    print("Выберите действие:")
    print("1 — Коллаборативная фильтрация (оцените несколько фильмов)")
    print("2 — Контентная фильтрация (похожие фильмы по свойствам)")
    print("0 — Выход")
    print("="*50)
    mode = input("Ваш выбор: ").strip()

    if mode == '1':
        # ---- Коллаборативный режим ----
        print("\nОцените несколько фильмов от 1 до 5 (или введите 'стоп' для завершения).")
        user_ratings = {}
        
        # Цикл сбора оценок от пользователя
        while True:
            title = input("\nНазвание фильма: ").strip()
            if title.lower() == 'стоп':
                break
                
            # Ищем совпадения по подстроке
            matching = [m for m in all_movies if title.lower() in m.lower()]
            if not matching:
                print("Фильм не найден. Попробуйте ещё раз.")
                continue
                
            # Если несколько совпадений, даем выбрать
            if len(matching) > 1:
                print("Найдено несколько вариантов:")
                for i, m in enumerate(matching[:5]): # Показываем только первые 5
                    print(f"  {i+1}. {m}")
                choice = input("Введите номер (или 0, чтобы ввести название заново): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(matching):
                        title = matching[idx]
                    else:
                        continue
                else:
                    continue
            else:
                title = matching[0]
                
            # Запрашиваем оценку
            try:
                rating = float(input("Оценка (1-5): "))
                if rating < 1 or rating > 5:
                    print("Оценка должна быть от 1 до 5.")
                    continue
                user_ratings[title] = rating
                print(f"Добавлено: {title} — {rating}")
            except ValueError:
                print("Введите число.")

        if len(user_ratings) == 0:
            print("Не введено ни одной оценки. Завершаем.")
            exit()

        print("\nВаши оценки:")
        for title, rating in user_ratings.items():
            print(f"  {title}: {rating}")

        print("\nИщем похожих пользователей...")
        recommendations, sim_scores = get_collab_recommendations(user_ratings, top_k=10, top_n=5)

        # Строим график сходства с типами
        plot_user_type_similarity(sim_scores, user_types_list)

        if not recommendations:
            print("Не удалось найти рекомендации. Попробуйте оценить другие фильмы.")
        else:
            print("\nРекомендуемые фильмы (коллаборативная фильтрация):")
            for i, (title, score) in enumerate(recommendations, 1):
                print(f"{i}. {title} — прогноз: {score:.2f}")

    elif mode == '2':
        # ---- Контентный режим ----
        title_input = input("\nВведите название фильма (для поиска похожих): ").strip()
        target, recs = get_content_recommendations(title_input, top_n=5)
        
        if target is None:
            print(recs) # Выводим сообщение об ошибке, если фильм не найден
        else:
            print(f"\nФильмы, похожие на '{target}' (по свойствам):")
            for i, (title, sim) in enumerate(recs, 1):
                print(f"{i}. {title} (сходство: {sim:.3f})")

    else:
        print("Выход.")
        sys.exit()
