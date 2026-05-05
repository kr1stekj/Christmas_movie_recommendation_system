import pandas as pd
import numpy as np
import random
import datetime
from tqdm import tqdm

# ======================================
# 1. Загружаем наш датасет
# ======================================
df = pd.read_csv("christmas_movies_clean.csv")
print(f"Загружено {len(df)} фильмов")

# Получаем все уникальные жанры
all_genres = []
for genres in df['genre_list']:
    for genre in genres:
        all_genres.append(genre)

all_genres = set(all_genres)
all_genres = sorted(list(all_genres))
print(f"Найдено жанров: {len(all_genres)}")
print(f"Жанры: {all_genres}")

# ======================================
# 2. Создаём пользовательские профили
# ======================================

user_types = [
    {
        'name': 'action_fan',
        'like_genres': ['Action', 'Adventure', 'Sci-Fi', 'Crime'],
        'dislike_genres': ['Drama', 'Romance', 'Musical'],
        'bias': 0.5,
        'noise': 0.5,
        'count': 150
    },
    {
        'name': 'drama_fan',
        'like_genres': ['Drama', 'Romance', 'Musical'],
        'dislike_genres': ['Horror', 'Thriller'],
        'bias': 0.3,
        'noise': 0.4,
        'count': 150
    },
    {
        'name': 'horror_fan',
        'like_genres': ['Horror', 'Thriller', 'Mystery'],
        'dislike_genres': ['Comedy', 'Family', 'Musical'],
        'bias': 0.2,
        'noise': 0.3,
        'count': 120
    },
    {
        'name': 'animation_fan',
        'like_genres': ['Animation', 'Family', 'Comedy'],
        'dislike_genres': ['Documentary', 'History'],
        'bias': 0.6,
        'noise': 0.4,
        'count': 100
    },
    {
        'name': 'documentary_fan',
        'like_genres': ['Documentary', 'Biography', 'History'],
        'dislike_genres': ['Horror', 'Action'],
        'bias': 0.1,
        'noise': 0.3,
        'count': 80
    },
    {
        'name': 'casual_viewer',
        'like_genres': [],
        'dislike_genres': [],
        'bias': 0.0,
        'noise': 0.8,
        'count': 200
    },
    {
        'name': 'critic',
        'like_genres': [],
        'dislike_genres': [],
        'bias': -0.5,
        'noise': 0.3,
        'count': 100
    },
    {
        'name': 'enthusiast',
        'like_genres': [],
        'dislike_genres': [],
        'bias': 0.8,
        'noise': 0.2,
        'count': 100
    }
]

# ======================================
# 3. Функция для расчёта оценки
# ======================================

def calculate_rating(movie_genres, user_profile):
    # Если у фильма нет жанров, ставим случайную оценку
    if not movie_genres:
        return round(random.uniform(2.5, 4.5), 1)
    
    # Считаем совпадения с любимыми жанрами
    like_overlap = 0
    if user_profile['like_genres']:
        like_overlap = len(set(movie_genres).intersection(set(user_profile['like_genres'])))
    
    # Считаем совпадения с нелюбимыми жанрами
    dislike_overlap = 0
    if user_profile['dislike_genres']:
        dislike_overlap = len(set(movie_genres).intersection(set(user_profile['dislike_genres'])))
    
    # Базовая оценка (3.0) + влияние предпочтений
    base_rating = 3.0
    like_boost = like_overlap * 0.8
    dislike_penalty = dislike_overlap * 0.7
    
    # Применяем bias пользователя
    rating = base_rating + like_boost - dislike_penalty + user_profile['bias']
    
    # Добавляем случайный шум
    noise = random.uniform(-user_profile['noise'], user_profile['noise'])
    rating += noise
    
    # Ограничиваем от 1 до 5
    rating = min(5.0, rating)
    rating = max(1.0, rating)
    
    return round(rating, 1)

# ======================================
# 4. Генерируем пользователей и оценки
# ======================================

ratings_data = []
user_id_counter = 1

print("Генерация пользователей и оценок...")

for user_type in user_types:
    print(f"  Создаём {user_type['count']} пользователей типа '{user_type['name']}'...")
    
    for _ in tqdm(range(user_type['count'])):
        num_ratings = random.randint(30, 150)
        sampled_movies = df.sample(n=min(num_ratings, len(df)))
        
        for idx, movie in sampled_movies.iterrows():
            # Берём уже готовый список жанров
            movie_genres = movie['genre_list']
            
            rating = calculate_rating(movie_genres, user_type)
            
            # Создаём случайное время отзыва
            td = random.random() * datetime.timedelta(days=1)
            td = str(td)
            #timestamp = td[:td.index(".")]
            ratings_data.append({
                'user_id': user_id_counter,
                'movie_name': movie['title'],
                'rating': rating,
                'timestamp': td,
                'user_type': user_type['name']
            })
        
        user_id_counter += 1

print(f"\nСгенерировано {user_id_counter - 1} пользователей")
print(f"Всего оценок: {len(ratings_data)}")

# ======================================
# 5. Сохраняем в CSV
# ======================================

ratings_df = pd.DataFrame(ratings_data)
ratings_df.to_csv('synthetic_ratings.csv', index=False)


