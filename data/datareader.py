from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

def read_ml1m(datasets_dir=None):
    GENRE_ID = {
        "Action" : 0,
        "Adventure" : 1,
        "Animation" : 2,
        "Children's" : 3,
        "Comedy" : 4,
        "Crime" : 5, 
        "Documentary" : 6,
        "Drama" : 7,
        "Fantasy" : 8,
        "Film-Noir" : 9,
        "Horror" : 10, 
        "Musical" : 11,
        "Mystery" : 12,
        "Romance" : 13,
        "Sci-Fi" : 14,
        "Thriller" : 15,
        "War" : 16,
        "Western" : 17
    }

    AGE_ID = {
        1 : 0,
        18 : 1, 
        25 : 2,
        35 : 3,
        45 : 4,
        50 : 5,
        56 : 6
    }

    data_dir = datasets_dir + '/ml-1m'
    movie_data_file = data_dir + '/movies.dat'
    user_data_file = data_dir + '/users.dat'
    ratings_data_file = data_dir + '/ratings.dat'

    # Read User Data
    user_data = {}
    with open(user_data_file, 'r', encoding='latin-1') as f:
        for line in f:
            user_info = line.split("::")
            user_id = int(user_info[0])
            gender_feat = 0 if user_info[1] == 'M' else 1
            age_feat = AGE_ID[int(user_info[2])]
            occupation_feat = int(user_info[3])

            user_data[user_id] = {}
            user_data[user_id]['gender'] = gender_feat
            user_data[user_id]['age'] = age_feat
            user_data[user_id]['occupation'] = occupation_feat
    
    user_id_reindexer = {old_idx : new_idx for new_idx, old_idx in enumerate(user_data)}
    reindexed_user_data = {user_id_reindexer[old_idx] : user_data[old_idx] for old_idx in user_data}
    num_users = len(reindexed_user_data)

    # Read Movie Data
    movie_data = {}
    movie_titles = []
    dates = []  
    with open(movie_data_file, 'r', encoding='latin-1') as f:
        for line in f:
            movie_info = line.split("::")

            movie_id = int(movie_info[0])
            title = movie_info[1][:-6].strip()
            date = int(movie_info[1][-6:][1:-1])
            genres = list(map(lambda genre : GENRE_ID[genre], movie_info[-1].strip().split('|')))
            one_hot = np.zeros((len(GENRE_ID),), dtype=float)
            one_hot[genres] = 1

            movie_data[movie_id] = {}
            movie_data[movie_id]['genres'] = one_hot
            movie_titles.append(title)
            dates.append(date)


    movie_id_reindexer = {old_idx : new_idx for new_idx, old_idx in enumerate(movie_data)}
    reindexed_movie_data = {movie_id_reindexer[old_idx] : movie_data[old_idx] for old_idx in movie_data}

    unique_dates = np.unique(dates).tolist()
    date_featurizer = {date : i for i, date in enumerate(unique_dates)}
    date_feats = list(map(lambda date : date_featurizer[date], dates))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    title_embeddings = model.encode(movie_titles)

    for idx in reindexed_movie_data:
        reindexed_movie_data[idx]['date'] = date_feats[idx]
        reindexed_movie_data[idx]['title_embedding'] = title_embeddings[idx]

    movie_data_with_scaled_idx = {idx + num_users : reindexed_movie_data[idx] for idx in reindexed_movie_data} # Item IDs start at num_users and go to (num_users + num_items - 1)
    del reindexed_movie_data, movie_data

    # Read Interactions
    ratings = defaultdict(list)
    with open(ratings_data_file, 'r', encoding='latin-1') as f:
        for line in f:
            interaction_data = line.split("::")

            user_id = user_id_reindexer[int(interaction_data[0])]
            item_id = movie_id_reindexer[int(interaction_data[1])] + num_users
            rating = int(interaction_data[2])
            time_stamp = int(interaction_data[-1].strip())

            ratings[user_id].append((item_id, rating, time_stamp, int(interaction_data[0]), int(interaction_data[1])))
    return reindexed_user_data, movie_data_with_scaled_idx, ratings, len(unique_dates)