from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz


def fuzzy_matching(mapper, fav_app, verbose=True):
    """
    return the closest match via fuzzy ratio. If no match found, return None
    
    Parameters
    ----------    
    mapper: dict (animes_to_app), map app title name to index of the app in data

    fav_app: str, name of user input app
    
    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_app.lower())
        if ratio > 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('App yang anda cari tidak ada')
        return
    if verbose:
        print('App yang anda cari ditemukan di database kami: {}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_recommendation(model_knn, data, mapper, fav_app, n_recommendation):
    list_rekom=[]
    """
    return top n similar movie recommendations based on user's input movie


    Parameters
    ----------
    model_knn: sklearn model, knn model

    data: df_use for training

    mapper: dict (app_to_idx), map app title name to index of the movie in data

    fav_app: str, name of user input app

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar app recommendations
    """
    model_knn.fit(data)
    # get input movie index
    print('Anda telah memilih App: {}'.format(fav_app))
    idx = fuzzy_matching(mapper, fav_app, verbose=True)
    print('Sedang memuat App')
    print('.......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendation+1)
    # get list of raw index
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x:x[1])[:0:-1]
    reverse_mapper = {v: k for k, v in mapper.items()}
    
#     print('Rekomendasi untuk: {}'.format(fav_app))
    for i, (idx, dist) in enumerate(raw_recommends):
        appitem='{}: {}, dengan distance of {}'.format(i+1, reverse_mapper[idx], dist)
        list_rekom.append(appitem)
    
    return list_rekom
        