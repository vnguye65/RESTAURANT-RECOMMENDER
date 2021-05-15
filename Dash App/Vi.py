#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:28:36 2021

@author: ving2000
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import linalg
from numpy import dot
from paretoset import paretorank


#path = '/Users/ving2000/Downloads/CDS 403/Final_project/uci-restaurant-consumer-data/'

users = pd.read_csv('cleaned_userprofile.csv').set_index('userID')
usercuisine = pd.read_csv('user_cuisine.csv')

restaurants = pd.read_csv('cleaned_restaurants.csv')
rescuisine = pd.read_csv('restaurant_cuisine.csv')

#ratings = pd.read_csv(os.path.join(path, 'rating_final.csv'))
ratings = pd.read_csv('rating_final.csv')
rclusters = pd.read_csv('Restaurant_clusters.csv')


def GetRMatrix (actual_df, ids):
    """
    Transforms actual_df to a userID x placeID matrix 
    actual_df : ratings dataframe
    """
   
    ratings_df = actual_df.copy()
    ratings_df['rating'] = ratings_df['rating'] + 1
    
    df = pd.DataFrame(columns = restaurants.placeID, index = ids)
    for i, row in df.iterrows():
        fdf = ratings_df[ratings_df['userID'] == i]
        dct = dict(zip(fdf.placeID, fdf.rating))
        df.loc[i] = dct
    return df.fillna(0).values

#----------------------------------------------------

def GetRMatrix2 (actual_df, restaurant_clusters):
    """
    Transforms actual_df to a userID x placeID matrix 
    actual_df : ratings dataframe
    """
   
    ratings_df = actual_df.copy()
    ## Add 1 to ratings
    ratings_df['rating'] = ratings_df['rating'] + 1
    df = pd.DataFrame(columns = restaurant_clusters.placeID, index = users.index)
    ## i = userID
    for i, row in df.iterrows():
        ## Get all actual ratings from user i
        fdf = ratings_df[ratings_df['userID'] == i]
        ## Store in a dictionary (placeID-column : rating)
        dct = dict(zip(fdf.placeID, fdf.rating))
        ## Get the cluster of each restaurant
        clusters = fdf.merge(restaurant_clusters, on = 'placeID', how = 'left')
        ## Calculate average rating of each restaurant group
        ## index  rating
        ## 0      3
        ## 5      6
        avg_clusters = clusters[['rating', 'cluster']].groupby('cluster').mean().to_dict()['rating']
        
        ## Loop thru every restaurant group user i has visited
        for n in avg_clusters.keys():
            ## Get all restaurants in group n
            ids = restaurant_clusters[restaurant_clusters['cluster'] == n].placeID
            ## Remove the ones that we already put in dct
            ids = ids[~ids.isin(fdf.placeID)]
            for res in ids:
                dct[res] = avg_clusters[n]
            
        df.loc[i] = dct
    return df.fillna(0).values

#----------------------------------------------------
def GroupRestaurants (restaurants, k = 12):
    res = restaurants.copy().drop(['latitude', 'longitude', 'name'], 1).set_index('placeID')
    
    #SSE = []
    #for cluster in tqdm(range(6,20)):
    #    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    #    kmeans.fit(res)
    #    SSE.append(kmeans.inertia_)
        
    #kn = KneeLocator(np.arange(6, 20), SSE, curve='convex', direction='decreasing').knee
    kmeans = KMeans(n_jobs = -1, n_clusters = k, init='k-means++')
    kmeans.fit(res)
    preds = kmeans.predict(res)
    res['cluster'] = preds
    
    return res.reset_index()[['placeID', 'cluster']]
#----------------------------------------------------
#https://stackoverflow.com/questions/22767695/python-non-negative-matrix-factorization-that-handles-both-zeros-and-missing-dat
def NMFModel (X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    #print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features,
    #                                                                                     max_iter))
    #X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)

        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            #print('Iteration {}:'.format(i))
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            #print('fit residual', np.round(fit_residual, 4))
            #print('total residual', np.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y

#----------------------------------------------------
def Recommend (restaurants, R_preds, userID, ids, num_recom = 10):
    
    """
    Sorts ratings and matches placeIDs with names
    """
    ## Clipping:
    R_preds[R_preds > 2.] = 2.
    R_preds[R_preds < 0.] = 0.

    
    df = pd.DataFrame(R_preds, index = ids, columns = restaurants.placeID)
    predictions = df.loc[userID].to_dict()
    spredictions = sorted(predictions.items(), key = lambda x: (-x[1], x[0]))[:num_recom]
    
    recoms = pd.DataFrame()
    for i in spredictions:
        dff = restaurants[restaurants['placeID'] == i[0]][['placeID', 'name']]
        dff['predicted'] = i[1]
        recoms = pd.concat((recoms, dff), axis = 0)
        
    return recoms, df



def Standardize (df, new_user_arr):
    
    scaler = StandardScaler()
    scaled_ = scaler.fit_transform(df)
    users_ = pd.DataFrame(scaled_, columns = df.columns, index = df.index)
    new_user_arr_ = scaler.transform(new_user_arr)
    return users_, new_user_arr_



def GetNeighbors (actual, users, usercuisine, new_userID, new_user_arr, new_cuisine):
    """
    users: users dataframe
    usercuisine
    new_cuisine: list
    """
    
    ## Users with similar tastes
    sim_taste = usercuisine[usercuisine['category'].isin(new_cuisine)]['userID'].unique()
    sim_users = users.loc[sim_taste, :].drop(['latitude', 'longitude'], axis = 1)
    if 'No preference' in new_cuisine:
        sim_users = users.drop(['latitude', 'longitude'], axis = 1)
        
    new_user_arr = np.delete(new_user_arr, [0, 1], axis=1)
    ## Scale 
    users_, new_user_arr_ = Standardize(sim_users, new_user_arr)
    
    ## KNN
    ## use the standard recommended n_neighbors
    n_neighbors = round(np.sqrt(sim_users.shape[0]))
    knn = NearestNeighbors(n_neighbors = n_neighbors)
    knn.fit(users_)
 
    ## Return at least 5 neighbors' indices
    ind = []
    rad = 0.5
    while len(ind) < 5:
        ind = knn.radius_neighbors(new_user_arr_, radius = rad, return_distance = False)[0]
        rad += 0.25
        if rad >= 5:
            break
    userids = sim_users.iloc[ind].index

    ## Calculate average ratings and append to actual ratings dataframe
    knn_df = actual[actual['userID'].isin(userids)].drop('userID', 1)
    gb = knn_df.groupby('placeID', as_index = True).mean().reset_index()
    gb.insert(0, 'userID', new_userID)
    final = pd.concat((actual, gb), axis = 0)
    
    return final


def Update (new_user_arr, userID, users):
    new = pd.DataFrame(new_user_arr, columns = users.columns)
    new.index = [userID]
    new_users = pd.concat((users, new), axis = 0)
    ## Update usercuisine
    return new_users



def harvesine_dist(lat1, lon1, lat2, lon2):
    """ Calculates distance between 2 coordinates"""
    R = 6373.0 # approximate radius of earth in km
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def Skyline (R_preds, ids, ulat, ulong, uid):
    
    R_preds[R_preds > 2.] = 2.
    R_preds[R_preds < 0.] = 0.

    
    df = pd.DataFrame(R_preds, index = ids, columns = restaurants.placeID)
    preds = df.loc[uid].to_frame()

    distances = []
    for res in preds.index:
        rlat, rlong = restaurants[restaurants['placeID'] == res][['latitude', 'longitude']].values[0]
        dist = harvesine_dist(ulat, ulong, rlat, rlong)
        distances.append(dist)
    
    preds['distance'] = distances
    preds.rename(columns = {uid: 'rating'}, inplace = True)
    preds = preds[preds['rating'] > 1]
    
    ind = paretorank(preds, sense = ['max', 'min'])
    
    pareto = pd.DataFrame()
    for i in range(1, max(ind) + 1):
        indx = (ind == i).nonzero()[0]
        pareto = pd.concat((pareto, preds.iloc[indx,:]), axis = 0)

    recoms = pareto.merge(restaurants, left_on = pareto.index, right_on = 'placeID', how = 'left').reset_index()
    recoms['rank'] = recoms.index + 1
    
    return recoms[['rank', 'name', 'rating', 'distance']]




