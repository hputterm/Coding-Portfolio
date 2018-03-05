import numpy as np
import matplotlib.pyplot as plt
from prob2utils_skeleton import train_model, get_err
from sklearn import preprocessing
import pandas as pd

def main():
    Y_train = np.loadtxt('data/data/train.txt').astype(int)
    Y_test = np.loadtxt('data/data/test.txt').astype(int)
    movies_total = np.loadtxt('data/data/data.txt').astype(int)
    movie_path = 'data/data/movies_UTF8.txt'
    i_cols = ['Movie ID', 'Movie Title', 'Unknown', 'Action', 'Adventure', \
    'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', \
    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',\
    'Thriller', 'War', 'Western'
    ]
    items = pd.read_csv(movie_path, sep='\t', header = None, names = i_cols, )

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies

    movies_frequency_rating = np.zeros(N+1)
    movies_total_rating = np.zeros(N+1)

    for (i, j, Yij) in movies_total:
        if(items['Documentary'][j]==1):
            print(items['Movie Title'][j])


if __name__ == "__main__":
    main()
