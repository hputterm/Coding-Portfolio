import numpy as np
import matplotlib.pyplot as plt
from prob2utils_nobias import train_model, get_err
from sklearn import preprocessing
import pandas as pd

def main():
    Y_train = np.loadtxt('data/data/train.txt').astype(int)
    Y_test = np.loadtxt('data/data/test.txt').astype(int)
    movies_total = np.loadtxt('data/data/data.txt').astype(int)
    movie_path = 'data/data/movies_UTF8.txt'
    i_cols = ['Movie ID', 'Movie Title', 'Unknown', 'Action', 'Adventure', \
    'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', \
    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', \
    'Thriller', 'War', 'Western'
    ]
    items = pd.read_csv(movie_path, sep='\t', header = None, names = i_cols, )

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    eta = 0.03 # learning rate

    # Use to compute Ein and Eout
    print("Training model with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, 20, eta, .1))
    U, V, e_in = train_model(M, N, 20, eta, .1, movies_total)

    print(U.shape)
    # For me the shape of U is (943, 20)
    print(V.shape)
    # For me the shape of V is (20, 1682)

    A, sigma, B = np.linalg.svd(V, full_matrices = True)
    AT = A[:,:2].T

    # Compute and normalize ATV
    ATV = np.dot(AT, V)
    for row in ATV:
        row -= row.mean()
        row /= row.std()


    groups_to_analyze = [
    [121, 300,1,100,258,50,288,294,286,181],
    [1599, 1201, 1189, 1122, 1653,  814, 1467, 1500, 1536, 1293],
    [50,181,222,227,210,168,172,395,627,135],
    [395,168,154,249,294,652,94,69,411,90],
    [811,48,814,757,766,813,677,850,645,1022],
    [135,28,82,577,176,195,89,204,121,228],
    [50, 181, 172, 222, 227, 228, 229, 230, 380, 449, 450]
    ]
    titles = ["Most Popular Movies No Bias","Highest Rated Movies No Bias",
    "Movies we Chose No Bias", "Comedies No Bias", "Documentary No Bias", "Sci-Fi No Bias", "Star Wars and Star Trek"]
    plt.figure(figsize = (12,12))
    for current_indices, title in zip(groups_to_analyze, titles):
        current_indices_titles = [items['Movie Title'][i-1] for i in current_indices]
        current_points = np.array([ATV[:, i-1] for i in current_indices])

        plt.scatter(current_points[:, 0], current_points[:, 1])
        for label, x, y in zip(current_indices_titles, current_points[:, 0], current_points[:, 1]):
            plt.annotate(
            label,
            xy=(x, y), xytext=(-10, 10),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.25),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

        plt.margins(x = .2, y = .2)
        plt.title(title)
        plt.rc({'font_size': 20})
        plt.savefig(title + '.png')

        print(current_points)
        print(current_indices_titles)
        plt.clf()

if __name__ == "__main__":
    main()
