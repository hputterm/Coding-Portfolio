import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), the two biases Ai and Bj, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    gradient = reg * Ui
    gradient -= Vj*(Yij - (np.dot(Ui, Vj)))
    return eta * gradient

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), the two biases Ai and Bj, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    gradient = reg * Vj
    gradient -= Ui * (Yij - (np.dot(Ui, Vj)))
    return eta * gradient

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T???.
    """
    err = 0
    for training_point in Y:
        i, j, Y_ij = training_point
        # correct for the 1 indexing
        i -= 1
        j -= 1
        Ui = U[i]
        Vj = V[:,j]
        err += .5*(np.dot(Ui, Vj)-Y_ij)**2
    if reg != 0:
        U_frobenius_norm = np.linalg.norm(U, ord='fro')
        V_frobenius_norm = np.linalg.norm(V, ord='fro')
        err += 0.5 * reg * (U_frobenius_norm ** 2)
        err += 0.5 * reg * (V_frobenius_norm ** 2)
    return err/len(Y)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.uniform(low = -.5, high = .5, size = (M, K))
    V = np.random.uniform(low = -.5, high = .5, size = (K, N))
    np.random.shuffle(Y)
    current_error = get_err(U,V,Y, reg = reg)
    error_initial = 0
    # Loop through all of the epochs
    for epoch in range(max_epochs):
        print("Epoch: " + str(epoch))
        # Loop through all of the training points
        for training_point in Y:
            i, j, Y_ij = training_point
            # Correct for the one indexing
            i-=1
            j-=1
            Ui = U[i]
            Vj = V[:,j]
            # Compute the gradients
            gradient_U = grad_U(Ui, Y_ij, Vj, reg, eta)
            gradient_V = grad_V(Vj, Y_ij, Ui, reg, eta)
            #Update using the gradients
            U[i] -= gradient_U
            V[:,j] -= gradient_V
        # Compute comparison loss for the first epoch
        if(epoch == 0):
            error_initial = current_error - get_err(U,V,Y, reg = reg)
        # Determine whether stopping condition is satisfied
        previous_error = current_error
        current_error = get_err(U,V,Y, reg = reg)
        print("Current Regularized Mean Squared Error: " + str(current_error))
        if((previous_error-current_error)/error_initial < eps):
            break
        # Shuffle the training points
        np.random.shuffle(Y)
    return (U,V,get_err(U,V,Y))
