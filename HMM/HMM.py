import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.
        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [[0 for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize the start state
        for state in range(0,self.L):
            probs[0][state] = np.log(self.A_start[state])+np.log(self.O[state][x[0]])
            seqs[0][state] = 0
        # Next we iterate for all the remaining possibilities
        # Iterate through every position in the sequence
        for t in range(1, M):
            # Iterate through every state in this position
            for state in range(0,self.L):
                # Compute the log probability
                log_transition_probabilities = [probs[t-1][prev_st]+np.log(self.A[prev_st][state])+np.log(self.O[state][x[t]]) for prev_st in range(0,self.L)]
                # Find which previous state maximized log probability
                maximum_log_probability_coordinate = np.argmax(log_transition_probabilities)
                maximum_log_probability = log_transition_probabilities[maximum_log_probability_coordinate]
                # Update or dynamic programming
                probs[t][state] = maximum_log_probability
                seqs[t][state] = maximum_log_probability_coordinate

        # Update using output emission
        for state in range(0, self.L):
            log_transition_probabilities = [probs[M-1][prev_st]+np.log(self.A[prev_st][state]) for prev_st in range(0,self.L)]
            # Find which previous state maximized log probability
            maximum_log_probability_coordinate = np.argmax(log_transition_probabilities)
            maximum_log_probability = log_transition_probabilities[maximum_log_probability_coordinate]
            # Update or dynamic programming
            probs[M][state] = maximum_log_probability
            seqs[M][state] = maximum_log_probability_coordinate
        # We have finished generating probs and seqs so now we need to extract the max_seq
        max_seq = ''
        starting_coordinate = np.argmax(probs[M-1])
        starting_value = seqs[M][starting_coordinate]
        max_seq += str(starting_value)
        for t in range(M-1, 0, -1):
            starting_value = seqs[t][starting_value]
            max_seq += str(starting_value)
        return max_seq[::-1]


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        for state in range(self.L):
            alphas[0][state] = self.A_start[state]
        # Initialize the alpha vector
        for state in range(self.L):
            alphas[1][state] = self.A_start[state] * self.O[state][x[0]]/sum(self.A_start)
        # # Loop through all of the tokens in the string
        for t in range(2, M+1):
            # Loop through all of the possible states
            current_sum = 0
            for state in range(0, self.L):
                # Update the alphas at this coordinate with the sum of all
                # the transition probabilities.
                temp_var = sum([alphas[t-1][prev_st]*self.A[prev_st][state]*self.O[state][x[t-1]] for prev_st in range(0,self.L)])
                current_sum += temp_var
                alphas[t][state] += temp_var

            if(normalize==True):
                for state in range(self.L):
                    alphas[t][state] /= current_sum

        # return final matrix
        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize the last row as the ending probabilities
        for state in range(0, self.L):
            betas[M][state] = 1
        # Loop through all of the tokens in backwards order
        for t in range(M - 1, 0, -1):
            # Loop through all of the states
            current_sum = 0
            for state in range(0, self.L):
                # Update betas with the probability of the next betas
                temp_var = sum([betas[t+1][next_st]*self.A[state][next_st]*self.O[next_st][x[t]] for next_st in range(0,self.L)])
                betas[t][state] += temp_var
                current_sum += temp_var

            if(normalize==True):
                for state in range(self.L):
                    betas[t][state] /= current_sum

        # transition back using start
        current_sum = 0
        for state in range(0, self.L):
            temp_var = sum([betas[t+1][next_st]*self.A_start[state]*self.O[next_st][x[0]] for next_st in range(0,self.L)])
            betas[0][state] += temp_var
            current_sum += temp_var
        if(normalize==True):
            for state in range(self.L):
                betas[0][state] /= current_sum
        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        # Loop through every element of the A matrix.
        for a in range(self.L):
            for b in range(self.L):
                # Initialize variables for the numerator and denominator of the expression
                numerator_sum = 0
                denominator_sum = 0
                # Loop through every training example
                for training_label_list in Y:
                    # Loop through every index in the training example
                    for training_label_index in range(1, len(training_label_list)):
                        # Update the numerator
                        if(training_label_list[training_label_index]==b and training_label_list[training_label_index-1]==a):
                            numerator_sum += 1
                        # Update the denominator
                        if(training_label_list[training_label_index-1]==a):
                            denominator_sum += 1
                # Update A
                self.A[a][b] = numerator_sum/denominator_sum


        # Calculate each element of O using the M-step formulas.
        # Loop through every elements of the A matrix
        for w in range(self.D):
            for a in range(self.L):
                #Initialize the variables for the numerator and denominator of the expression
                numerator_sum = 0
                denominator_sum = 0
                # Loop through every training example
                for training_points_list, training_label_list in zip(X, Y):
                    # Loop through every index in the training example
                    for training_index in range(0, len(training_label_list)):
                        # Update the numerator
                        if(training_points_list[training_index]==w and training_label_list[training_index]==a):
                            numerator_sum += 1
                        # Update the numerator
                        if(training_label_list[training_index]==a):
                            denominator_sum += 1
                # Update O
                self.O[a][w] = numerator_sum/denominator_sum

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        for i in range(N_iters):
            print(i)
            # Initialize numerator and denominator matrices
            A_numerator = np.zeros((self.L, self.L))
            A_denominator = np.zeros(self.L)
            O_numerator = np.zeros((self.L, self.D))
            O_denominator = np.zeros(self.L)
            # Loop through every training example
            for x in X:
                # Do the E step for this example
                alpha = self.forward(x, normalize = True)
                beta = self.backward(x, normalize = True)
                # Loop through every index in the example
                for t in range(1, len(x)+1):
                    p1 = np.zeros(self.L)
                    # Loop up to l to compute marginal probability 1
                    for a in range(self.L):
                        # Update the marginal 1 probability
                        p1[a] = beta[t][a] * alpha[t][a]
                    # normalize
                    p1 /= np.sum(p1)
                    # Update everything except A_numerator
                    O_denominator += p1
                    O_numerator[:,x[t-1]] += p1
                    # It is offset from the denominator of A
                    if (t != len(x)):
                        A_denominator += p1
                # Loop through every index in the example
                for t in range(1, len(x)):
                    p2 = np.zeros((self.L, self.L))
                    # Loop up to l,l to compute marginal probability 2
                    for a in range(self.L):
                        for b in range(self.L):
                            # Update the marginal 2 probability
                            p2[a][b] = alpha[t-1][a] * beta[t][b] * self.O[b][x[t]] * self.A[a][b]
                    # normalize
                    p2 /= np.sum(p2)
                    # Update the numerator of A
                    A_numerator += p2
            # Set A and O using the equations for the numerators and the denominators.
            for j in range(len(A_numerator[0])):
                A_numerator[:, j] /= A_denominator
            for j in range(len(O_numerator[0])):
                O_numerator[:, j] /= O_denominator
            #normalize
            self.A = A_numerator
            self.O = O_numerator
            # ta said that the solution code included this normalization step but it seems kind of redundant and incorrect
            # for row in self.A:
            #     row /= np.sum(row)
            # for row in self.O:
            #     row /= np.sum(row)
        self.A = self.A.tolist()
        self.O = self.O.tolist()

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        i = 0
        state = np.random.choice(np.array([i for i in range(self.L)]), p = np.array(self.A_start))
        while(i < M):
            i = i+1
            state = np.random.choice(np.array([i for i in range(self.L)]), p =np.array(self.A[state])/np.sum(self.A[state]))
            output = np.random.choice(np.array([o for o in range(self.D)]), p = np.array(self.O[state])/np.sum(self.O[state]))
            emission.append(output)
            states.append(state)

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
