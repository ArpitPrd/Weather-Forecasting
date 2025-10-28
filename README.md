# Weather-Forecasting
Assignment 4: COL333

Multi-Start (main): The main function is now a master loop. It runs a specified number of RESTARTS (e.g., 5-10). Each restart begins with a new, randomly initialized network.

Random Initialization (initialize_cpts_randomly): I've added a new function that initializes all CPTs with random (but normalized) probabilities. This ensures each restart explores a different part of the solution space.

Mini-Batch EM (learn_parameters_em): The main EM loop no longer iterates over the entire dataset. Instead, each "E-step" consists of processing a small, random batch_size of data. This is much faster per iteration and the injected randomness helps the algorithm "jiggle" out of poor local maxima.

Log-Likelihood Scorer (calculate_observed_log_likelihood): To compare the results of each restart, we need a "scorer". This function calculates the total log-likelihood of the entire dataset given the final trained network.

Robust Posterior Calculation (calculate_posterior_and_likelihood): The old calculate_posterior_log function has been upgraded. It now returns two things:

    The normalized posterior (for the E-step).

    The log-likelihood of the single observed data point (for the scorer).

Time-Keeping: The main loop now includes a timer. It will automatically stop new restarts if it gets close to the 110-second mark, ensuring it finishes within the 2-minute (120s) time limit.