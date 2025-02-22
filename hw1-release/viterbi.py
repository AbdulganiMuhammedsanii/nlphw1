# Name(s): Abdulgani Muhammedsani, Edwin Dake
# Netid(s): ed433, amm546
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np


def viterbi(model, observation, tags):
    """
  Returns the model's predicted tag sequence for a particular observation.
  Use `get_tag_likelihood` method to obtain model scores at each iteration.

  Input: 
    model: HMM model
    observation: List[String]
    tags: List[String]
  Output:
    predictions: List[String]
  """
    real_tags = [t for t in tags if t != "qf"]
    N = len(observation)        # Number of tokens
    K = len(real_tags)          # Number of actual (non-"qf") tags
   
    dp = np.full((N, K), -np.inf)
    backpointer = np.zeros((N, K), dtype=int)

   
    for j, tag_j in enumerate(real_tags):
        dp[0, j] = model.get_tag_likelihood(
            predicted_tag=tag_j,
            previous_tag="qf",    # or any dummy, because i=0 uses start_state_probs
            document=observation,
            i=0
        )
        backpointer[0, j] = 0  # no predecessor at time 0

    for i in range(1, N):
        for j, tag_j in enumerate(real_tags):
            best_score = -np.inf
            best_prev_index = 0
            for k, tag_k in enumerate(real_tags):
                score = (dp[i-1, k]
                         + model.get_tag_likelihood(
                               predicted_tag=tag_j,
                               previous_tag=tag_k,
                               document=observation,
                               i=i
                           ))
                if score > best_score:
                    best_score = score
                    best_prev_index = k
            dp[i, j] = best_score
            backpointer[i, j] = best_prev_index

    best_final_score = -np.inf
    best_final_index = 0
    for k, tag_k in enumerate(real_tags):
        score_to_qf = (dp[N-1, k]
                       + model.get_tag_likelihood(
                             predicted_tag="qf",
                             previous_tag=tag_k,
                             document=observation,
                             i=N   # pass i=N so no emission is used, only transition
                         ))
        if score_to_qf > best_final_score:
            best_final_score = score_to_qf
            best_final_index = k

    best_path = [best_final_index]  # which real_tags[] index is best at i = N-1
    for i in range(N-1, 0, -1):
        best_path.append(backpointer[i, best_path[-1]])
    best_path.reverse()

    predictions = [real_tags[i] for i in best_path]

    return predictions
