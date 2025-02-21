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
    num_states = len(observation) + 1  # +1 for final qf state
    viterbi_probs = {}
    backpointers = {}

    # Initialize first position (i=0)
    viterbi_probs[0] = {}
    for tag in tags:  # qf can't be start state
        viterbi_probs[0][tag] = model.get_tag_likelihood(tag, None, observation, 0)

    # Fill DP matrices
    for i in range(1, num_states):
        # For last position, only consider transitions to 'qf'
        current_tags = ["qf"] if i == len(observation) else tags

        for curr_tag in current_tags:
            max_prob = float("-inf")
            best_prev_tag = None

            # Check all possible previous tags
            prev_tags = tags if i < len(observation) else tags
            for prev_tag in prev_tags:
                # Here, we compute the probability through this path
                if (i - 1) in viterbi_probs and prev_tag in viterbi_probs[i - 1]:
                    path_prob = viterbi_probs[i - 1][
                        prev_tag
                    ] + model.get_tag_likelihood(
                        curr_tag,
                        prev_tag,
                        observation,
                        i if i < len(observation) else None,
                    )

                    # We are interested in max probs so update if better path found
                    if path_prob > max_prob:
                        max_prob = path_prob
                        best_prev_tag = prev_tag

            if best_prev_tag is not None:
                viterbi_probs[i][curr_tag] = max_prob
                backpointers[i][curr_tag] = best_prev_tag

    # Backtrack to get best sequence
    if not backpointers:
        return []

    # Start from final state
    tags_sequence = []
    curr_pos = len(observation)
    curr_tag = "qf"

    # Build sequence from end to start
    while curr_pos > 0:
        prev_tag = backpointers[curr_pos][curr_tag]
        tags_sequence.append(prev_tag)
        curr_tag = prev_tag
        curr_pos -= 1

    # Reverse since we built backwards
    return tags_sequence[::-1]
