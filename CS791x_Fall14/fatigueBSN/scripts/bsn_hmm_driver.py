#!/usr/bin/python

__author__ = 't'

import collections

from ghmm import *
import pylab as pl

from bsn_data_point import *
import merge_sensor_data as msd
import reaction_test_visualizer as rtv

def main():
    data_points, fmt = msd.unpack_binary_data_into_list('../fatigue_test_data/Luke/12_03_2002_merged.dat')

    # print 'number of time stamps: ', len(data_points)
    # print 'duration of experiment: ',\
    #     data_points[len(data_points) - 1][0] - data_points[0][0]

    data = rtv.read_reaction_data_into_list(rtv.LUKE_FILE)
    labels = rtv.generate_labels_with_times(data, 1.1)

    all_labels = []
    j = 0
    for i in range(len(data_points)):
        if labels[j][0] < data_points[i][0]:
            j += 1
        all_labels.append(labels[j][1])

    sampled_data = []
    for i in range(len(data_points)):
        sampled_data.append(BSNDataPoint(
            timestamp=data_points[i][0],
            label=all_labels[i],
            heart_rate=data_points[i][2],
            low_alpha_frequency=data_points[i][3],
            high_alpha_frequency=data_points[i][4],
            torso_position=data_points[i][1]
        ))

    # aka sigma
    emission_aplphabet = BSN_EMISSION_ALPHABET

    # aka pi
    initial_state_model = [0.9,  # non-fatigue
                           0.1]  # fatigue
    # aka A
    a = 0.5
    b = 0.9
    state_transition_matrix = [[a,       1.0 - a],    # non-fatigue
                               [1.0 - b,       b]]    # fatigue

    non_fatigue_emission_probabilities = \
        [
            0.25,  # aaaa
            0.10,  # aaab
            0.10,  # aaba
            0.05,  # aabb
            0.10,  # abaa
            0.05,  # abab
            0.05,  # abba
            0.01,  # abbb
            0.10,  # baaa
            0.05,  # baab
            0.05,  # baba
            0.01,  # babb
            0.05,  # bbaa
            0.01,  # bbab
            0.01,  # bbba
            0.01   # bbbb
        ]
    # print sum(non_fatigue_emission_probabilities)

    fatigue_emission_probabilities = \
        [
            0.01,  # aaaa
            0.01,  # aaab
            0.01,  # aaba
            0.05,  # aabb
            0.01,  # abaa
            0.05,  # abab
            0.05,  # abba
            0.10,  # abbb
            0.01,  # baaa
            0.05,  # baab
            0.05,  # baba
            0.10,  # babb
            0.05,  # bbaa
            0.10,  # bbab
            0.10,  # bbba
            0.25   # bbbb
        ]
    # print sum(fatigue_emission_probabilities)

    # if not probabilities_sum_to_one(non_fatigue_emission_probabilities):
    #     raise Exception('FIX THE NON-FATIGUE PROBS!!!!')
    # if not probabilities_sum_to_one(fatigue_emission_probabilities):
    #     raise Exception('FIX THE FATIGUE PROBS!!!!')

    # aka B
    emission_likelihood_matrix = [non_fatigue_emission_probabilities,
                                  fatigue_emission_probabilities]

    # simply initialize the model
    bsn_hmm_model = HMMFromMatrices(
        emission_aplphabet,         # sigma: the alphabet
        DiscreteDistribution(       # the DiscreteDistribution is for ghmm's
            emission_aplphabet      # internals
        ),
        state_transition_matrix,     # A: state transitions model matrix
        emission_likelihood_matrix,  # B: emission likelihoods matrix
        initial_state_model          # PI: initial state model
    )

    # print "State estimation on test_list after retraining:
    sample_emissions = EmissionSequence(
        BSN_EMISSION_ALPHABET,
        [x.to_discrete_emission_string() for x in sampled_data]
    )

    bsn_hmm_model.baumWelch(sample_emissions)
    print bsn_hmm_model

    predicted_labels, log_likelihood = bsn_hmm_model.viterbi(sample_emissions)
    determine_percent_correct(
        text_labels_to_numerical([x.label for x in sampled_data]),
        predicted_labels
    )
    # print test_labels

    pl.plot(range(len(predicted_labels)), predicted_labels, label='Predicted')
    pl.plot(range(len(predicted_labels)), text_labels_to_numerical([x.label for x in sampled_data]), label='Actual', color='r')
    pl.xlabel("Time Step")
    pl.ylabel("BSN Wearer State (Non-Fatigued = 0, Fatigued = 1)")
    pl.ylim([-0.2,1.2])
    pl.title("BSN Fatigue State Prediction")
    legend = pl.legend(loc='best', ncol=2, shadow=None)
    legend.get_frame().set_facecolor('#00FFCC')
    pl.show()


def determine_percent_correct(input_list, output_list):
    num_correct = 0
    for i in range(0, len(input_list)):
        if input_list[i] == output_list[i]:
            num_correct += 1

    print "{} / {} = {}%".format(num_correct, len(input_list), float(num_correct) * 100.0 / float(len(input_list)))

    return float(num_correct) / float(len(input_list))


def determine_most_likely_state(state_labels):
    label_counts = collections.defaultdict(lambda: 0)

    most_frequent_label = label_counts[0]
    high_frequency = 0
    for label in state_labels:
        if label_counts[label] > high_frequency:
            most_frequent_label = label
            high_frequency = label_counts[label]

    return most_frequent_label


def text_labels_to_numerical(text_labels):
    new_list = []
    for label in text_labels:
        if label == NON_FATIGUE_LABEL:
            new_list.append(0)
        else:
            new_list.append(1)
    return new_list


if __name__ == "__main__":
    main()
