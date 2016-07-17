from ghmm import Alphabet, LabelDomain

__author__ = 't'

""" Defines a class that retains only essential data for HMM emissions.

    This class is used to contain the raw data only, but can generate the
    string used as an emission label (member of an 'alphabet') for GHMM since
    GHMM backs us into the corner of using strings to represent vectors.
"""

# Fatigue state labels
FATIGUE_LABEL = 'fatigue'
NON_FATIGUE_LABEL = 'non-fatigue'
STATE_ALPHABET = LabelDomain([NON_FATIGUE_LABEL, FATIGUE_LABEL])

# Emission labels
BSN_EMISSION_ALPHABET = Alphabet(
    ['aaaa', 'aaab', 'aaba', 'aabb',
     'abaa', 'abab', 'abba', 'abbb',
     'baaa', 'baab', 'baba', 'babb',
     'bbaa', 'bbab', 'bbba', 'bbbb']
)

# Discretization thresholds
HIGH_ALPHA_FATIGUE_FREQUENCY = 200000  # TODO
LOW_ALPHA_FATIGUE_FREQUENCY = 200000   # TODO
HEART_RATE_FATIGUE_LEVEL = 120      # TODO
TORSO_FATIGUE_ANGLE = 2             # TODO


class BSNDataPoint(object):
    def __init__(self, timestamp, low_alpha_frequency, high_alpha_frequency, heart_rate,
                 torso_position, label=None):
        self.timestamp = timestamp
        self.low_alpha = low_alpha_frequency
        self.high_alpha = high_alpha_frequency
        self.heart_rate = heart_rate
        self.torso = torso_position
        self.label = label

    def get_high_alpha_category(self):
        if self.high_alpha <= HIGH_ALPHA_FATIGUE_FREQUENCY:
            return "a"
        else:
            return "b"

    def get_low_alpha_category(self):
        if self.low_alpha <= LOW_ALPHA_FATIGUE_FREQUENCY:
            return "a"
        else:
            return "b"

    def get_heart_rate_category(self):
        if self.heart_rate <= HEART_RATE_FATIGUE_LEVEL:
            return "a"
        else:
            return "b"

    def get_torso_category(self):
        if abs(self.torso) <= TORSO_FATIGUE_ANGLE:
            return "a"
        else:
            return "b"

    def to_discrete_emission_string(self):
        return "{}{}{}{}".format(
            self.get_low_alpha_category(),
            self.get_high_alpha_category(),
            self.get_heart_rate_category(),
            self.get_torso_category()
        )

    def __str__(self):
        return (
            'Time:                 {}'
            'Low-Alpha Frequency:  {} -> {}'
            'High-Alpha Frequency: {} -> {}'
            'Heart Rate:           {} -> {}'
            'Torso Position:       {} -> {}'
            'State Emitted From:   {}'
        ).format(
            self.timestamp,
            self.low_alpha, self.get_low_alpha_category(),
            self.high_alpha, self.get_high_alpha_category(),
            self.heart_rate, self.get_heart_rate_category(),
            self.torso, self.get_torso_category(),
            self.label
        )