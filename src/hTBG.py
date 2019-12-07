"""Hierarchical Time-biased Gain

Usage:
    hTBG.py --relevance=<json> --prediction=<json> [options]
    hTBG.py --help

Options:
    -h --help                    show this help message and exit
    -t --tbg                     run with non-hierarchical version of hTBG, stopping probabilities set to zero
    -v --verbose                 be verbose, print out hyper-parameters
    --override==<str>            json str to override default hTBG parameters [default: {}]
    -o <path>, --output <path>   output path [default ./out.json]

Example:
    python hTBG.py --relevance ./hTBG_test/truth.json --prediction ./hTBG_test/prediction.json \
    --override '{"t_half_lives": [3, 5, 10]}'

=========================
The relevance structure is:
{
    "query_name": {
        "higher_level_name": [
            true_score (0 or 1),
            {
                "lower_level_name": [stopping_probability, cost_of_lower_level],
                ...
            }
        ],
        ...
    },
    ...
}

An example of
truth.json:
{
   "q_1": {
      "user_1": [
         1,
         {
            "doc_1": [
               0.2,
               56
            ],
            "doc_2": [
               0.1,
               194
            ]
         }
      ],
      "user_2": [
         0,
         {
            "doc_1": [
               0,
               35
            ],
            "doc_2": [
               0,
               14
            ],
            "doc_3": [
               0,
               46
            ]
         }
      ],
      "user_3": [
         1,
         {
            "doc_1": [
               0,
               35
            ],
            "doc_2": [
               0.5,
               14
            ],
            "doc_3": [
               0.7,
               46
            ]
         }
      ]
   },
   "q_2": {
      "user_1": [
         0,
         {
            "doc_1": [
               0,
               56
            ],
            "doc_2": [
               0,
               194
            ]
         }
      ],
      "user_2": [
         1,
         {
            "doc_1": [
               0.3,
               35
            ],
            "doc_2": [
               0.3,
               14
            ],
            "doc_3": [
               0.1,
               46
            ]
         }
      ],
      "user_3": [
         1,
         {
            "doc_1": [
               0.3,
               35
            ],
            "doc_2": [
               0.3,
               14
            ],
            "doc_3": [
               0.1,
               46
            ]
         }
      ]
   }
}
=========================
The prediction structure is:
{
    "query_name": {
        "higher_level_name": [
            predicted_score (float),
            {
                "lower_level_name": predicted_lower_level_score (float),
                ...
            }
        ],
        ...
    },
    ...
}

An example of
prediction.json:
{
  "q_1": {
     "user_1": [
        0.56,
        {
           "doc_1": 0.6,
           "doc_2": 0.4
        }
     ],
     "user_2": [
        0.45,
        {
           "doc_1": 0.1,
           "doc_2": 0.3,
           "doc_3": 0.4
        }
     ],
     "user_3": [
        0.46,
        {
           "doc_1": 0.5,
           "doc_2": 0.3,
           "doc_3": 0.4
        }
     ]
  },
  "q_2": {
     "user_1": [
        0.56,
        {
           "doc_1": 0.2,
           "doc_2": 0.6
        }
     ],
     "user_2": [
        0.45,
        {
           "doc_1": 0.1,
           "doc_2": 0.5,
           "doc_3": 0.6
        }
     ],
     "user_3": [
        0.43,
        {
           "doc_1": 0.1,
           "doc_2": 0.5,
           "doc_3": 0.6
        }
     ]
  }
}
=========================
output format
{
    "query_name": {
        t_half_life: score,
        ...
    },
    ...
}


example output
evaluation results:
{'q_1': {3: 0.5248706964598764, 5: 0.588460647126441, 10: 0.7099210881142366},
 'q_2': {3: 0.06428104166337158, 5: 0.17063498548694406, 10: 0.3878882375890544}}
best possible values:
{'q_1': {3: 0.543081360426777, 5: 0.6180888697456681, 10: 0.7412800897670984},
 'q_2': {3: 0.5406284846869924, 5: 0.6143850709821594, 10: 0.7375797438106515}}


"""

import json
from docopt import docopt
import numpy as np
from typing import Dict, Optional, List, Union, Any, Tuple
from itertools import permutations
from pprint import pprint


default_htbg_parameters = {
    "p_click_true": 0.64,
    "p_click_false": 0.39,
    "p_save_true": 0.77,
    "p_save_false": 0.27,
    "t_summary": 4.4,
    "t_alpha": 0.018,
    "t_beta": 7.8,
    "t_half_lives": [224., 1800.]
}


class hTBG:
    def __init__(self, relevance: Dict[str, Any], prediction: Dict[str, Any],
                 p_click_true: float = 0.64, p_click_false: float = 0.39,
                 p_save_true: float = 0.77, p_save_false: float = 0.27, t_summary: float = 4.4,
                 t_alpha: float = 0.018, t_beta: float = 7.8,
                 t_half_lives: List[float] = [224., 1800.],
                 hierarchical: bool = True,
                 verbose: bool = False):
        self.relevance = relevance
        self.prediction = prediction
        self.hierarchical = hierarchical
        self.verbose = verbose
        self.p_click_true = p_click_true
        self.p_click_false = p_click_false
        self.p_save_true = p_save_true
        self.p_save_false = p_save_false
        self.t_summary = t_summary
        self.t_alpha = t_alpha
        self.t_beta = t_beta
        self.t_half_lives = t_half_lives

    def calculate_time_biased_gain(self, gains: List[float], time_at_k: List[float], t_half_life: float):
        gains = np.array(gains)
        time_at_k = np.array(time_at_k)
        discount = np.exp(-time_at_k * np.log(2) / t_half_life)
        if self.verbose:
            print('gains', gains)
            print('time_at_k', time_at_k)
            print('discount', discount)
        return np.dot(gains, discount)

    def calculate_expect_word_read(self,
                                   relevance_doc_dict: Dict[str, List[float]],
                                   prediction_score_dict: Dict[str, float]
                                   ) -> float:
        # assert that the same user has the same set of documents
        assert set(relevance_doc_dict.keys()) == set(prediction_score_dict.keys())

        word_support_score = [(relevance_doc_dict[doc_id][1],  # lower level cost, or word count
                               relevance_doc_dict[doc_id][0],  # stopping probability
                               prediction_score_dict[doc_id]) for doc_id in relevance_doc_dict]
        if self.hierarchical:
            word_support_score = sorted(word_support_score, key=lambda x: x[-1], reverse=True)

            cumulative_skipping_prob = 1
            expect_word_read = 0
            for doc_word_count, stopping_prob, score in word_support_score:
                expect_word_read += doc_word_count * cumulative_skipping_prob
                cumulative_skipping_prob *= (1 - stopping_prob)
            return expect_word_read
        else:
            return sum([doc_word_count for doc_word_count, _, _ in word_support_score])

    def evaluate_query(self,
                       relevance: Dict[str, List[Any]],
                       prediction: Dict[str, List[Any]]) -> Dict[float, float]:
        # assert that they have the same set of users
        assert set(relevance.keys()) == set(prediction.keys())
        gold_label_list = [relevance[user][0] for user in relevance]
        prediction_list = [prediction[user][0] for user in relevance]
        relevance_doc_dict_list = [relevance[user][1] for user in relevance]
        prediction_score_dict_list = [prediction[user][1] for user in relevance]

        predictions_with_gold = [(prediction, gold_label, relevance_doc_dict, prediction_score_dict)
                                 for prediction, gold_label, relevance_doc_dict, prediction_score_dict
                                 in zip(prediction_list, gold_label_list,
                                        relevance_doc_dict_list,
                                        prediction_score_dict_list)
                                 if gold_label is not None]

        sorted_predictions_with_gold = sorted(predictions_with_gold, key=lambda x: x[0], reverse=True)

        gains = [self.p_click_true * self.p_save_true if gold_label > 0 else 0.
                 for prediction, gold_label, relevance_doc_dict, prediction_score_dict
                 in sorted_predictions_with_gold]

        time_at_k = []
        # time at k is the time needed to reach rank k, thus the time to reach rank 1 is always 0
        time_at_k.append(0.)
        for prediction, gold_label, relevance_doc_dict, prediction_score_dict in sorted_predictions_with_gold[:-1]:
            p_click = self.p_click_true if gold_label > 0 else self.p_click_false
            expect_word_read = self.calculate_expect_word_read(relevance_doc_dict, prediction_score_dict)
            if self.verbose:
                print(expect_word_read)
            time = time_at_k[-1] + self.t_summary + (self.t_alpha * expect_word_read + self.t_beta) * p_click
            time_at_k.append(time)

        return {_t_half_life: self.calculate_time_biased_gain(gains, time_at_k, _t_half_life)
                for _t_half_life in self.t_half_lives}

    def best_possible_htbg(self, relevance: Dict[str, List[Any]]) -> Dict[float, float]:
        gold_label_dict = {user: relevance[user][0] for user in relevance}
        relevance_doc_dict_dict = {user: relevance[user][1] for user in relevance}
        user_time = {}

        for user, relevance_doc_dict in relevance_doc_dict_dict.items():
            cost_stopping_prob = {doc_id: (relevance_doc_dict[doc_id][1],  # lower level cost, or word count
                                           relevance_doc_dict[doc_id][0])  # stopping probability
                                  for doc_id in relevance_doc_dict}
            cost_non_zero_stopping_doc_id = set([doc_id
                                                for doc_id, item in cost_stopping_prob.items()
                                                if item[-1] > 0.])
            cost_zero_stopping_doc_id = set([doc_id
                                             for doc_id, item in cost_stopping_prob.items()
                                             if item[-1] == 0.])

            all_prod_word = np.prod([cost_stopping_prob[doc_id][0] for doc_id in cost_stopping_prob])
            prediction_score_dict = {doc_id: word_stopprob[1]*all_prod_word/word_stopprob[0]
                                     for doc_id, word_stopprob in cost_stopping_prob.items()
                                     if doc_id in cost_non_zero_stopping_doc_id}
            prediction_score_dict.update({doc_id: -np.inf for doc_id in cost_zero_stopping_doc_id})
            expect_word_read = self.calculate_expect_word_read(relevance_doc_dict, prediction_score_dict)

            user_time[user] = expect_word_read

        # relevant items will always rank before irrelevant items
        # within relevant items, the shorter time ones rank higher
        # TODO what about graded relevance?
        if self.verbose:
            print(user_time)
        sorted_users = sorted(gold_label_dict.keys(),
                              key=lambda user: (gold_label_dict[user], -user_time[user]),
                              reverse=True)

        gains = [self.p_click_true * self.p_save_true if gold_label_dict[user] > 0 else 0.
                 for user
                 in sorted_users]

        time_at_k = []
        # time at k is the time needed to reach rank k, thus the time to reach rank 1 is always 0
        time_at_k.append(0.)
        for user in sorted_users[:-1]:
            p_click = self.p_click_true if gold_label_dict[user] > 0 else self.p_click_false
            expect_word_read = user_time[user]
            time = time_at_k[-1] + self.t_summary + (self.t_alpha * expect_word_read + self.t_beta) * p_click
            time_at_k.append(time)

        return {_t_half_life: self.calculate_time_biased_gain(gains, time_at_k, _t_half_life)
                for _t_half_life in self.t_half_lives}

    def evaluate(self) -> Dict[str, Dict[float, float]]:
        # assert that query keys match
        assert set(self.relevance.keys()) == set(self.prediction.keys())

        if len(self.relevance) == 0:
            return {}

        hTBG_score = {}
        for query_name in self.relevance:
            hTBG_score[query_name] = self.evaluate_query(self.relevance[query_name],
                                                         self.prediction[query_name])

        return hTBG_score

    def evaluate_best(self) -> Dict[str, Dict[float, float]]:
        best_hTBG_score = {}
        for query_name in self.relevance:
            best_hTBG_score[query_name] = self.best_possible_htbg(self.relevance[query_name])

        return best_hTBG_score


if __name__ == '__main__':
    arguments = docopt(__doc__)

    relevance_path = arguments['--relevance']
    prediction_path = arguments['--prediction']
    hierarchical = True if not arguments['--tbg'] else False
    verbose = arguments['--verbose']
    override = json.loads(arguments['--override'])
    output_path = arguments['--output']

    if verbose:
        print(arguments)

    with open(relevance_path, 'r') as input_file:
        # print(input_file.readlines())
        relevance = json.load(input_file)
        if verbose:
            print(relevance)

    with open(prediction_path, 'r') as input_file:
        # print(input_file.readlines())
        prediction = json.load(input_file)
        if verbose:
            print(prediction)

    default_htbg_parameters.update(override)

    if verbose:
        print(default_htbg_parameters)

    htbg = hTBG(relevance=relevance,
                prediction=prediction,
                hierarchical=hierarchical,
                verbose=verbose,
                **default_htbg_parameters)

    print('evaluation results:')
    pprint(htbg.evaluate())

    print('best possible values:')
    pprint(htbg.evaluate_best())
