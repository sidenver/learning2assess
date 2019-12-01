from allennlp.training.metrics import Metric
from typing import Dict, Optional, List, Union, Any, Tuple
import torch
import pytrec_eval
import numpy as np
from .hTBG import hTBG
import json

from allennlp.common.registrable import Registrable


class Converter(Registrable):
    """
    a Converter takes a raw string class and turns it into a numeric score
    """
    pass


@Converter.register('reddit_convert')
class RedditConvert(Converter):

    def __init__(self, mapping: Optional[Dict[str, int]] = None):
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = {'a': 0,
                            'b': 1,
                            'c': 2,
                            'd': 3}

    def __call__(self, raw: str) -> Union[int, None]:
        return self.mapping[raw] if raw in self.mapping else None


@Converter.register('reddit_convert_zero')
class RedditConvertZero(Converter):

    def __init__(self, mapping: Optional[Dict[str, int]] = None):
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = {'a': 0,
                            'b': 1,
                            'c': 2,
                            'd': 3}

    def __call__(self, raw: str) -> int:
        return self.mapping[raw] if raw in self.mapping else 0


@Converter.register('binary_convert')
class BinaryConvert(Converter):

    def __init__(self, mapping: Optional[Dict[str, int]] = None):
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = {'a': 0,
                            'b': 0,
                            'c': 0,
                            'd': 1}

    def __call__(self, raw: str) -> int:
        return self.mapping[raw] if raw in self.mapping else 0


@Converter.register('exponential_convert')
class ExponentialConvert(Converter):

    def __init__(self, mapping: Optional[Dict[str, int]] = None):
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = {'a': 0,
                            'b': 2,
                            'c': 4,
                            'd': 8}

    def __call__(self, raw: str) -> int:
        return self.mapping[raw] if raw in self.mapping else 0


class Scorer(Registrable):
    """
    a Score takes a support (a list of tuple of a class and confidence) and turns it into a probability
    """
    pass


@Scorer.register('expect_score')
class ExpectScorer(Scorer):
    """
    For hTBG, calculate probability with the max
    """

    def __init__(self, converter: Converter):
        self.converter = converter
        self.max = max(list(self.converter.mapping.values()))

    def __call__(self, support: List[List[Any]]) -> float:
        if len(support) == 0:
            return 0.
        score = 1 - np.prod([1 - ((self.converter(annotation) * confidence) / self.max)
                            for annotation, confidence in support])
        return score


@Scorer.register('max_score')
class MaxScorer(Scorer):
    """
    For hTBG, calculate probability with the max
    """

    def __init__(self, converter: Converter):
        self.converter = converter
        self.max = max(list(self.converter.mapping.values()))

    def __call__(self, support: List[List[Any]]) -> float:
        if len(support) == 0:
            return 0.
        score = max([self.converter(annotation) * confidence
                     for annotation, confidence in support])
        return score / self.max


@Scorer.register('average_score')
class AverageScorer(Scorer):
    """
    For hTBG, calculate probability with average
    """

    def __init__(self, converter: Converter):
        self.converter = converter
        self.max = max(list(self.converter.mapping.values()))

    def __call__(self, support: List[List[Any]]) -> float:
        if len(support) == 0:
            return 0.
        score = sum([self.converter(annotation) * confidence
                     for annotation, confidence in support])
        return score / (self.max * len(support))


@Scorer.register('zero_score')
class ZeroScorer(Scorer):
    """
    mimic TBG
    """

    def __init__(self, converter: Converter):
        self.converter = converter

    def __call__(self, support: List[List[Any]]) -> float:
        return 0.


@Metric.register('stateful_ndcg')
class StatefulNDCG(Metric):

    def __init__(self, converter: Converter) -> None:
        self.predictions_with_gold = []
        self.gold_labels = []
        self._converter = converter

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: List[str],
                 mask: Optional[torch.Tensor] = None) -> None:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        prediction_list = predictions.tolist()
        gold_label_list = [self._converter(raw_label) for raw_label in gold_labels]
        self.predictions_with_gold += [(pre, gold)
                                       for pre, gold in zip(prediction_list, gold_label_list)
                                       if gold is not None]

    def get_metric(self, reset: bool) -> Dict[str, float]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        if reset:
            ndcg_score = self.evaluate()
            self.reset()
            print(ndcg_score)
            return ndcg_score
        else:
            return {}

    def reset(self) -> None:
        self.predictions_with_gold = []
        self.gold_labels = []

    def evaluate(self) -> Dict[str, float]:
        if len(self.predictions_with_gold) == 0:
            return {}

        if sum([gold for pre, gold in self.predictions_with_gold]) == 0:
            return {}

        qrel = {
            'q1': {'d{}'.format(i + 1): int(gold)
                   for i, (pre, gold) in enumerate(self.predictions_with_gold)}
        }
        run = {
            'q1': {'d{}'.format(i + 1): pre
                   for i, (pre, gold) in enumerate(self.predictions_with_gold)}
        }
        print(len(self.predictions_with_gold))

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg', 'ndcg_cut'})
        results = evaluator.evaluate(run)
        return results['q1']


@Metric.register('stateful_time_biased_gain')
class StatefulTimeBiasedGain(Metric):

    def __init__(self, converter: Converter,
                 p_click_true: float = 0.64, p_click_false: float = 0.39,
                 p_save_true: float = 0.77, p_save_false: float = 0.27, t_summary: float = 4.4,
                 t_alpha: float = 0.018, t_beta: float = 7.8) -> None:
        self.predictions_with_gold = []
        self.gold_labels = []
        self._converter = converter
        self.p_click_true = p_click_true
        self.p_click_false = p_click_false
        self.p_save_true = p_save_true
        self.p_save_false = p_save_false
        self.t_summary = t_summary
        self.t_alpha = t_alpha
        self.t_beta = t_beta

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: List[str],
                 word_counts: List[int],
                 mask: Optional[torch.Tensor] = None) -> None:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        prediction_list = predictions.tolist()
        gold_label_list = [self._converter(raw_label) for raw_label in gold_labels]
        self.predictions_with_gold += [(pre, gold, word_count)
                                       for pre, gold, word_count in zip(prediction_list, gold_label_list, word_counts)
                                       if gold is not None]

    def get_metric(self, reset: bool) -> Dict[str, float]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        if reset:
            time_biased_gain_score = self.evaluate()
            self.reset()
            print(time_biased_gain_score)
            return time_biased_gain_score
        else:
            return {}

    def reset(self) -> None:
        self.predictions_with_gold = []
        self.gold_labels = []

    def calculate_time_biased_gain(self, gains: List[float], time_at_k: List[float], t_half_life: float):
        gains = np.array(gains)
        time_at_k = np.array(time_at_k)
        discount = np.exp(-time_at_k * np.log(2) / t_half_life)
        return np.dot(gains, discount)

    def evaluate(self) -> Dict[str, float]:
        if len(self.predictions_with_gold) == 0:
            return {}

        if sum([gold for pre, gold, word_count in self.predictions_with_gold]) == 0:
            return {}

        sorted_predictions_with_gold = sorted(self.predictions_with_gold, key=lambda x: x[0], reverse=True)
        gains = [self.p_click_true * self.p_save_true if gold > 0 else 0.
                 for pre, gold, word_count in sorted_predictions_with_gold]

        time_at_k = []
        # time at k is the time needed to reach rank k, thus the time to reach rank 1 is always 0
        time_at_k.append(0.)
        for pre, gold, word_count in sorted_predictions_with_gold[:-1]:
            p_click = self.p_click_true if gold > 0 else self.p_click_false
            time = time_at_k[-1] + self.t_summary + (self.t_alpha * word_count + self.t_beta) * p_click
            time_at_k.append(time)

        t_half_lives = [224., 1800.]

        return {'TBG_{}'.format(str(int(t_half_life))):
                self.calculate_time_biased_gain(gains, time_at_k, t_half_life)
                for t_half_life in t_half_lives}


@Metric.register('stateful_hierarchical_time_biased_gain')
class StatefulHierarchicalTimeBiasedGain(StatefulTimeBiasedGain):

    def __init__(self, converter: Converter, scorer: Scorer, doc_ranking_mode: str = 'sort',
                 p_click_true: float = 0.64, p_click_false: float = 0.39,
                 p_save_true: float = 0.77, p_save_false: float = 0.27, t_summary: float = 4.4,
                 t_alpha: float = 0.018, t_beta: float = 7.8) -> None:
        super().__init__(converter=converter,
                         p_click_true=p_click_true, p_click_false=p_click_false,
                         p_save_true=p_save_true, p_save_false=p_save_false,
                         t_summary=t_summary, t_alpha=t_alpha, t_beta=t_beta)
        self._support_scorer = scorer
        assert doc_ranking_mode in ['sort', 'forward', 'backward']
        self.doc_ranking_mode = doc_ranking_mode

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: List[str],
                 word_counts: List[List[int]],
                 supports: List[List[List[Any]]],
                 document_attentions: torch.Tensor,
                 meta: List[Dict[str, Any]],
                 mask: Optional[torch.Tensor] = None) -> None:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        prediction_list = predictions.tolist()
        gold_label_list = [self._converter(raw_label) for raw_label in gold_labels]
        # support is the probability for stopping at the post
        supports_list = [[self._support_scorer(support) for support in support_list]
                         for support_list in supports]
        document_attentions_list = document_attentions.tolist()
        # word_count is a list of document word count, support here is a list of support probability
        self.predictions_with_gold += [(pre, gold, word_count, support, document_attention)
                                       for pre, gold, word_count, support, document_attention
                                       in zip(prediction_list, gold_label_list,
                                              word_counts, supports_list,
                                              document_attentions_list)
                                       if gold is not None]

    def calculate_expect_word_read(self,
                                   word_count: List[int],
                                   support: List[float],
                                   document_attention: List[float]) -> float:
        word_support_attention = list(zip(word_count, support, document_attention))
        if self.doc_ranking_mode == 'sort':
            word_support_attention = sorted(word_support_attention, key=lambda x: x[-1], reverse=True)
        if self.doc_ranking_mode == 'backward':
            word_support_attention = reversed(word_support_attention)

        cumulative_skipping_prob = 1
        expect_word_read = 0
        for doc_word_count, stopping_prob, attention in word_support_attention:
            expect_word_read += doc_word_count * cumulative_skipping_prob
            cumulative_skipping_prob *= (1 - stopping_prob)
        return expect_word_read

    def evaluate(self) -> Dict[str, float]:
        if len(self.predictions_with_gold) == 0:
            return {}

        if sum([gold for pre, gold, word_count, support, document_attention
                in self.predictions_with_gold]) == 0:
            return {}

        sorted_predictions_with_gold = sorted(self.predictions_with_gold, key=lambda x: x[0], reverse=True)
        gains = [self.p_click_true * self.p_save_true if gold > 0 else 0.
                 for pre, gold, word_count, support, document_attention in sorted_predictions_with_gold]

        time_at_k = []
        # time at k is the time needed to reach rank k, thus the time to reach rank 1 is always 0
        time_at_k.append(0.)
        for pre, gold, word_count, support, document_attention in sorted_predictions_with_gold[:-1]:
            p_click = self.p_click_true if gold > 0 else self.p_click_false
            expect_word_read = self.calculate_expect_word_read(word_count, support, document_attention)
            time = time_at_k[-1] + self.t_summary + (self.t_alpha * expect_word_read + self.t_beta) * p_click
            time_at_k.append(time)

        t_half_lives = [224., 1800.]

        return {'hTBG_{}'.format(str(int(t_half_life))):
                self.calculate_time_biased_gain(gains, time_at_k, t_half_life)
                for t_half_life in t_half_lives}


@Metric.register('hierarchical_time_biased_gain')
class HierarchicalTimeBiasedGain(StatefulTimeBiasedGain):

    def __init__(self, converter: Converter, scorer: Scorer, doc_ranking_mode: str = 'sort',
                 p_click_true: float = 0.64, p_click_false: float = 0.39,
                 p_save_true: float = 0.77, p_save_false: float = 0.27, t_summary: float = 4.4,
                 t_alpha: float = 0.018, t_beta: float = 7.8,
                 t_half_lives: List[float] = [224., 600., 1200., 1800.]) -> None:
        super().__init__(converter=converter,
                         p_click_true=p_click_true, p_click_false=p_click_false,
                         p_save_true=p_save_true, p_save_false=p_save_false,
                         t_summary=t_summary, t_alpha=t_alpha, t_beta=t_beta)
        self._support_scorer = scorer
        self.t_half_lives = t_half_lives
        assert doc_ranking_mode in ['sort', 'forward', 'backward']
        self.doc_ranking_mode = doc_ranking_mode
        self.relevance_dict = {}
        self.prediction_dict = {}

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: List[str],
                 word_counts: List[List[int]],
                 supports: List[List[List[Any]]],
                 document_attentions: torch.Tensor,
                 meta: List[Dict[str, Any]],
                 mask: Optional[torch.Tensor] = None) -> None:
        user_ids = [meta_dict['user_id'] for meta_dict in meta]
        post_ids_list = [meta_dict['post_id'] for meta_dict in meta]
        prediction_list = predictions.tolist()
        gold_label_list = [self._converter(raw_label) for raw_label in gold_labels]
        # support is the probability for stopping at the post
        supports_list = [[self._support_scorer(support) for support in support_list]
                         for support_list in supports]
        document_attentions_list = document_attentions.tolist()

        self.relevance_dict.update(self.build_relevance_dict(user_ids,
                                                             post_ids_list,
                                                             gold_label_list,
                                                             supports_list,
                                                             word_counts))

        self.prediction_dict.update(self.build_prediction_dict(user_ids,
                                                               post_ids_list,
                                                               prediction_list,
                                                               document_attentions_list))

    def build_relevance_dict(self,
                             user_ids: List[str],
                             post_ids_list: List[List[str]],
                             gold_label_list: List[int],
                             supports_list: List[List[float]],
                             word_counts_list: List[List[int]]):
        return {
            user_id: [gold_label,
                      {post_id: [support, word_count]
                       for post_id, support, word_count
                       in zip(post_ids, supports, word_counts)}]
            for user_id, gold_label, post_ids, supports, word_counts
            in zip(user_ids, gold_label_list, post_ids_list, supports_list, word_counts_list)
        }

    def build_prediction_dict(self,
                              user_ids: List[str],
                              post_ids_list: List[List[str]],
                              prediction_list: List[float],
                              document_attentions_list: List[List[float]]):
        return {
            user_id: [prediction,
                      {post_id: document_attention
                       for post_id, document_attention
                       in zip(post_ids, self.sort_from_ranking_mode(document_attentions))}]
            for user_id, prediction, post_ids, document_attentions
            in zip(user_ids, prediction_list, post_ids_list, document_attentions_list)
        }

    def sort_from_ranking_mode(self, document_attentions):
        num_docs = len(document_attentions)
        if self.doc_ranking_mode == 'forward':
            return [num_docs - index for index, _ in enumerate(document_attentions)]
        if self.doc_ranking_mode == 'backward':
            return [index for index, _ in enumerate(document_attentions)]
        return document_attentions

    def reset(self) -> None:
        self.relevance_dict = {}
        self.prediction_dict = {}

    def evaluate(self) -> Dict[str, float]:
        if len(self.relevance_dict) == 0:
            return {}

        htbg = hTBG(relevance={"q_1": self.relevance_dict},
                    prediction={"q_1": self.prediction_dict},
                    p_click_true=self.p_click_true, p_click_false=self.p_click_false,
                    p_save_true=self.p_save_true, p_save_false=self.p_save_false, t_summary=self.t_summary,
                    t_alpha=self.t_alpha, t_beta=self.t_beta,
                    t_half_lives=self.t_half_lives)

        # print("relevance")
        # print(json.dumps(self.relevance_dict))
        # print("prediction")
        # print(json.dumps(self.prediction_dict))

        results = htbg.evaluate()["q_1"]

        best_possible_results = htbg.evaluate_best()["q_1"]

        outputs = {}
        outputs.update({
            'hTBG_{}'.format(str(int(t_half_life))): results[t_half_life]
            for t_half_life in results
        })

        outputs.update({
            'hTBG_{}_best'.format(str(int(t_half_life))): best_possible_results[t_half_life]
            for t_half_life in best_possible_results
        })

        return outputs
