from allennlp.training.metrics import Metric
from typing import Dict, Optional, List, Union
import torch
import pytrec_eval
import numpy as np

from allennlp.common.registrable import Registrable


class Converter(Registrable):
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
                 p_click_true: float, p_click_false: float,
                 p_save_true: float, p_save_false: float, t_summary: float,
                 t_alpha: float, t_beta: float) -> None:
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

        sorted_predictions_with_gold = sorted(self.predictions_with_gold, lambda x: x[0], reverse=True)
        gains = [self.p_click_true * self.p_save_true if gold > 0 else 0.
                 for pre, gold, word_count in sorted_predictions_with_gold]

        time_at_k = []
        # time at k is the time needed to reach rank k, thus the time to reach rank 1 is always 0
        time_at_k.append(0.)
        for pre, gold, word_count in sorted_predictions_with_gold[:-1]:
            p_click = self.p_click_true if gold > 0 else self.p_click_false
            time = time_at_k[-1] + self.t_summary + (self.t_alpha * word_count + self.t_beta) * p_click
            time_at_k.append(time)

        t_half_lives = [224., 500., 1000., 2000.]

        return {'TBG_{}'.format(str(int(t_half_life))):
                self.calculate_time_biased_gain(gains, time_at_k, t_half_life)
                for t_half_life in t_half_lives}
