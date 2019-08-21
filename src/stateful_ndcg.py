from allennlp.training.metrics import Metric
from typing import Dict, Optional, List, Union
import torch
import pytrec_eval

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
