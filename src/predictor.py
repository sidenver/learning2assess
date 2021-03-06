from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('han_user_attenion_predictor')
class HanUserAttentionPredictor(Predictor):
    """Predictor wrapper for the han_clpsych with user embedding"""
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)

        output_dict['true_label'] = inputs['label'] if 'label' in inputs else None
        output_dict['user_id'] = inputs['user_id']
        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        output_dict["tokens"] = inputs["tokens"]
        return output_dict

    """using user_clpsych_post_time_reader"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user_id = json_dict['user_id']
        tokens = json_dict['tokens']
        subreddit = json_dict['subreddit']
        timestamp = json_dict['timestamp']
        post_id = json_dict['post_id']
        support = json_dict['support']
        label = json_dict['label'] if 'label' in json_dict else None
        return self._dataset_reader.text_to_instance(user_id=user_id,
                                                     tokens=tokens, subreddit=subreddit,
                                                     post_id=post_id, support=support,
                                                     timestamp=timestamp, label=label)


@Predictor.register('han_clpsych_predictor')
class HanClpsychPredictor(Predictor):
    """Predictor wrapper for the han_clpsych with user embedding"""
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)

        output_dict['true_label'] = inputs['label'] if 'label' in inputs else None
        output_dict['user_id'] = inputs['user_id']
        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        return output_dict

    """using user_clpsych_reader"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user_id = json_dict['user_id']
        tokens = json_dict['tokens']
        subreddit = json_dict['subreddit']
        timestamp = json_dict['timestamp']
        label = json_dict['label'] if 'label' in json_dict else None
        return self._dataset_reader.text_to_instance(user_id=user_id,
                                                     tokens=tokens, subreddit=subreddit,
                                                     timestamp=timestamp, label=label)


@Predictor.register('han_user_htbg_predictor')
class HanUserHTBGPredictor(Predictor):
    """Predictor wrapper for the han_clpsych with user embedding"""
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)

        output_dict['true_label'] = inputs['label'] if 'label' in inputs else None
        output_dict['user_id'] = inputs['user_id']
        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        return output_dict

    """using user_clpsych_post_time_reader"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user_id = json_dict['user_id']
        tokens = json_dict['tokens']
        subreddit = json_dict['subreddit']
        timestamp = json_dict['timestamp']
        post_id = json_dict['post_id']
        support = json_dict['support']
        label = json_dict['label'] if 'label' in json_dict else None
        return self._dataset_reader.text_to_instance(user_id=user_id,
                                                     tokens=tokens, subreddit=subreddit,
                                                     post_id=post_id, support=support,
                                                     timestamp=timestamp, label=label)


@Predictor.register('han_post_htbg_predictor')
class HanPostHTBGPredictor(Predictor):
    """Predictor wrapper for the han_clpsych with user embedding"""
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        output_list = []
        for post_instance in self._json_to_instance(inputs):
            output_dict = self.predict_instance(post_instance)

            output_dict['true_label'] = inputs['label'] if 'label' in inputs else None
            output_dict['user_id'] = inputs['user_id']
            # label_dict will be like {0: "ACL", 1: "AI", ...}
            label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
            # Convert it to list ["ACL", "AI", ...]
            all_labels = [label_dict[i] for i in range(len(label_dict))]
            output_dict["all_labels"] = all_labels
            output_list.append(output_dict)
        return output_list

    """using user_clpsych_reader"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user_id = json_dict['user_id']
        tokens = json_dict['tokens']
        subreddit = json_dict['subreddit']
        timestamp = json_dict['timestamp']
        post_id = json_dict['post_id']
        support = json_dict['support']
        label = json_dict['label'] if 'label' in json_dict else None
        return self._dataset_reader.text_to_instance(user_id=user_id,
                                                     tokens=tokens, subreddit=subreddit,
                                                     post_id=post_id, support=support,
                                                     timestamp=timestamp, label=label)


@Predictor.register('han_clpsych_pretrain_predictor')
class HanClpsychPretrainPredictor(HanClpsychPredictor):
    """Predictor wrapper for the han_clpsych with user embedding"""

    """using user_clpsych_reader"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user_id = json_dict['user_id']
        tokens = json_dict['tokens']
        subreddit = json_dict['subreddit']
        timestamp = json_dict['timestamp']
        # label = json_dict['label'] if 'label' in json_dict else None
        return self._dataset_reader.text_to_instance(user_id=user_id,
                                                     tokens=tokens, subreddit=subreddit,
                                                     timestamp=timestamp, label=None)
