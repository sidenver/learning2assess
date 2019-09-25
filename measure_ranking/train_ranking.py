"""Usage: train_ranking.py [options]

Options:
-l, --loss=LOSS      loss function to use [default: pairwise_logistic_loss]
-s, --split=SPLIT    the data split, 0-4  [default: 0]
-o, --output=DIR     output dir [default: ./]
"""


import tensorflow as tf
import tensorflow_ranking as tfr
import json
from docopt import docopt
import os


score_mapping = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3
}


def convert_to_libsvm_format(path):
    out_path = path + '.svm'
    with open(out_path, 'w') as outputs:
        with open(path, 'r') as inputs:
            for line in inputs:
                jsonl = json.loads(line)
                user_embedding = jsonl['user_embedding']
                true_label = score_mapping[jsonl['true_label']]
                outputs.write('{} {} {}\n'.format(true_label,
                                                  'qid:1',
                                                  ' '.join(['{}:{}'.format(idx + 1, feature)
                                                            for idx, feature in enumerate(user_embedding)])))
    return out_path


# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.


# class RankingLossKey(object):
#   """Ranking loss key strings."""
#   # Names for the ranking based loss functions.
#   PAIRWISE_HINGE_LOSS = 'pairwise_hinge_loss'
#   PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'
#   PAIRWISE_SOFT_ZERO_ONE_LOSS = 'pairwise_soft_zero_one_loss'
#   SOFTMAX_LOSS = 'softmax_loss'
#   SIGMOID_CROSS_ENTROPY_LOSS = 'sigmoid_cross_entropy_loss'
#   MEAN_SQUARED_LOSS = 'mean_squared_loss'
#   LIST_MLE_LOSS = 'list_mle_loss'
#   APPROX_NDCG_LOSS = 'approx_ndcg_loss'
#   APPROX_MRR_LOSS = 'approx_mrr_loss'

# In the TF-Ranking framework, a training instance is represented
# by a Tensor that contains features from a list of documents
# associated with a single query. For simplicity, we fix the shape
# of these Tensors to a maximum list size and call it "list_size,"
# the maximum number of documents per query in the dataset.
# In this demo, we take the following approach:
#   * If a query has fewer documents, its Tensor will be padded
#     appropriately.
#   * If a query has more documents, we shuffle its list of
#     documents and trim the list down to the prescribed list_size.
_LIST_SIZE = 1500

# The total number of features per query-document pair.
# We set this number to the number of features in the MSLR-Web30K
# dataset.
_NUM_FEATURES = 100

# Parameters to the scoring function.
_BATCH_SIZE = 32
_HIDDEN_LAYER_DIMS = ["50", "50"]


def input_fn(path):
    train_dataset = tf.data.Dataset.from_generator(
        tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
        output_types=(
            {str(k): tf.float32 for k in range(1, _NUM_FEATURES + 1)},
            tf.float32
        ),
        output_shapes=(
            {str(k): tf.TensorShape([_LIST_SIZE, 1])
             for k in range(1, _NUM_FEATURES + 1)},
            tf.TensorShape([_LIST_SIZE])
        )
    )

    train_dataset = train_dataset.shuffle(1000).repeat().batch(_BATCH_SIZE)
    return train_dataset.make_one_shot_iterator().get_next()


def example_feature_columns():
    """Returns the example feature columns."""
    feature_names = [
        "%d" % (i + 1) for i in range(0, _NUM_FEATURES)
    ]
    return {
        name: tf.feature_column.numeric_column(
            name, shape=(1,), default_value=0.0) for name in feature_names
    }


def make_score_fn():
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a documents."""
        del params
        del config
        # Define input layer.
        example_input = [
            tf.layers.flatten(group_features[name])
            for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
            cur_layer = tf.layers.dense(
                cur_layer,
                units=layer_width,
                activation="tanh")

        logits = tf.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn


def eval_metric_fns():
    """Returns a dict from name to metric functions.

    This can be customized as follows. Care must be taken when handling padded
    lists.

    def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
    metric_fns["auc"] = _auc

    Returns:
    A dict mapping from metric name to a metric function with above signature.
    """
    metric_fns = {}
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10, 15, 30, 50, 100]
    })

    return metric_fns


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    train_path_jsonl = "/fs/clip-psych/shing/task_B_user_embedding/task_B.train.{}.train.prediction".format(args['--split'])
    dev_path_jsonl = "/fs/clip-psych/shing/task_B_user_embedding/task_B.train.{}.test.prediction".format(args['--split'])
    test_path_jsonl = "/fs/clip-psych/shing/task_B_user_embedding/task_B.test.{}.prediction".format(args['--split'])

    # Store the paths to files containing training and test instances.
    # As noted above, we will assume the data is in the LibSVM format
    # and that the content of each file is sorted by query ID.
    _TRAIN_DATA_PATH = convert_to_libsvm_format(train_path_jsonl)
    _TEST_DATA_PATH = convert_to_libsvm_format(dev_path_jsonl)

    _LOSS = args['--loss']

    PAIRWISE_HINGE_LOSS = 'pairwise_hinge_loss'
    PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'
    PAIRWISE_SOFT_ZERO_ONE_LOSS = 'pairwise_soft_zero_one_loss'
    SOFTMAX_LOSS = 'softmax_loss'
    SIGMOID_CROSS_ENTROPY_LOSS = 'sigmoid_cross_entropy_loss'
    MEAN_SQUARED_LOSS = 'mean_squared_loss'
    LIST_MLE_LOSS = 'list_mle_loss'
    APPROX_NDCG_LOSS = 'approx_ndcg_loss'
    APPROX_MRR_LOSS = 'approx_mrr_loss'

    all_losses = [PAIRWISE_HINGE_LOSS, PAIRWISE_LOGISTIC_LOSS,
                  PAIRWISE_SOFT_ZERO_ONE_LOSS, SOFTMAX_LOSS,
                  SIGMOID_CROSS_ENTROPY_LOSS, MEAN_SQUARED_LOSS,
                  LIST_MLE_LOSS, APPROX_NDCG_LOSS, APPROX_MRR_LOSS]

    assert(_LOSS in all_losses)

    def get_estimator(hparams):
        """Create a ranking estimator.

        Args:
        hparams: (tf.contrib.training.HParams) a hyperparameters object.

        Returns:
        tf.learn `Estimator`.
        """
        def _train_op_fn(loss):
            """Defines train op used in ranking head."""
            return tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=hparams.learning_rate,
                optimizer="Adagrad")

        ranking_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(_LOSS),
            eval_metric_fns=eval_metric_fns(),
            train_op_fn=_train_op_fn)

        return tf.estimator.Estimator(
            model_fn=tfr.model.make_groupwise_ranking_fn(
                group_score_fn=make_score_fn(),
                group_size=1,
                transform_fn=None,
                ranking_head=ranking_head),
            params=hparams)

    hparams = tf.contrib.training.HParams(learning_rate=0.05)
    ranker = get_estimator(hparams)

    ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH), steps=100)
    results = ranker.evaluate(input_fn=lambda: input_fn(_TEST_DATA_PATH), steps=100)

    with open(os.path.join(args['--output'], '{}_{}.json'.format(_LOSS, args['--split'])), 'w') as outfile:
        json.dump({key: float(results[key]) for key in results}, outfile)
