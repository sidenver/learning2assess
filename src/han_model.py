from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import Attention
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Metric
from allennlp.training.metrics.fbeta_measure import FBetaMeasure

from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel

from typing import Optional, Dict, List, Any, Union

import torch
import torch.nn as nn

# from torch.autograd import Variable


@Seq2VecEncoder.register('attention_rnn')
class AttentionRNN(Seq2VecEncoder):
    """
    A base-to-higher-level module.
    """

    def __init__(self, encoder: Seq2SeqEncoder, attention: Attention) -> None:
        super(AttentionRNN, self).__init__()

        self.encoder = encoder
        self.attention = attention
        self.context = nn.Parameter(torch.Tensor(encoder.get_output_dim(), 1).uniform_(-0.1, 0.1).view(-1))

    def forward(self, matrix: torch.Tensor, matrix_mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs: pack_padded_sequence of (batch, max_length, input_size)
        Outpus: (batch, hidden_size*2)
        """

        # run through the bidirectional GRU
        encoded = self.encoder(matrix, matrix_mask)
        batch_size = encoded.shape[0]
        broadcast_context = self.context.repeat(1, batch_size).view(batch_size, -1)
        attention_weights = self.attention(broadcast_context, encoded, matrix_mask)

        higher_tensor = torch.bmm(attention_weights.unsqueeze(-2), encoded).squeeze(-2)

        return higher_tensor, attention_weights

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a ``Seq2VecEncoder``. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        return self.encoder.get_input_dim()

    def get_output_dim(self) -> int:
        """
        Returns the dimension of the final vector output by this ``Seq2VecEncoder``.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        return self.encoder.get_output_dim()


@Model.register('3HAN')
class HierarchicalAttentionRNN3(Model):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self._embeddings = word_embeddings
        self._word_to_sentence = word_to_sentence
        self._sentence_to_doc = sentence_to_doc
        self._doc_to_user = doc_to_user
        self._classifier_input_dim = self._doc_to_user.get_output_dim()
        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._predictor = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        output['loss'] = self._loss(prediction, label)
        # output['accuracy'] =

        self._accuracy(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}


@Model.register('3HAN_ndcg')
class HierarchicalAttentionRNN3NDCG(HierarchicalAttentionRNN3):
    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder,
                 ndcg_metric: Metric,
                 normalized_ndcg_metric: Metric
                 ) -> None:
        super().__init__(vocab,
                         word_embeddings,
                         word_to_sentence,
                         sentence_to_doc,
                         doc_to_user)

        self._ndcg = ndcg_metric
        self._normalized_ndcg = normalized_ndcg_metric

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                raw_label: Optional[List[str]] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        output['loss'] = self._loss(prediction, label)
        # output['accuracy'] =

        self._accuracy(prediction, label)
        if raw_label is not None:
            normalized = nn.Softmax(dim=-1)
            positive_index = self.vocab.get_token_index('positive', namespace='labels')
            self._ndcg(prediction[:, positive_index], raw_label)
            self._normalized_ndcg(normalized(prediction)[:, positive_index], raw_label)
            # self._rev_ndcg(prediction[:, control_index], raw_label)
            # self._rev_normalized_ndcg(normalized(prediction)[:, control_index], raw_label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset),
                **self._ndcg.get_metric(reset),
                **{'n_' + key: val
                   for key, val in self._normalized_ndcg.get_metric(reset).items()}}


@Model.register('3HAN_clpsych')
class HierarchicalAttentionRNN3CLPsych(Model):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self._embeddings = word_embeddings
        self._word_to_sentence = word_to_sentence
        self._sentence_to_doc = sentence_to_doc
        self._doc_to_user = doc_to_user
        self._classifier_input_dim = self._doc_to_user.get_output_dim()
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        print('num_label:', self._num_labels)

        self._predictor = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()
        self.label_index_to_token = self.vocab.get_index_to_token_vocabulary(namespace="labels")
        print(self.label_index_to_token)
        index_list = list(range(self._num_labels))
        print(index_list)
        self._f1 = FBetaMeasure(average=None, labels=index_list)
        self._f1_micro = FBetaMeasure(average='micro')
        self._f1_macro = FBetaMeasure(average='macro')

        print(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        output['loss'] = self._loss(prediction, label)
        # output['user_embedding'] = users
        output['prediction'] = prediction

        if label is not None:
            output['truth'] = label
            # output['accuracy'] =

            self._accuracy(prediction, label)
            self._f1(prediction, label)
            self._f1_micro(prediction, label)
            self._f1_macro(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_scores = {'{}_{}'.format(self.label_index_to_token[index], key): f
                     for key, val in self._f1.get_metric(reset).items()
                     for index, f in enumerate(val)}
        f1_micro = {'micro_' + key: val
                    for key, val in self._f1_micro.get_metric(reset).items()}
        f1_macro = {'macro_' + key: val
                    for key, val in self._f1_macro.get_metric(reset).items()}

        return {"accuracy": self._accuracy.get_metric(reset),
                **f1_scores,
                **f1_micro,
                **f1_macro}


@Model.register('2HAN_clpsych')
class HierarchicalAttentionRNN2CLPsych(Model):
    '''
    Contains 2 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self._embeddings = word_embeddings
        self._word_to_sentence = word_to_sentence
        self._sentence_to_doc = sentence_to_doc
        self._classifier_input_dim = self._sentence_to_doc.get_output_dim()
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        print('num_label:', self._num_labels)

        self._predictor = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()
        self.label_index_to_token = self.vocab.get_index_to_token_vocabulary(namespace="labels")
        print(self.label_index_to_token)
        index_list = list(range(self._num_labels))
        print(index_list)
        self._f1 = FBetaMeasure(average=None, labels=index_list)
        self._f1_micro = FBetaMeasure(average='micro')
        self._f1_macro = FBetaMeasure(average='macro')

        print(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                doc_word_counts: Optional[List[int]] = None,
                support: Optional[List[List[Any]]] = None,
                meta: Optional[List[Dict[str, Any]]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        # doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=1)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        # docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        # users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(docs)

        output = {}
        output['doc_embedding'] = docs
        output['prediction'] = prediction
        output['doc_word_counts'] = doc_word_counts
        output['support'] = support
        output['meta'] = meta

        if label is not None:
            output['truth'] = label
            output['loss'] = self._loss(prediction, label)
            # output['accuracy'] =

            self._accuracy(prediction, label)
            self._f1(prediction, label)
            self._f1_micro(prediction, label)
            self._f1_macro(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_scores = {'{}_{}'.format(self.label_index_to_token[index], key): f
                     for key, val in self._f1.get_metric(reset).items()
                     for index, f in enumerate(val)}
        f1_micro = {'micro_' + key: val
                    for key, val in self._f1_micro.get_metric(reset).items()}
        f1_macro = {'macro_' + key: val
                    for key, val in self._f1_macro.get_metric(reset).items()}

        return {"accuracy": self._accuracy.get_metric(reset),
                **f1_scores,
                **f1_micro,
                **f1_macro}


@Model.register('3HAN_clpsych_time_ndcg')
class HierarchicalAttentionRNN3CLPsychTimed(HierarchicalAttentionRNN3CLPsych):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder,
                 ndcg_metric: Metric,
                 time_metric: Metric,
                 ) -> None:
        super().__init__(vocab,
                         word_embeddings,
                         word_to_sentence,
                         sentence_to_doc,
                         doc_to_user)
        self._ndcg = ndcg_metric
        self._time = time_metric

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                raw_label: Optional[List[str]] = None,
                word_count: Optional[List[int]] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        output['loss'] = self._loss(prediction, label)
        output['user_embedding'] = users
        output['prediction'] = prediction

        if label is not None:
            output['truth'] = label
            # output['accuracy'] =
            a_index = self.vocab.get_token_index('a', namespace='labels')
            b_index = self.vocab.get_token_index('b', namespace='labels')
            c_index = self.vocab.get_token_index('c', namespace='labels')
            d_index = self.vocab.get_token_index('d', namespace='labels')

            indexes_with_score = [(a_index, 0), (b_index, 1), (c_index, 2), (d_index, 3)]

            normalized = nn.Softmax(dim=-1)

            prediction_probability = normalized(prediction)
            scores_for_ranking = sum([score * prediction_probability[:, index]
                                      for index, score in indexes_with_score])

            self._ndcg(scores_for_ranking, raw_label)
            self._time(scores_for_ranking, raw_label, word_count)

            self._accuracy(prediction, label)
            self._f1(prediction, label)
            self._f1_micro(prediction, label)
            self._f1_macro(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_scores = {'{}_{}'.format(self.label_index_to_token[index], key): f
                     for key, val in self._f1.get_metric(reset).items()
                     for index, f in enumerate(val)}
        f1_micro = {'micro_' + key: val
                    for key, val in self._f1_micro.get_metric(reset).items()}
        f1_macro = {'macro_' + key: val
                    for key, val in self._f1_macro.get_metric(reset).items()}

        return {"accuracy": self._accuracy.get_metric(reset),
                **f1_scores,
                **f1_micro,
                **f1_macro,
                **self._ndcg.get_metric(reset),
                **self._time.get_metric(reset)}


@Model.register('3HAN_clpsych_htbg_time_ndcg')
class HierarchicalAttentionRNN3CLPsychHierarchicalTimed(HierarchicalAttentionRNN3CLPsych):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder,
                 ndcg_metric: Metric,
                 tbg_metric: Metric,
                 htbg_metrics: Dict[str, Metric],
                 ) -> None:
        super().__init__(vocab,
                         word_embeddings,
                         word_to_sentence,
                         sentence_to_doc,
                         doc_to_user)
        self._ndcg = ndcg_metric
        self._tbg_metric = tbg_metric
        self._htbg_metrics = htbg_metrics

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                raw_label: Optional[List[str]] = None,
                doc_word_counts: Optional[List[List[int]]] = None,
                support: Optional[List[List[List[Any]]]] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users, document_attentions = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        # output['user_embedding'] = users
        output['document_attentions'] = document_attentions
        # output['word_attentions'] = word_attentions
        # output['sentence_attentions'] = sentence_attentions
        output['support'] = support
        output['doc_word_counts'] = doc_word_counts
        output['meta'] = meta
        output['prediction'] = prediction

        if label is not None:
            output['truth'] = label
            output['loss'] = self._loss(prediction, label)
            # output['accuracy'] =
            a_index = self.vocab.get_token_index('a', namespace='labels')
            b_index = self.vocab.get_token_index('b', namespace='labels')
            c_index = self.vocab.get_token_index('c', namespace='labels')
            d_index = self.vocab.get_token_index('d', namespace='labels')

            indexes_with_score = [(a_index, 0), (b_index, 2), (c_index, 4), (d_index, 8)]

            normalized = nn.Softmax(dim=-1)

            prediction_probability = normalized(prediction)
            scores_for_ranking = sum([score * prediction_probability[:, index]
                                      for index, score in indexes_with_score])

            self._ndcg(scores_for_ranking, raw_label)
            self._tbg_metric(scores_for_ranking, raw_label, [sum(word_counts) for word_counts in doc_word_counts])
            for htbg_metric in self._htbg_metrics:
                self._htbg_metrics[htbg_metric](scores_for_ranking, raw_label,
                                                doc_word_counts, support,
                                                document_attentions,
                                                meta)

            self._accuracy(prediction, label)
            self._f1(prediction, label)
            self._f1_micro(prediction, label)
            self._f1_macro(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_scores = {'{}_{}'.format(self.label_index_to_token[index], key): f
                     for key, val in self._f1.get_metric(reset).items()
                     for index, f in enumerate(val)}
        f1_micro = {'micro_' + key: val
                    for key, val in self._f1_micro.get_metric(reset).items()}
        f1_macro = {'macro_' + key: val
                    for key, val in self._f1_macro.get_metric(reset).items()}
        htbg_metrics = {htbg_type + '_' + key: val
                        for htbg_type, htbg_metric in self._htbg_metrics.items()
                        for key, val in htbg_metric.get_metric(reset).items()}

        return {"accuracy": self._accuracy.get_metric(reset),
                **f1_scores,
                **f1_micro,
                **f1_macro,
                **self._ndcg.get_metric(reset),
                **self._tbg_metric.get_metric(reset),
                **htbg_metrics}


@Model.register('3HAN_clpsych_attention_out')
class HierarchicalAttentionRNN3CLPsychAttentionOut(HierarchicalAttentionRNN3CLPsych):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder,
                 ndcg_metric: Metric,
                 tbg_metric: Metric,
                 htbg_metrics: Dict[str, Metric],
                 ) -> None:
        super().__init__(vocab,
                         word_embeddings,
                         word_to_sentence,
                         sentence_to_doc,
                         doc_to_user)
        self._ndcg = ndcg_metric
        self._tbg_metric = tbg_metric
        self._htbg_metrics = htbg_metrics

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                raw_label: Optional[List[str]] = None,
                doc_word_counts: Optional[List[List[int]]] = None,
                support: Optional[List[List[List[Any]]]] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        print('embedded', embedded.shape)
        print('embedded_at_word', embedded_at_word.shape)
        print('word_mask_at_word', word_mask_at_word.shape)

        sentences, word_attentions = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        print('word_attentions', word_attentions.shape, type(word_attentions))
        print('sentences', sentences.shape, type(sentences))
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, sentence_attentions = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        print('sentence_attentions', sentence_attentions.shape, type(sentence_attentions))
        print('docs', docs.shape, type(docs))
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users, document_attentions = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        print('document_attentions', document_attentions.shape, type(document_attentions))
        print('users', users.shape, type(users))
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        # output['user_embedding'] = users
        output['document_attentions'] = document_attentions
        output['word_attentions'] = word_attentions.view(embedded.shape[0], embedded.shape[1], embedded.shape[2], -1)
        output['sentence_attentions'] = sentence_attentions.view(embedded.shape[0], embedded.shape[1], -1)
        output['support'] = support
        output['doc_word_counts'] = doc_word_counts
        output['meta'] = meta
        output['prediction'] = prediction

        if label is not None:
            output['truth'] = label
            output['loss'] = self._loss(prediction, label)
            # output['accuracy'] =
            a_index = self.vocab.get_token_index('a', namespace='labels')
            b_index = self.vocab.get_token_index('b', namespace='labels')
            c_index = self.vocab.get_token_index('c', namespace='labels')
            d_index = self.vocab.get_token_index('d', namespace='labels')

            indexes_with_score = [(a_index, 0), (b_index, 2), (c_index, 4), (d_index, 8)]

            normalized = nn.Softmax(dim=-1)

            prediction_probability = normalized(prediction)
            scores_for_ranking = sum([score * prediction_probability[:, index]
                                      for index, score in indexes_with_score])

            self._ndcg(scores_for_ranking, raw_label)
            self._tbg_metric(scores_for_ranking, raw_label, [sum(word_counts) for word_counts in doc_word_counts])
            for htbg_metric in self._htbg_metrics:
                self._htbg_metrics[htbg_metric](scores_for_ranking, raw_label,
                                                doc_word_counts, support,
                                                document_attentions,
                                                meta)

            self._accuracy(prediction, label)
            self._f1(prediction, label)
            self._f1_micro(prediction, label)
            self._f1_macro(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_scores = {'{}_{}'.format(self.label_index_to_token[index], key): f
                     for key, val in self._f1.get_metric(reset).items()
                     for index, f in enumerate(val)}
        f1_micro = {'micro_' + key: val
                    for key, val in self._f1_micro.get_metric(reset).items()}
        f1_macro = {'macro_' + key: val
                    for key, val in self._f1_macro.get_metric(reset).items()}
        htbg_metrics = {htbg_type + '_' + key: val
                        for htbg_type, htbg_metric in self._htbg_metrics.items()
                        for key, val in htbg_metric.get_metric(reset).items()}

        return {"accuracy": self._accuracy.get_metric(reset),
                **f1_scores,
                **f1_micro,
                **f1_macro,
                **self._ndcg.get_metric(reset),
                **self._tbg_metric.get_metric(reset),
                **htbg_metrics}


@Model.register('3HAN_av_clpsych_htbg_time_ndcg')
class HierarchicalAttentionAverageRNN3CLPsychHierarchicalTimed(HierarchicalAttentionRNN3CLPsych):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder,
                 ndcg_metric: Metric,
                 tbg_metric: Metric
                 ) -> None:
        super().__init__(vocab,
                         word_embeddings,
                         word_to_sentence,
                         sentence_to_doc,
                         doc_to_user)
        self._ndcg = ndcg_metric
        self._tbg_metric = tbg_metric

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                raw_label: Optional[List[str]] = None,
                doc_word_counts: Optional[List[List[int]]] = None,
                support: Optional[List[List[List[Any]]]] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        output['user_embedding'] = users
        # output['document_attentions'] = document_attentions
        output['support'] = support
        output['doc_word_counts'] = doc_word_counts
        output['meta'] = meta
        output['prediction'] = prediction

        if label is not None:
            output['truth'] = label
            output['loss'] = self._loss(prediction, label)
            # output['accuracy'] =
            a_index = self.vocab.get_token_index('a', namespace='labels')
            b_index = self.vocab.get_token_index('b', namespace='labels')
            c_index = self.vocab.get_token_index('c', namespace='labels')
            d_index = self.vocab.get_token_index('d', namespace='labels')

            indexes_with_score = [(a_index, 0), (b_index, 2), (c_index, 4), (d_index, 8)]

            normalized = nn.Softmax(dim=-1)

            prediction_probability = normalized(prediction)
            scores_for_ranking = sum([score * prediction_probability[:, index]
                                      for index, score in indexes_with_score])

            self._ndcg(scores_for_ranking, raw_label)
            self._tbg_metric(scores_for_ranking, raw_label, [sum(word_counts) for word_counts in doc_word_counts])
            # for htbg_metric in self._htbg_metrics:
            #     self._htbg_metrics[htbg_metric](scores_for_ranking, raw_label,
            #                                     doc_word_counts, support,
            #                                     document_attentions,
            #                                     meta)

            self._accuracy(prediction, label)
            self._f1(prediction, label)
            self._f1_micro(prediction, label)
            self._f1_macro(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_scores = {'{}_{}'.format(self.label_index_to_token[index], key): f
                     for key, val in self._f1.get_metric(reset).items()
                     for index, f in enumerate(val)}
        f1_micro = {'micro_' + key: val
                    for key, val in self._f1_micro.get_metric(reset).items()}
        f1_macro = {'macro_' + key: val
                    for key, val in self._f1_macro.get_metric(reset).items()}

        return {"accuracy": self._accuracy.get_metric(reset),
                **f1_scores,
                **f1_micro,
                **f1_macro,
                **self._ndcg.get_metric(reset),
                **self._tbg_metric.get_metric(reset)
                }


@Model.register('3HAN_clpsych_pretrain')
class HierarchicalAttentionRNN3CLPsychPre(Model):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self._embeddings = word_embeddings
        self._word_to_sentence = word_to_sentence
        self._sentence_to_doc = sentence_to_doc
        self._doc_to_user = doc_to_user
        self._classifier_input_dim = self._doc_to_user.get_output_dim()
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        print('num_label:', self._num_labels)

        self._predictor_binary = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

        print(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor_binary(users)

        output = {}
        # output['accuracy'] =

        output['user_embedding'] = users
        output['prediction'] = prediction

        if label is not None:
            output['loss'] = self._loss(prediction, label)
            output['truth'] = label

            self._accuracy(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}


@Model.register('3HAN_av_clpsych_pretrain')
class HierarchicalAttentionAverageRNN3CLPsychPre(Model):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self._embeddings = word_embeddings
        self._word_to_sentence = word_to_sentence
        self._sentence_to_doc = sentence_to_doc
        self._doc_to_user = doc_to_user
        self._classifier_input_dim = self._doc_to_user.get_output_dim()
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        print('num_label:', self._num_labels)

        self._predictor_binary = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

        print(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=2)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor_binary(users)

        output = {}
        # output['accuracy'] =

        output['user_embedding'] = users
        output['prediction'] = prediction

        if label is not None:
            output['loss'] = self._loss(prediction, label)
            output['truth'] = label

            self._accuracy(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}


@Model.register('2HAN_clpsych_pretrain')
class HierarchicalAttentionRNN2CLPsychPre(Model):
    '''
    Contains 2 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self._embeddings = word_embeddings
        self._word_to_sentence = word_to_sentence
        self._sentence_to_doc = sentence_to_doc
        self._classifier_input_dim = self._sentence_to_doc.get_output_dim()
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        print('num_label:', self._num_labels)

        self._predictor_binary = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

        print(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                meta: Optional[List[Dict[str, Any]]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask
        # print(tokens.shape)
        # print(tokens)
        word_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        # print(word_mask.shape)
        # print(word_mask)
        # sentence_mask = get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        # print(sentence_mask.shape)
        # print(sentence_mask)
        # doc_mask = get_text_field_mask(tokens, num_wrapping_dims=0)
        # doc_mask = (sentence_mask.sum(dim=-1) > 0).long()
        # print(doc_mask.shape)
        # print(doc_mask)

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        embedded = self._embeddings(tokens, num_wrapping_dims=1)
        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        # print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)

        sentences, _ = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        # print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        # docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        # users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor_binary(docs)

        output = {}
        # output['accuracy'] =

        output['doc_embedding'] = docs
        output['prediction'] = prediction

        if label is not None:
            output['loss'] = self._loss(prediction, label)
            output['truth'] = label

            self._accuracy(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}


@Model.register('3HAN_bert')
class HierarchicalAttentionRNN3Bert(Model):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder,
                 bert_model: Union[str, BertModel],
                 dropout: float = 0.0,
                 num_labels: int = None,
                 index: str = "bert",
                 label_namespace: str = "labels",
                 bert_trainable: bool = False
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        for param in self.bert_model.parameters():
            param.requires_grad = bert_trainable

        self._label_namespace = label_namespace

        if num_labels:
            out_features = num_labels
        else:
            out_features = vocab.get_vocab_size(namespace=self._label_namespace)

        self._dropout = torch.nn.Dropout(p=dropout)

        self._sentence_to_doc = sentence_to_doc
        self._doc_to_user = doc_to_user
        self._classifier_input_dim = self._doc_to_user.get_output_dim()

        self._predictor_binary = nn.Linear(self._classifier_input_dim, out_features)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()
        self._index = index

    def bert_word_to_sentence(self, input_ids, token_type_ids, word_mask, sentence_mask):
        total_sentence_len = len(sentence_mask.view(-1))
        sentence_mask = sentence_mask.view(-1).nonzero().view(-1)

        mask_input_ids = input_ids.view(-1, input_ids.shape[-1])[sentence_mask, :]
        mask_token_type_ids = token_type_ids.view(-1, input_ids.shape[-1])[sentence_mask, :]
        mask_word_mask = word_mask.view(-1, input_ids.shape[-1])[sentence_mask, :]

        _, pooled = self.bert_model(input_ids=mask_input_ids,
                                    token_type_ids=mask_token_type_ids,
                                    attention_mask=mask_word_mask)
        pooled = self._dropout(pooled)

        device = pooled.get_device()
        padded_pool = torch.zeros((total_sentence_len, pooled.shape[-1]))
        if torch.cuda.is_available():
            padded_pool = padded_pool.to(device)
        padded_pool[sentence_mask, :] = pooled

        sentences = padded_pool.view(input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], -1)
        return sentences

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask

        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]

        word_mask = (input_ids != 0).long()
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()

        sentences = self.bert_word_to_sentence(input_ids, token_type_ids, word_mask, sentence_mask)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)

        docs, _ = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)

        users, _ = self._doc_to_user(docs_at_doc, doc_mask_at_doc)

        prediction = self._predictor_binary(users)

        output = {}
        output['loss'] = self._loss(prediction, label)

        self._accuracy(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}
