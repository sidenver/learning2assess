from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from typing import Iterator, List, Dict, Optional, Any
from allennlp.training.metrics import CategoricalAccuracy, Metric
from allennlp.nn.util import get_text_field_mask

import torch
import torch.nn as nn


@Model.register('lr_glove_bow_empath_readability')
class LRGloveBowEmpathReadability(Model):
    '''
    Contains 3 layers Hierachical Attention RNNs
    '''

    def __init__(self,
                 vocab: Vocabulary,
                 word_to_sentence: Seq2VecEncoder,
                 sentence_to_doc: Seq2VecEncoder,
                 doc_to_user: Seq2VecEncoder,
                 word_embeddings: TextFieldEmbedder,
                 bow_embeddings: TextFieldEmbedder,
                 empath_embeddings: Optional[TextFieldEmbedder] = None,
                 readability_embeddings: Optional[TextFieldEmbedder] = None
                 ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self._word_to_sentence = word_to_sentence
        self._sentence_to_doc = sentence_to_doc
        self._doc_to_user = doc_to_user
        self._word_embeddings = word_embeddings
        self._bow_embeddings = bow_embeddings
        self._empath_embeddings = empath_embeddings
        self._readability_embeddings = readability_embeddings
        self._classifier_input_dim = self._word_embeddings.get_output_dim() \
            + self._bow_embeddings.get_output_dim() \
            + (self._empath_embeddings.get_output_dim() if self._empath_embeddings else 0) \
            + (self._readability_embeddings.get_output_dim() if self._readability_embeddings else 0)
        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._predictor = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        def reshape_for_seq2vec(vec, mask):
            reshaped_vec = vec.view(-1, mask.shape[-1], vec.shape[-1])
            reshaped_mask = mask.view(-1, mask.shape[-1])
            return reshaped_vec, reshaped_mask

        word_mask = get_text_field_mask(tokens, num_wrapping_dims=2)
        sentence_mask = (word_mask.sum(dim=-1) > 0).long()
        doc_mask = (sentence_mask.sum(dim=-1) > 0).long()

        # print(tokens.keys())
        # print(tokens['tokens'].shape)
        _word_embedded = self._word_embeddings(tokens, num_wrapping_dims=2)
        print(_word_embedded.shape)
        embedded = _word_embedded
        # _empath_embedded = self._empath_embeddings(tokens, num_wrapping_dims=2)
        # _readability_embedded = self._readability_embeddings(tokens, num_wrapping_dims=2)

        embedded_at_word, word_mask_at_word = reshape_for_seq2vec(embedded, word_mask)
        # print(embedded.shape)
        print(embedded_at_word.shape)
        # print(word_mask_at_word.shape)
        _bow_embedded = self._bow_embeddings(tokens, num_wrapping_dims=2)
        print(_bow_embedded.shape)

        sentences = self._word_to_sentence(embedded_at_word, word_mask_at_word)
        sentences_at_sentence, sentence_mask_at_sentence = reshape_for_seq2vec(sentences, sentence_mask)
        print(sentences.shape, sentences_at_sentence.shape, sentence_mask_at_sentence.shape)

        docs = self._sentence_to_doc(sentences_at_sentence, sentence_mask_at_sentence)
        docs_at_doc, doc_mask_at_doc = reshape_for_seq2vec(docs, doc_mask)
        # print(docs.shape, docs_at_doc.shape, doc_mask_at_doc.shape)

        users = self._doc_to_user(docs_at_doc, doc_mask_at_doc)
        # print(users.shape)

        prediction = self._predictor(users)

        output = {}
        output['loss'] = self._loss(prediction, label)
        # output['accuracy'] =

        self._accuracy(prediction, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}
