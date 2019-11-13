from typing import Iterator, List, Dict, Optional, Any

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.dataset_readers import DatasetReader

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ListField, MetadataField

import json
import numpy as np


@DatasetReader.register('user_reader')
class UserDatasetReader(DatasetReader):
    """
    For pre-sentenized, pre-tokenized json-line SuicideWatch Dataset
    """

    def __init__(self, token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def tokens_to_user_field(self, tokens) -> ListField:
        doc_list = []
        for doc in tokens:
            sent_list = []
            for sentence in doc:
                word_list = []
                for word in sentence:
                    word_list.append(Token(word))
                sent_list.append(TextField(word_list, self.token_indexers))
            doc_list.append(ListField(sent_list))
        return ListField(doc_list)

    def text_to_instance(self, user_id: int, tokens: List[List[List[str]]],
                         subreddit: List[str], timestamp: List[int],
                         label: Optional[str] = None) -> Instance:
        user_field = self.tokens_to_user_field(tokens)
        fields = {"tokens": user_field}

        if label:
            label_field = LabelField(label)
            fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                user = json.loads(line.strip())
                # user_id, label, tokens, subreddit, timestamp = user

                yield self.text_to_instance(user["user_id"],
                                            user['tokens'], user["subreddit"],
                                            user["timestamp"], user["label"])


@DatasetReader.register('user_detailed_reader')
class UserDetailedDatasetReader(DatasetReader):
    """
    For pre-sentenized, pre-tokenized json-line SuicideWatch Dataset
    """

    def __init__(self, token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def tokens_to_user_field(self, tokens) -> ListField:
        doc_list = []
        for doc in tokens:
            sent_list = []
            for sentence in doc:
                word_list = []
                for word in sentence:
                    word_list.append(Token(word))
                sent_list.append(TextField(word_list, self.token_indexers))
            doc_list.append(ListField(sent_list))
        return ListField(doc_list)

    def text_to_instance(self, user_id: int, tokens: List[List[List[str]]],
                         subreddit: List[str], timestamp: List[int],
                         label: Optional[str] = None,
                         raw_label: Optional[str] = None) -> Instance:
        user_field = self.tokens_to_user_field(tokens)
        fields = {"tokens": user_field}

        if label:
            label_field = LabelField(label)
            fields["label"] = label_field

        raw_meta_field = MetadataField(raw_label)
        fields["raw_label"] = raw_meta_field

        fields["meta"] = MetadataField({"user_id": user_id,
                                        "subreddit": subreddit,
                                        "timestamp": timestamp})

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                user = json.loads(line.strip())

                yield self.text_to_instance(user["user_id"],
                                            user['tokens'], user["subreddit"],
                                            user["timestamp"], user["label"],
                                            user["raw_label"])


@DatasetReader.register('user_clpsych_reader')
class UserCLPsychDatasetReader(DatasetReader):
    """
    For pre-sentenized, pre-tokenized json-line SuicideWatch Dataset
    """

    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 max_doc: int = 50,
                 max_sent: int = 16,
                 max_word: int = 64,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.max_doc = max_doc
        self.max_sent = max_sent
        self.max_word = max_word
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def tokens_to_user_field(self, tokens) -> ListField:
        doc_list = []
        for doc in tokens[-self.max_doc:]:
            sent_list = []
            for sentence in doc[:self.max_sent]:
                word_list = []
                for word in sentence[:self.max_word]:
                    word_list.append(Token(word))
                sent_list.append(TextField(word_list, self.token_indexers))
            doc_list.append(ListField(sent_list))
        return ListField(doc_list)

    def text_to_instance(self, user_id: int, tokens: List[List[List[str]]],
                         subreddit: List[str], timestamp: List[int],
                         label: Optional[str] = None) -> Instance:
        user_field = self.tokens_to_user_field(tokens)
        fields = {"tokens": user_field}

        if label:
            label_field = LabelField(label)
            fields["label"] = label_field

        fields["meta"] = MetadataField({"user_id": user_id,
                                        "subreddit": subreddit,
                                        "timestamp": timestamp})

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                user = json.loads(line.strip())

                yield self.text_to_instance(user["user_id"],
                                            user['tokens'], user["subreddit"],
                                            user["timestamp"], user["label"])


@DatasetReader.register('user_clpsych_time_reader')
class UserCLPsychTimeDatasetReader(UserCLPsychDatasetReader):
    """
    For pre-sentenized, pre-tokenized json-line SuicideWatch Dataset
    """

    def tokens_to_user_field(self, tokens) -> ListField:
        doc_list = []
        for doc in tokens[-self.max_doc:]:
            sent_list = []
            for sentence in doc[:self.max_sent]:
                word_list = []
                for word in sentence[:self.max_word]:
                    word_list.append(Token(word))
                sent_list.append(TextField(word_list, self.token_indexers))
            doc_list.append(ListField(sent_list))
        return ListField(doc_list)

    def word_count(self, tokens: List[List[List[str]]]):
        totol_word_count = 0
        for doc in tokens[-self.max_doc:]:
            for sentence in doc:
                totol_word_count += len(sentence)

        return totol_word_count

    def text_to_instance(self, user_id: int, tokens: List[List[List[str]]],
                         subreddit: List[str], timestamp: List[int],
                         label: Optional[str] = None) -> Instance:
        user_field = self.tokens_to_user_field(tokens)
        fields = {"tokens": user_field}
        tokens_word_count = self.word_count(tokens)
        fields["word_count"] = MetadataField(tokens_word_count)

        if label:
            label_field = LabelField(label)
            fields["label"] = label_field
            raw_meta_field = MetadataField(label)
            fields["raw_label"] = raw_meta_field

        fields["meta"] = MetadataField({"user_id": user_id,
                                        "subreddit": subreddit[-self.max_doc:],
                                        "timestamp": timestamp[-self.max_doc:]})

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                user = json.loads(line.strip())

                yield self.text_to_instance(user["user_id"],
                                            user['tokens'], user["subreddit"],
                                            user["timestamp"], user["label"])


@DatasetReader.register('user_clpsych_post_time_reader')
class UserCLPsychPostTimeDatasetReader(UserCLPsychDatasetReader):
    """
    For pre-sentenized, pre-tokenized json-line SuicideWatch Dataset
    """

    def tokens_to_user_field(self, tokens) -> ListField:
        doc_list = []
        for doc in tokens[-self.max_doc:]:
            sent_list = []
            for sentence in doc[:self.max_sent]:
                word_list = []
                for word in sentence[:self.max_word]:
                    word_list.append(Token(word))
                sent_list.append(TextField(word_list, self.token_indexers))
            doc_list.append(ListField(sent_list))
        return ListField(doc_list)

    def doc_word_counts(self, tokens: List[List[List[str]]]):
        doc_word_count_list = []
        for doc in tokens[-self.max_doc:]:
            totol_word_count = 0
            for sentence in doc:
                totol_word_count += len(sentence)
            doc_word_count_list.append(totol_word_count)

        return doc_word_count_list

    def text_to_instance(self, user_id: int, tokens: List[List[List[str]]],
                         subreddit: List[str], post_id: List[str],
                         support: List[List[Any]], timestamp: List[int],
                         label: Optional[str] = None) -> Instance:
        user_field = self.tokens_to_user_field(tokens)
        fields = {"tokens": user_field}
        doc_word_count_list = self.doc_word_counts(tokens)
        fields["doc_word_counts"] = MetadataField(doc_word_count_list)
        fields["support"] = MetadataField(support[-self.max_doc:])

        if label:
            label_field = LabelField(label)
            fields["label"] = label_field
            raw_meta_field = MetadataField(label)
            fields["raw_label"] = raw_meta_field

        fields["meta"] = MetadataField({"user_id": user_id,
                                        "subreddit": subreddit[-self.max_doc:],
                                        "post_id": post_id[-self.max_doc:],
                                        "timestamp": timestamp[-self.max_doc:]})

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                user = json.loads(line.strip())

                yield self.text_to_instance(user["user_id"],
                                            user['tokens'], user["subreddit"],
                                            user["post_id"], user["support"],
                                            user["timestamp"], user["label"])


@DatasetReader.register('user_bert_reader')
class UserBertDatasetReader(DatasetReader):
    """
    For pre-sentenized, pre-tokenized json-line SuicideWatch Dataset
    """

    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 max_doc: int = 50,
                 overflow_doc_strategy: str = 'random',
                 max_sent: int = 16,
                 max_word: int = 64,
                 max_word_len: int = 30,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.max_doc = max_doc
        self.max_sent = max_sent
        self.max_word = max_word
        self.max_word_len = max_word_len
        self.overflow_doc_strategy = overflow_doc_strategy
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def tokens_to_user_field(self, tokens) -> Optional[ListField]:
        doc_list = []
        if self.overflow_doc_strategy == 'latest':
            docs = tokens[-self.max_doc:]
        elif self.overflow_doc_strategy == 'earliest':
            docs = tokens[:self.max_doc]
        elif self.overflow_doc_strategy == 'all':
            docs = tokens
        elif self.overflow_doc_strategy == 'random':
            if len(tokens) > self.max_doc:
                doc_indexes = range(len(tokens))
                selected_doc_indexes = np.sort(np.random.choice(doc_indexes,
                                                                self.max_doc,
                                                                replace=False))
                docs = [tokens[i] for i in selected_doc_indexes]
            else:
                docs = tokens
        else:
            raise ValueError('{} as docs overflow strategy is not valid, \
choose from latest, earliest, or random'.format(self.overflow_doc_strategy))

        for doc in docs:
            sent_list = []
            for sentence in doc[:self.max_sent]:
                word_list = []
                for word in sentence[:self.max_word]:
                    if len(word) < self.max_word_len:
                        word_list.append(Token(word))
                    else:
                        word_list.append(Token(word[:self.max_word_len]))
                if len(word_list) > 0:
                    sent_list.append(TextField(word_list, self.token_indexers))

            if len(sent_list) > 0:
                doc_list.append(ListField(sent_list))

        if len(doc_list) > 0:
            return ListField(doc_list)
        else:
            return None

    def text_to_instance(self, user_id: int, tokens: List[List[List[str]]],
                         subreddit: List[str], timestamp: List[int],
                         label: Optional[str] = None) -> Optional[Instance]:
        user_field = self.tokens_to_user_field(tokens)
        if user_field is not None:
            fields = {"tokens": user_field}

            if label:
                label_field = LabelField(label)
                fields["label"] = label_field

            fields["meta"] = MetadataField({"user_id": user_id,
                                            "subreddit": subreddit,
                                            "timestamp": timestamp})

            return Instance(fields)
        else:
            return None

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                user = json.loads(line.strip())
                instance = self.text_to_instance(user["user_id"],
                                                 user['tokens'], user["subreddit"],
                                                 user["timestamp"], user["label"])
                if instance is not None:
                    yield instance
                else:
                    print('ill formed user:')
                    print(user)
                    continue
