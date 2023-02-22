import pathlib
import unicodedata
import os

TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_PAD = '[PAD]'
TOKEN_START = '<s>'
TOKEN_END = '</s>'



class Tokenizer(object):
    def __init__(self, token_dict, token_cls=TOKEN_CLS, token_sep=TOKEN_SEP, token_unk=TOKEN_UNK, pad_index=0):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._token_cls = token_cls
        self._token_sep = token_sep
        self._token_unk = token_unk
        self._pad_index = pad_index
        self.vocabsize = len(token_dict)  # 包含pad 不用加一

    def _convert_tokens_to_ids(self, tokens):
        unk_id = self._token_dict.get(self._token_unk)

        # res = []
        # for token in tokens:
        #     if token not in self._token_dict:
        #         print(tokens)
        #     res.append(self._token_dict.get(token, unk_id))
        # return res
        return [self._token_dict.get(token, unk_id) for token in tokens]

    @staticmethod
    def _is_punctuation(ch):
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_space(ch):
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')


class LabelTokenizer(object):
    def __init__(self, label_dict, label_unk=TOKEN_UNK):

        self._label_dict = label_dict
        self._label_dict_inv = {v: k for k, v in label_dict.items()}
        self._token_unk = label_unk
        self.vocabsize = len(label_dict)

    @staticmethod
    def load_from_vocab_txt(vocab_file):
        with open(vocab_file) as f:
            label_dict = dict([(key.strip(), index) for index, key in enumerate(f.readlines())])
        return LabelTokenizer(label_dict)

    @classmethod
    def load_from_txt(cls, txt_file, vocab_file, binary=False):
        if os.path.exists(vocab_file):
            return cls.load_from_vocab_txt(vocab_file)
        with open(txt_file) as f:
            vocab_set = set([word for line in f.readlines() for word in line.split()])
            #vocab_set = set([line.strip() for line in f.readlines()])
        with open(vocab_file, "w") as f:
            if not binary:
                add_token = [TOKEN_UNK]
                vocab_set = vocab_set - {"_UNK"}  # "_UNK" change to [UNK]
                for label in add_token:
                    f.write(f"{label}\n")           
            for label in sorted(vocab_set):
                f.write(f"{label}\n")
        return cls.load_from_vocab_txt(vocab_file)

    def encode(self, text):
        # assert text.strip() in self._label_dict
        unk_id = self._label_dict.get(self._token_unk)
        label_ids = self._label_dict.get(text.strip(), unk_id)
        return label_ids

    def batch_encode(self, batch):
        ids = []
        for i in batch:
            indices = self.encode(i)
            ids.append(indices)
        return ids

    def decode(self, ids):
        labels = [self._label_dict_inv[i] for i in ids]
        return labels


class Seq2SeqTokenizer(Tokenizer):
    def __init__(self, token_dict, token_cls=TOKEN_CLS, token_sep=TOKEN_SEP, token_unk=TOKEN_UNK, pad_index=0):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        super(Seq2SeqTokenizer, self).__init__(token_dict, token_cls=token_cls, token_sep=token_sep, token_unk=token_unk, pad_index=pad_index)

    @staticmethod
    def load_from_vocab_txt(vocab_file):
        with open(vocab_file) as f:
            token_dict = dict([(key.strip(), index) for index, key in enumerate(f.readlines())])
        return Seq2SeqTokenizer(token_dict)

    @classmethod
    def load_from_txt(cls, txt_file, vocab_file):
        if pathlib.Path(vocab_file).exists():
            return cls.load_from_vocab_txt(vocab_file)

        with open(txt_file) as f:
            vocab_set = set([word for line in f.readlines() for word in line.split()])
        with open(vocab_file, "w") as f:
            add_token = [TOKEN_PAD, TOKEN_START, TOKEN_END]
            vocab_set = vocab_set - {TOKEN_START, TOKEN_END}
            for token in add_token:
                f.write(f"{token}\n")
            for token in sorted(vocab_set):
                f.write(f"{token}\n")
        return cls.load_from_vocab_txt(vocab_file)

    @staticmethod
    def tokenize(text):
        return text.strip().split()

    def encode(self, text, max_len=None):
        tokens = self.tokenize(text)
        self._truncate(tokens, max_len)
        token_ids = self._convert_tokens_to_ids(tokens)
        if max_len is not None:
            assert max_len > 2, "max_len 必须大于2"
            pad_len = max_len - len(tokens)
            token_ids += [self._pad_index] * pad_len
        return token_ids

    def batch_encode(self, batch, max_len=None):
        ids = []
        for i in batch:
            indices = self.encode(i, max_len=max_len)
            ids.append(indices)
        return ids

    def decode(self, ids):
        try:
            stop = ids.index(self._pad_index)
        except ValueError:
            stop = len(ids)
        tokens = [self._token_dict_inv[i] for i in ids]
        return tokens[:stop]

    @staticmethod
    def _truncate(tokens, max_len=None):
        if max_len is None:
            return
        else:
            del tokens[max_len:]


class BertBaseTokenizer(Tokenizer):
    def __init__(self, token_dict, token_cls=TOKEN_CLS, token_sep=TOKEN_SEP, token_unk=TOKEN_UNK, pad_index=0):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        super(BertBaseTokenizer, self).__init__(token_dict, token_cls=token_cls, token_sep=token_sep, token_unk=token_unk, pad_index=pad_index)

    def tokenize(self, first, second=None):
        """Split text to tokens.

        :param first: First text.
        :param second: Second text.
        :return: A list of strings.
        """
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        tokens, _, _ = self._pack(first_tokens, second_tokens)
        return tokens

    def encode(self, first, second=None, max_len=None):
        pass

    def batch_encode(self, batch_first, batch_second=None, max_len=None):
        pass

    def decode(self, ids, sequence_length=None):

        try:
            sep = ids.index(self._token_dict[self._token_sep])
        except ValueError:
            return [self._token_unk for i in range(len(ids))]

        try:
            stop = ids.index(self._pad_index)
        except ValueError:
            stop = len(ids)
        tokens = [self._token_dict_inv[i] for i in ids]
        first = tokens[1:sep]
        if sequence_length:
            stop = sequence_length

        if sep < stop - 1:
            second = tokens[sep + 1:stop - 1]
            return first, second
        return first

    @staticmethod
    def _tokenize(text):
        pass

    @staticmethod
    def _truncate(first_tokens, second_tokens=None, max_len=None):
        if max_len is None:
            return

        if second_tokens is not None:
            while True:
                total_len = len(first_tokens) + len(second_tokens)
                if total_len <= max_len - 3:  # 3 for [CLS] .. tokens_a .. [SEP] .. tokens_b [SEP]
                    break
                if len(first_tokens) > len(second_tokens):
                    first_tokens.pop()
                else:
                    second_tokens.pop()
        else:
            del first_tokens[max_len - 2:]  # 2 for [CLS] .. tokens .. [SEP]

    def _pack(self, first_tokens, second_tokens=None):
        first_packed_tokens = [self._token_cls] + first_tokens + [self._token_sep]
        if second_tokens is not None:
            second_packed_tokens = second_tokens + [self._token_sep]
            return first_packed_tokens + second_packed_tokens, len(first_packed_tokens), len(second_packed_tokens)
        else:
            return first_packed_tokens, len(first_packed_tokens), 0


class BertTextTokenizer(BertBaseTokenizer):
    def __init__(self, token_dict, token_cls=TOKEN_CLS, token_sep=TOKEN_SEP, token_unk=TOKEN_UNK, pad_index=0, cased=False):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """

        super(BertTextTokenizer, self).__init__(token_dict, token_cls=token_cls, token_sep=token_sep, token_unk=token_unk, pad_index=pad_index)
        self._cased = cased

    @staticmethod
    def load_from_vocab_txt(bert_vocab_file):
        with open(bert_vocab_file) as f:
            token_dict = dict([(key.strip() if key.strip() else " ", index) for index, key in enumerate(f.readlines())])
        return BertTextTokenizer(token_dict)

    def encode(self, first, second=None, max_len=None):
        first_tokens, first_tokens_length = self._tokenize_split(first.strip().split(' '))
        second_tokens, second_tokens_length = self._tokenize_split(second.strip().split(' ')) if second is not None else [None, None]
        self._truncate(first_tokens, second_tokens, max_len)
        tokens, first_len, second_len = self._pack(first_tokens, second_tokens)

        token_ids = self._convert_tokens_to_ids(tokens)
        segment_ids = [0] * first_len + [1] * second_len

        if max_len is not None:
            assert max_len > 2, "max_len 必须大于2"
            pad_len = max_len - first_len - second_len
            token_ids += [self._pad_index] * pad_len
            segment_ids += [0] * pad_len

        return token_ids, segment_ids, first_tokens_length, second_tokens_length

    def batch_encode(self, batch_first, batch_second=None, max_len=None):
        ids, type_ids, first_token_lengths, second_tokens_lengths = [], [], [], []
        if batch_second is None:
            for i in batch_first:
                indices, segments, first_tokens_length, _ = self.encode(i, max_len=max_len)
                ids.append(indices)
                type_ids.append(segments)
                first_token_lengths.append(first_tokens_length)
            return ids, type_ids, first_token_lengths, None
        else:
            for i, j in zip(batch_first, batch_second):
                if "<s> </s>" in j or j == "":
                    indices, segments, first_tokens_length, second_tokens_length = self.encode(i, max_len=max_len)
                else:
                    indices, segments, first_tokens_length, second_tokens_length = self.encode(i, j, max_len=max_len)
                ids.append(indices)
                type_ids.append(segments)
                first_token_lengths.append(first_tokens_length)
                second_tokens_lengths.append(second_tokens_length)

            return ids, type_ids, first_token_lengths, second_tokens_lengths

    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch) or ch.isdigit():
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

    def _tokenize_split(self, text_list):
        token_length = []
        tokens = []
        for t in text_list:
            token = self._tokenize(t)
            tokens = tokens + token
            token_length.append(len(token))
        return tokens, token_length

    def _word_piece_tokenize(self, word):
        if word in self._token_dict:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens


class BertNerTokenizer(BertBaseTokenizer):
    def __init__(self, token_dict, token_cls=TOKEN_CLS, token_sep=TOKEN_SEP, token_unk=TOKEN_UNK, pad_index=0):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        """
        super(BertNerTokenizer, self).__init__(token_dict, token_cls=token_cls, token_sep=token_sep, token_unk=token_unk, pad_index=pad_index)

    @staticmethod
    def load_from_vocab_txt(vocab_file):
        with open(vocab_file) as f:
            token_dict = dict([(key.strip(), index) for index, key in enumerate(f.readlines())])
        return BertNerTokenizer(token_dict)

    @classmethod
    def load_from_txt(cls, txt_files, vocab_file):
        if os.path.exists(vocab_file):
            return cls.load_from_vocab_txt(vocab_file)
        vocab_set = set()
        for txt_file in txt_files:
            if 'prev' in str(txt_file):
                continue
            with open(txt_file) as f:
                vocab_set = vocab_set | set([word for line in f.readlines() for word in line.split()])
                vocab_set = vocab_set - {TOKEN_PAD, "_UNK", TOKEN_CLS, TOKEN_SEP}  # "_UNK" change to [UNK]
        with open(vocab_file, "w") as f:
            add_token = [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP]
            for token in add_token:
                f.write(f"{token}\n")
            for token in sorted(vocab_set):
                f.write(f"{token}\n")
        return cls.load_from_vocab_txt(vocab_file)

    def encode(self, first, second=None, max_len=None, first_token_length=None, second_token_length=None, feed='slots'):
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        # 根据text中每个token被分成subwword的长度，ner也相应扩展成对应的长度
        # first代表模型输入的前半句，多轮对话的第二句
        if first_token_length:
            first_tokens = self._align_text_ner(first_tokens, first_token_length, feed)
        if second_token_length:
            second_tokens = self._align_text_ner(second_tokens, second_token_length, feed)

        self._truncate(first_tokens, second_tokens, max_len)
        tokens, first_len, second_len = self._pack(first_tokens, second_tokens)
        token_ids = self._convert_tokens_to_ids(tokens)

        if max_len is not None:
            assert max_len > 2, "max_len 必须大于2"
            pad_len = max_len - first_len - second_len
            token_ids += [self._pad_index] * pad_len

        return token_ids

    def batch_encode(self, batch_first, batch_second=None, max_len=None, first_token_lengths=None, second_token_lengths=None, feed='slots'):
        ids = []
        if batch_second is None:
            for i, f in zip(batch_first, first_token_lengths):
                indices = self.encode(i, max_len=max_len, first_token_length=f)
                ids.append(indices)
        else:
            for i, j, f, s in zip(batch_first, batch_second, first_token_lengths, second_token_lengths):
                # first代表模型输入的前半句，多轮对话的第二句
                indices = self.encode(i, j, max_len=max_len, first_token_length=f, second_token_length=s, feed='slots')
                ids.append(indices)
        return ids

    @staticmethod
    def _tokenize(text):
        return text.strip().split()

    @staticmethod
    def _align_text_ner(tokens, token_length, feed):
        tokens_list = []
        for t, l in zip(tokens, token_length):
            if t == 'O' or t == '_UNK' or t.startswith('I-'):
                tokens_list += [t] * l
            elif t.startswith('B-') and feed == 'slots':
                append_token = [t.replace('B-', 'I-')]
                tokens_list += [t] + append_token * (l - 1)
            elif t.startswith('B-') and feed == 'slot_intent':
                tokens_list += [t] * l
            else:
                print('error slots:', tokens)

        assert len(tokens_list) == sum(token_length)

        return tokens_list


class PropertyTokenizer(object):
    def __init__(self):

        self._properties = [
            '<district_name>', '<person_name>', '<music_language>', '<call_org>', '<movie_tv_name>', '<blank>', '<city_name>', '<channel_list>', '<music_name>',
            '<musicStyle>', '<music_artist>', '<music_playMode>', '<music_album>', '<poi_name>', '<addr_name>'
        ]

        self._property_dict = dict([(k, v) for v, k in enumerate(self._properties)])
        self._property_dict_inv = {v: k for k, v in self._property_dict.items()}
        self.property_feature_length = len(self._properties)
        self.pad_property_feature = [0 for _ in range(self.property_feature_length)]
        self.pad_property = "<pad>"

    @staticmethod
    def _truncate(first_tokens, second_tokens=None, max_len=None):
        if max_len is None:
            return

        if second_tokens is not None:
            while True:
                total_len = len(first_tokens) + len(second_tokens)
                if total_len <= max_len - 3:  # 3 for [CLS] .. tokens_a .. [SEP] .. tokens_b [SEP]
                    break
                if len(first_tokens) > len(second_tokens):
                    first_tokens.pop()
                else:
                    second_tokens.pop()
        else:
            del first_tokens[max_len - 2:]  # 2 for [CLS] .. tokens .. [SEP]

    def _pack(self, first_tokens, second_tokens=None):
        first_packed_tokens = [self.pad_property] + first_tokens + [self.pad_property]
        if second_tokens is not None:
            second_packed_tokens = second_tokens + [self.pad_property]
            return first_packed_tokens + second_packed_tokens, len(first_packed_tokens), len(second_packed_tokens)
        else:
            return first_packed_tokens, len(first_packed_tokens), 0

    def _convert_tokens_to_features(self, tokens):
        features = []
        for token in tokens:
            feature = [0 for _ in range(self.property_feature_length)]
            props = token.split(":")
            for i in range(len(props)):
                if props[i] == self.pad_property:
                    break
                index = self._property_dict[props[i]]
                feature[index] = 1
            features.append(feature)
        return features

    def encode(self, first, second=None, max_len=None, first_token_length=None, second_token_length=None):
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        # 根据text中每个token被分成subwword的长度，property也相应扩展成对应的长度
        # first代表模型输入的前半句，多轮对话的第二句
        if first_token_length:
            first_tokens = self._align_text_property(first_tokens, first_token_length)
        if second_token_length:
            second_tokens = self._align_text_property(second_tokens, second_token_length)
        self._truncate(first_tokens, second_tokens, max_len)
        tokens, first_len, second_len = self._pack(first_tokens, second_tokens)
        features = self._convert_tokens_to_features(tokens)

        if max_len is not None:
            assert max_len > 2, "max_len 必须大于2"
            pad_len = max_len - first_len - second_len
            features.extend([self.pad_property_feature for _ in range(pad_len)])

        return features

    def batch_encode(self, batch_first, batch_second=None, max_len=None, first_token_lengths=None, second_token_lengths=None):
        features = []
        if batch_second is None:
            for i, f in zip(batch_first, first_token_lengths):
                feature = self.encode(i, max_len=max_len, first_token_length=f)
                features.append(feature)
        else:
            for i, j, f, s in zip(batch_first, batch_second, first_token_lengths, second_token_lengths):
                # first代表模型输入的前半句，多轮对话的第二句
                feature = self.encode(i, j, max_len=max_len, first_token_length=f, second_token_length=s)
                features.append(feature)
        return features

    @staticmethod
    def _tokenize(text):
        return text.strip().split()

    @staticmethod
    def _align_text_property(tokens, token_length):
        tokens_list = []
        for t, l in zip(tokens, token_length):
            tokens_list += [t] * l
        assert len(tokens_list) == sum(token_length)
        return tokens_list
