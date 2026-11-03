"""Row-to-text rules shared by every script. Read this file first.

Text layout, from the S6E10 1st-place solution by Chris Deotte:

    one cell        column is value
    feature text    cells joined by ", " in a random order
    target first    is_fraud is True, amount_usd is 61.16, channel is POS, ...<|endoftext|>

Rules that must not change:
  * Values are the exact strings in the CSV. They are never parsed into numbers, so 3233.0 stays 3233.0.
  * A blank CSV cell is written None.
  * The target is written True or False, whatever the CSV uses (0, 1, 0.0, 1.0, True, False).
  * The tokenizer is GPT-2's own.

RowTokens stores rows as integer codes plus one token list per distinct (column, value). This keeps 700k rows
small, and building a sequence is a list concatenation. check_matches_tokenizer() proves that this gives the same
tokens as tokenizing the whole text.
"""
import numpy as np
import pandas as pd

MISSING_TEXT = "None"
LABEL_TEXT = {"0": "False", "0.0": "False", "False": "False", "1": "True", "1.0": "True", "True": "True"}
LABEL_SPELLINGS = {"truefalse": ("True", "False"), "01": ("1", "0")}


def read_csv_as_strings(path):
    """Read a CSV keeping every value exactly as written. A blank cell stays an empty string."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def feature_columns(df, id_col, target_col):
    """All columns except the id and the target, in CSV order. Row text never relies on this order."""
    return [c for c in df.columns if c not in (id_col, target_col)]


def value_text(csv_string):
    """Text of one value. be_great applies str(value).strip(), so strip here too."""
    text = csv_string.strip()
    return text if text else MISSING_TEXT


def label_text(csv_string, spelling="truefalse"):
    """True or False for one target value. Raises on anything else rather than guessing."""
    value = LABEL_TEXT[csv_string.strip()]
    return LABEL_SPELLINGS[spelling][value == "False"]


def cell_text(column, csv_string):
    return f"{column} is {value_text(csv_string)}"


def features_text(columns, csv_strings, order):
    """Cells joined by ", " in the given order of column positions."""
    return ", ".join(cell_text(columns[j], csv_strings[j]) for j in order)


def target_first_text(target_col, label, body, eos):
    """The whole training or scoring text. Written as the post writes it: prefix + " " + body + eos."""
    return f"{target_col} is {label}, {body}{eos}"


def label_prefix_ids(tokenizer, target_col, spelling="truefalse"):
    """Token ids of "is_fraud is True," and "is_fraud is False,". They must differ in one position only."""
    true, false = LABEL_SPELLINGS[spelling]
    true_ids = tokenizer(f"{target_col} is {true},", add_special_tokens=False)["input_ids"]
    false_ids = tokenizer(f"{target_col} is {false},", add_special_tokens=False)["input_ids"]
    differing = [a != b for a, b in zip(true_ids, false_ids)]
    if len(true_ids) != len(false_ids) or sum(differing) != 1:
        raise ValueError("the two label prefixes must have the same length and differ in exactly one token")
    return true_ids, false_ids


def random_orders(seed_key, n_sequences, n_columns):
    """One random permutation of the column positions per sequence, shape (n_sequences, n_columns).

    The result depends only on seed_key, so a resumed job reproduces it exactly.
    """
    rng = np.random.default_rng(seed_key)
    base = np.tile(np.arange(n_columns, dtype=np.int16), (n_sequences, 1))
    return rng.permuted(base, axis=1)


def anywhere_orders(seed, fold, step, rows, n_columns):
    """Independent row-keyed permutations, including the label's slot (last index)."""
    return np.array([np.random.default_rng([seed, fold, step, int(row)]).permutation(n_columns + 1)
                     for row in rows], dtype=np.int16)


class RowTokens:
    """Token ids for the rows of one table, built only from the rows handed in."""

    def __init__(self, df, columns, tokenizer):
        self.columns = list(columns)
        self.n_rows = len(df)
        self.eos_id = tokenizer.eos_token_id
        self.comma_ids = tokenizer(",", add_special_tokens=False)["input_ids"]
        self.codes = np.empty((self.n_rows, len(self.columns)), dtype=np.int32)
        self.pieces = []
        self.first_pieces = None
        self.tokenizer = tokenizer
        self.unique_values = []
        piece_lengths = []
        for j, column in enumerate(self.columns):
            codes, uniques = pd.factorize(df[column].map(value_text))
            self.unique_values.append(uniques)
            self.codes[:, j] = codes
            texts = [f" {column} is {value}" for value in uniques]
            ids = tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []
            self.pieces.append(ids)
            piece_lengths.append(np.array([len(x) for x in ids], dtype=np.int32))
        separators = (len(self.columns) - 1) * len(self.comma_ids) + 1  # the commas and the end token
        lengths = np.zeros(self.n_rows, dtype=np.int32) + separators
        for j in range(len(self.columns)):
            lengths += piece_lengths[j][self.codes[:, j]]
        self.lengths = lengths  # the token count of a body does not depend on the column order

    def body_ids(self, row, order):
        """Tokens of " col is value, col is value...<eos>" for one row and one order of column positions."""
        codes = self.codes[row].tolist()
        last = len(order) - 1
        ids = []
        for position, j in enumerate(order.tolist() if hasattr(order, "tolist") else order):
            ids += self.pieces[j][codes[j]]
            if position < last:
                ids += self.comma_ids
        ids.append(self.eos_id)
        return ids

    def with_label(self, row, order, label, label_ids, include_field_span=False):
        """Return whole row ids and the label-token span, excluding EOS.

        order is a permutation of feature indices and the final label index.
        """
        if self.first_pieces is None:
            self.first_pieces = [self.tokenizer(
                [f"{column} is {value}" for value in values], add_special_tokens=False)["input_ids"]
                if len(values) else [] for column, values in zip(self.columns, self.unique_values)]
        codes = self.codes[row]
        ids = []
        span = None
        field_span = None
        for position, field in enumerate(order):
            if position:
                ids += self.comma_ids
            start = len(ids)
            if field == len(self.columns):
                ids += label_ids[label][position == 0]
                value_offset = label_ids[label][3 if position == 0 else 2]
                span = (start + value_offset, len(ids))
                field_span = (start, len(ids))
            else:
                pieces = self.first_pieces if position == 0 else self.pieces
                ids += pieces[field][codes[field]]
        ids.append(self.eos_id)
        if include_field_span:
            return ids, span, field_span
        return ids, span


def label_field_ids(tokenizer, target_col, spelling="truefalse"):
    leading_prefix = tokenizer(f" {target_col} is", add_special_tokens=False)["input_ids"]
    bare_prefix = tokenizer(f"{target_col} is", add_special_tokens=False)["input_ids"]
    result = {}
    for label in LABEL_SPELLINGS[spelling]:
        leading = tokenizer(f" {target_col} is {label}", add_special_tokens=False)["input_ids"]
        bare = tokenizer(f"{target_col} is {label}", add_special_tokens=False)["input_ids"]
        if leading[:len(leading_prefix)] != leading_prefix or bare[:len(bare_prefix)] != bare_prefix:
            raise ValueError("label value does not start after a stable field prefix")
        result[label] = (leading, bare, len(leading_prefix), len(bare_prefix))
    return result


def check_anywhere_matches_tokenizer(df, tokens, tokenizer, target_col, n_checks=200, seed=0, spelling="truefalse"):
    if len(df) == 0:
        return
    rng = np.random.default_rng(seed)
    values = df[tokens.columns].to_numpy()
    labels = label_field_ids(tokenizer, target_col, spelling)
    for row in rng.integers(0, len(df), size=n_checks):
        order = rng.permutation(len(tokens.columns) + 1)
        for label in LABEL_SPELLINGS[spelling]:
            fields = [f"{target_col} is {label}" if j == len(tokens.columns)
                      else cell_text(tokens.columns[j], values[row, j]) for j in order]
            whole = tokenizer(", ".join(fields) + tokenizer.eos_token,
                              add_special_tokens=False)["input_ids"]
            built, _ = tokens.with_label(int(row), order, label, labels)
            if whole != built:
                raise ValueError(f"anywhere token mismatch on row {int(row)}")


def check_matches_tokenizer(df, tokens, tokenizer, target_col, n_checks=200, seed=0, spelling="truefalse"):
    """Raise unless RowTokens equals tokenizing the whole text, on a random sample of rows and orders."""
    if len(df) == 0:
        return
    rng = np.random.default_rng(seed)
    values = df[tokens.columns].to_numpy()
    true_ids, _ = label_prefix_ids(tokenizer, target_col, spelling)
    for row in rng.integers(0, len(df), size=n_checks):
        order = rng.permutation(len(tokens.columns))
        body = features_text(tokens.columns, values[row], order)
        text = target_first_text(target_col, LABEL_SPELLINGS[spelling][0], body, tokenizer.eos_token)
        whole = tokenizer(text, add_special_tokens=False)["input_ids"]
        built = true_ids + tokens.body_ids(int(row), order)
        if whole != built:
            raise ValueError(f"token mismatch on row {int(row)}: built tokens differ from the tokenizer")
