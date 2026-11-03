"""The text round trip: exact values, column order, target first, and tokens built from pieces."""
import numpy as np
import pandas as pd
import pytest

from conftest import ID_COL, TARGET_COL, tiny_table
from tabtext import (RowTokens, cell_text, check_matches_tokenizer, feature_columns, features_text, label_prefix_ids,
                     label_text, random_orders, read_csv_as_strings, target_first_text, value_text)
from train_replica import EpochSeededSampler, original_frame, repeatable_order_dataset


def parse_cells(text):
    """Read 'col is value, col is value' back into a dict. The inverse of features_text."""
    return dict(cell.split(" is ", 1) for cell in text.split(", "))


def test_csv_strings_stay_exact(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("id,a,b,c\n0,3233.0,1.20,\n1,1e-05,007,None\n")
    table = read_csv_as_strings(path)
    assert table["a"].tolist() == ["3233.0", "1e-05"]
    assert table["b"].tolist() == ["1.20", "007"]
    assert table["c"].tolist() == ["", "None"]
    assert cell_text("a", table["a"][0]) == "a is 3233.0"


def test_blank_cell_is_written_none():
    assert value_text("") == "None"
    assert value_text("  ") == "None"
    assert value_text(" 7 ") == "7"


@pytest.mark.parametrize("raw,expected", [("0", "False"), ("0.0", "False"), ("False", "False"), ("1", "True"),
                                          ("1.0", "True"), ("True", "True")])
def test_target_is_true_or_false(raw, expected):
    assert label_text(raw) == expected


def test_unknown_target_raises():
    with pytest.raises(KeyError):
        label_text("2.0")


def test_target_first_text_is_the_posts_string(tokenizer):
    """The post writes 'satisfaction is True,' + ' Gender is Female, ...' + eos."""
    columns = ["Gender", "Age"]
    body = features_text(columns, ["Female", "36"], [0, 1])
    text = target_first_text("satisfaction", "True", body, tokenizer.eos_token)
    assert text == "satisfaction is True," + " Gender is Female, Age is 36" + tokenizer.eos_token


def test_round_trip_through_text_and_tokens(tokenizer):
    table = tiny_table(40, seed=3)
    columns = feature_columns(table, ID_COL, TARGET_COL)
    tokens = RowTokens(table, columns, tokenizer)
    prefix_true, _ = label_prefix_ids(tokenizer, TARGET_COL)
    rng = np.random.default_rng(0)
    for row in range(len(table)):
        order = rng.permutation(len(columns))
        decoded = tokenizer.decode(prefix_true + tokens.body_ids(row, order))
        assert decoded.startswith(f"{TARGET_COL} is True, ")
        assert decoded.endswith(tokenizer.eos_token)
        inner = decoded[len(f"{TARGET_COL} is True, "):-len(tokenizer.eos_token)]
        assert parse_cells(inner) == table.loc[row, columns].to_dict()
        assert [c.split(" is ")[0] for c in inner.split(", ")] == [columns[j] for j in order]


def test_id_and_target_are_not_features():
    table = tiny_table(5)
    columns = feature_columns(table, ID_COL, TARGET_COL)
    assert ID_COL not in columns and TARGET_COL not in columns
    assert len(columns) == len(table.columns) - 2


def test_pieces_equal_tokenizing_the_whole_text(tokenizer):
    table = tiny_table(60, seed=4)
    columns = feature_columns(table, ID_COL, TARGET_COL)
    tokens = RowTokens(table, columns, tokenizer)
    check_matches_tokenizer(table, tokens, tokenizer, TARGET_COL, n_checks=300)
    tokens.pieces[0][0] = tokens.pieces[0][0] + [11]
    with pytest.raises(ValueError):
        check_matches_tokenizer(table, tokens, tokenizer, TARGET_COL, n_checks=300)


def test_a_value_ending_in_punctuation_is_caught(tokenizer):
    """GPT-2 merges adjacent punctuation, so a value like 'A.' would break the pieces. The check must say so."""
    table = pd.DataFrame({"a": ["x.", "y."], "b": ["1", "2"]})
    tokens = RowTokens(table, ["a", "b"], tokenizer)
    with pytest.raises(ValueError):
        check_matches_tokenizer(table, tokens, tokenizer, TARGET_COL, n_checks=50)


def test_token_count_does_not_depend_on_the_order(tokenizer):
    table = tiny_table(30, seed=5)
    columns = feature_columns(table, ID_COL, TARGET_COL)
    tokens = RowTokens(table, columns, tokenizer)
    rng = np.random.default_rng(0)
    for row in range(len(table)):
        for _ in range(5):
            assert len(tokens.body_ids(row, rng.permutation(len(columns)))) == tokens.lengths[row]


def test_label_prefixes_differ_only_in_the_label_token(tokenizer):
    true_ids, false_ids = label_prefix_ids(tokenizer, TARGET_COL)
    assert len(true_ids) == len(false_ids)
    assert [i for i, (a, b) in enumerate(zip(true_ids, false_ids)) if a != b] == [len(true_ids) - 2]
    assert true_ids[-2] == tokenizer(" True")["input_ids"][0]
    assert false_ids[-2] == tokenizer(" False")["input_ids"][0]


def test_random_orders_are_permutations_and_repeatable():
    orders = random_orders([1, 2], 1000, 24)
    assert (np.sort(orders, axis=1) == np.arange(24)).all()
    assert (orders == random_orders([1, 2], 1000, 24)).all()
    assert not (orders == random_orders([1, 3], 1000, 24)).all()
    assert len({tuple(o) for o in orders.tolist()}) == 1000


def test_original_table_given_to_be_great(tmp_path):
    table = tiny_table(10, seed=6, target_true="1", target_false="0").rename(columns={ID_COL: "transaction_id"})
    path = tmp_path / "orig.csv"
    table.to_csv(path, index=False)
    frame = original_frame(path, "transaction_id", TARGET_COL)
    assert "transaction_id" not in frame.columns
    assert set(frame[TARGET_COL]) <= {"True", "False"}
    assert (frame["amount_usd"] == table["amount_usd"]).all()


def test_be_great_writes_the_same_text_with_target_anywhere_and_no_end_token(tmp_path):
    from be_great.great_dataset import GReaTDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # be_great does this itself
    table = tiny_table(10, seed=7).drop(columns=[ID_COL])
    path = tmp_path / "orig.csv"
    table.to_csv(path, index=False)
    frame = original_frame(path, "absent", TARGET_COL)
    dataset = GReaTDataset.from_pandas(frame)
    dataset.set_tokenizer(tokenizer, None)
    seen_positions = set()
    for row in range(len(frame)):
        for _ in range(10):
            text = tokenizer.decode(dataset._getitem(row)["input_ids"])
            assert tokenizer.eos_token not in text
            assert parse_cells(text) == frame.loc[row].to_dict()
            seen_positions.add(list(parse_cells(text)).index(TARGET_COL))
    assert len(seen_positions) > 1


def test_replica_orders_are_fresh_per_epoch_and_row_but_repeatable(tmp_path):
    from be_great.great_dataset import GReaTDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    table = tiny_table(20, seed=8).drop(columns=[ID_COL])
    path = tmp_path / "orig.csv"
    table.to_csv(path, index=False)
    frame = original_frame(path, "absent", TARGET_COL)
    dataset_class = repeatable_order_dataset(GReaTDataset, seed=42)
    dataset = dataset_class.from_pandas(frame)
    dataset.set_tokenizer(tokenizer, None)

    def order_of(row, epoch):
        dataset_class.epoch = epoch
        text = tokenizer.decode(dataset._getitem(row)["input_ids"])
        assert parse_cells(text) == frame.loc[row].to_dict()
        return list(parse_cells(text))

    assert order_of(3, 0) == order_of(3, 0)
    assert len({tuple(order_of(row, 0)) for row in range(20)}) > 10
    assert len({tuple(order_of(3, epoch)) for epoch in range(20)}) > 10


def test_replica_row_order_is_a_fresh_full_shuffle_per_epoch_and_repeatable():
    class Holder:
        epoch = 0

    sampler = EpochSeededSampler(50, 42, Holder)
    first = list(sampler)
    assert sorted(first) == list(range(50)) and list(sampler) == first and len(sampler) == 50
    Holder.epoch = 1
    assert list(sampler) != first
