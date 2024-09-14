import unittest
from dataclasses import dataclass

from pytorch_model_commons.utils import dict_to_dataclass


@dataclass
class DummyClass:
    some_field_0: str
    some_field_1: int
    some_field_2: str


class TestDictToDataclass(unittest.TestCase):
    def test_valid_dict(self):
        data_dict = {"some_field_0": "value0", "some_field_1": 10, "some_field_2": "value2"}
        instance = dict_to_dataclass(DummyClass, data_dict)
        self.assertEqual(instance, DummyClass(some_field_0="value0", some_field_1=10, some_field_2="value2"))

    def test_dict_with_extra_keys(self):
        data_dict = {
            "some_field_0": "value0",
            "some_field_1": 10,
            "some_field_2": "value2",
            "extra_field": "extra_value",
        }
        instance = dict_to_dataclass(DummyClass, data_dict)
        self.assertEqual(instance, DummyClass(some_field_0="value0", some_field_1=10, some_field_2="value2"))

    def test_dict_with_missing_keys(self):
        data_dict = {
            "some_field_0": "value0",
            # Missing 'some_field_1' and 'some_field_2'
        }
        with self.assertRaises(TypeError):
            dict_to_dataclass(DummyClass, data_dict)
