import pytest
from ..data_parser import parse_tuple_like


def test_parse_tuple_like():
    tuple_like_data = "(59462101751_s.bmp,59469557468_s.bmp,59461911165_s.bmp,59465191869_s.bmp,59467726628_s.bmp,59466495936_s.bmp)"
    assert parse_tuple_like(tuple_like_data) == ('59462101751_s.bmp', '59469557468_s.bmp',
                                                 '59461911165_s.bmp', '59465191869_s.bmp', '59467726628_s.bmp', '59466495936_s.bmp')
