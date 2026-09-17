from tools import count_characters


def test_count_characters_null_text():
    result = count_characters(None)
    assert result == "Total characters=0, Chinese characters=0"


def test_count_characters_normal():
    result = count_characters("你好hi")
    assert "Total characters=4" in result
    assert "Chinese characters=2" in result
