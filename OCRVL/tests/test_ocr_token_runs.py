import torch

from OCRVL.model.ocr_llava_arch import _find_image_token_runs


def test_basic_runs():
    image_id = 999
    ids = torch.tensor([1, 2, image_id, image_id, 3, image_id, 4, image_id, image_id, image_id, 5])
    runs = _find_image_token_runs(ids, image_id)
    assert runs == [(2, 4), (5, 6), (7, 10)]


def test_no_runs():
    ids = torch.tensor([1, 2, 3, 4])
    runs = _find_image_token_runs(ids, 999)
    assert runs == []


if __name__ == "__main__":
    test_basic_runs()
    test_no_runs()
    print("OK")

