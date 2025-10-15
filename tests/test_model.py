from src.model import create_model

def test_model_creation():
    model = create_model()
    assert model is not None
