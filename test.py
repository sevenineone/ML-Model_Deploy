import pytest
import server

@pytest.fixture
def app():
    app = server.app
    return app

def test_example(client):
    response = client.post("/predict", json={"sepal_length": 5.1,"sepal_width": 3.5,"petal_length": 1.4,"petal_width":0.2})
    assert response.status_code == 200