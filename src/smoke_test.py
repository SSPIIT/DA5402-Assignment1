import requests
import json

BASE_URL = "http://127.0.0.1:5000"
PREDICT_URL = f"{BASE_URL}/predict"

def test_home():
    try:
        r = requests.get(BASE_URL)
        assert r.status_code == 200
        print("Test 1 passed home endpoint working")
    except:
        print("Test 1 failed")

def test_valid_prediction():
    try:
        sample_input = {
            "features": [300, 305, 1500, 45, 5, 0, 0, 0, 0, 0, 0]
        }
        r = requests.post(PREDICT_URL, json=sample_input)

        assert r.status_code == 200
        response = r.json()

        assert "prediction" in response
        assert "model_version" in response

        print("Test 2 passed valid prediction working")

    except:
        print("Test 2 failed")

def test_invalid_input():
    try:
        wrong_input = {
            "wrong_key": [1,2,3]
        }

        r = requests.post(PREDICT_URL, json=wrong_input)

        assert r.status_code != 200

        print("Test 3 passed invalid input rejected")

    except:
        print("Test 3 failed")


if __name__ == "__main__":
    print("\nsmoke tests\n")
    test_home()
    test_valid_prediction()
    test_invalid_input()
