import os
import pytest

from src.weather import fetch_weather


@pytest.mark.skipif(not os.getenv("OPENWEATHER_API_KEY"), reason="OPENWEATHER_API_KEY not set")
def test_fetch_weather_paris():
    data = fetch_weather("Paris")
    assert "weather" in data and "main" in data


