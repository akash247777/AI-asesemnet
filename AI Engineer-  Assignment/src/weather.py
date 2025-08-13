from typing import Dict, Any
import requests

try:
    from src.config import get_settings
    from src.llm import build_llm, build_answer_prompt, format_output
except Exception:
    from config import get_settings
    from llm import build_llm, build_answer_prompt, format_output


def fetch_weather(city: str, units: str = "metric") -> Dict[str, Any]:
    settings = get_settings()
    if not settings.openweather_api_key:
        raise ValueError("Missing OPENWEATHER_API_KEY in environment")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": settings.openweather_api_key, "units": units}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def summarize_weather(weather_json: Dict[str, Any], city: str) -> str:
    llm = build_llm()
    prompt = build_answer_prompt(
        "You turn raw weather JSON into a brief, user-friendly summary. Be concise and practical."
    )
    context = str(weather_json)
    question = f"Create a 3-sentence weather summary for {city}."
    result = (prompt | llm).invoke({"context": context, "question": question})
    return format_output(result)


