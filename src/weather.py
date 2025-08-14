from typing import Dict, Any
import requests
import re

try:
    from src.config import get_settings
    from src.llm import build_llm, build_answer_prompt, format_output
except Exception:
    from config import get_settings
    from llm import build_llm, build_answer_prompt, format_output


def _sanitize_city_name(city: str) -> str:
    # Remove trailing punctuation and common temporal words like 'now', 'today', etc.
    cleaned = re.sub(r"[?!.,]+$", "", city).strip()
    stopwords = {"now", "today", "tonight", "tomorrow", "please"}
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    tokens = [t for t in tokens if t.lower() not in stopwords]
    return " ".join(tokens)


def fetch_weather(city: str, units: str = "metric") -> Dict[str, Any]:
    settings = get_settings()
    if not settings.openweather_api_key:
        raise ValueError("Missing OPENWEATHER_API_KEY in environment")

    url = "https://api.openweathermap.org/data/2.5/weather"
    primary_city = _sanitize_city_name(city)
    params = {"q": primary_city, "appid": settings.openweather_api_key, "units": units}
    resp = requests.get(url, params=params, timeout=15)
    try:
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        if resp.status_code == 404:
            # Fallback: try the last token only (e.g., drop 'now' or accidental extras)
            last_only = primary_city.split()[-1] if primary_city.split() else primary_city
            if last_only and last_only != primary_city:
                resp2 = requests.get(url, params={"q": last_only, "appid": settings.openweather_api_key, "units": units}, timeout=15)
                resp2.raise_for_status()
                return resp2.json()
        raise


def summarize_weather(weather_json: Dict[str, Any], city: str) -> str:
    try:
        llm = build_llm()
        prompt = build_answer_prompt(
            "You turn raw weather JSON into a brief, user-friendly summary. Be concise and practical."
        )
        context = str(weather_json)
        question = f"Create a 3-sentence weather summary for {city}."
        try:
            result = (prompt | llm).invoke({"context": context, "question": question})
        except RuntimeError as exc:
            msg = str(exc)
            if "There is no current event loop" in msg or "no running event loop" in msg:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                result = loop.run_until_complete((prompt | llm).ainvoke({"context": context, "question": question}))
            else:
                raise
        return format_output(result)
    except Exception:
        # Fallback deterministic summary without LLM
        main = weather_json.get("weather", [{}])[0].get("description", "weather data")
        temp = weather_json.get("main", {}).get("temp")
        humidity = weather_json.get("main", {}).get("humidity")
        wind = weather_json.get("wind", {}).get("speed")
        parts = [f"Current conditions in {city}: {main}."]
        if temp is not None:
            parts.append(f"Temperature: {temp}°C.")
        if humidity is not None:
            parts.append(f"Humidity: {humidity}%.")
        if wind is not None:
            parts.append(f"Wind: {wind} m/s.")
        return " " .join(parts)


