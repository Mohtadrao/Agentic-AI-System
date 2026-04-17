import app


def test_parse_tool_arguments_handles_json_string():
    assert app.parse_tool_arguments('{"city": "Karachi"}') == {"city": "Karachi"}


def test_weather_vector_store_returns_reasonable_match():
    result = app.get_weather("Karachi", "what is the weather in karachi today")
    assert result["matched_city"] == "Karachi"
    assert result["retrieval_backend"] in {"faiss", "numpy"}


def test_unique_sources_allows_public_urls_and_blocks_private():
    sources = app.unique_sources([
        {"name": "Wiki", "url": "https://en.wikipedia.org/wiki/Karachi"},
        {"name": "Private", "url": "http://127.0.0.1:8080/x"},
        {"name": "Wiki Dup", "url": "https://en.wikipedia.org/wiki/Karachi"},
    ])
    assert sources == [{"name": "Wiki", "url": "https://en.wikipedia.org/wiki/Karachi"}]


def test_execute_tool_call_validates_bad_args():
    trace, _ = app.execute_tool_call({"function": {"name": "get_weather", "arguments": {}}})
    assert "error" in trace["result"]


def test_ask_endpoint_returns_weather_without_ollama():
    client = app.app.test_client()
    res = client.post("/ask", json={"query": "What is the weather in Karachi?"})
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["tool_trace"][0]["tool"] == "get_weather"
    assert "get_weather" in payload["answer"]