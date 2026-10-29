"""LLM service retries/config and compliance guardrails."""

from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from app.core.compliance import DISCLAIMER, is_advice_request, screen_output
from app.core.config import Settings
from app.core.logging import scrub
from app.services.llm_service import LLMError, LLMNotConfiguredError, LLMService, parse_json_object


class FakeClient:
    def __init__(self, fail_times: int, text: str = "ok") -> None:
        self.fail_times, self.calls, self.text = fail_times, 0, text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise APITimeoutError(request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))], model=kwargs["model"], usage=None)


def _settings(**kw) -> Settings:
    return Settings(openrouter_api_key="sk-or-test-1234567890", llm_max_retries=2, **kw)


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = FakeClient(fail_times=2)
    resp = LLMService(_settings(), client=client).chat("sys", "user")
    assert resp.text == "ok" and client.calls == 3


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(LLMError):
        LLMService(_settings(), client=FakeClient(fail_times=10)).chat("sys", "user")


def test_not_configured():
    with pytest.raises(LLMNotConfiguredError):
        LLMService(Settings(openrouter_api_key=None)).chat("s", "u")


def test_json_parsing():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_secrets_scrubbed():
    assert "sk-or-" not in scrub("key=sk-or-v1-abcdefghijklmnop failed")
    assert repr(Settings(openrouter_api_key="sk-or-secret-123456").openrouter_api_key) != "sk-or-secret-123456"


@pytest.mark.parametrize("text", ["You should subscribe to this IPO.", "I recommend you buy the shares.",
                                  "This offers guaranteed returns.", "It is a sure-shot listing gain."])
def test_screen_flags_advice(text):
    r = screen_output(text)
    assert r.flagged and "not investment advice" in r.text


@pytest.mark.parametrize("text", ["Revenue grew 25% in Fiscal 2025 (Source: DRHP, page 4).",
                                  "Investors should note the customer concentration risk."])
def test_screen_allows_analysis(text):
    assert not screen_output(text).flagged


def test_advice_question_detection():
    assert is_advice_request("Should I subscribe to this IPO?")
    assert is_advice_request("Is it worth investing in this company?")
    assert not is_advice_request("What are the key risks?")


def test_disclaimer_text():
    assert "does not constitute personalized investment advice" in DISCLAIMER


def test_advice_detector_false_positives_and_positives():
    from app.core.compliance import is_advice_request

    for q in ("What does the company apply the net proceeds to?", "Where can investors apply for shares?",
              "How do I bid in the IPO?", "Who invested in the pre-IPO round?", "Which investors sold shares in the offer?"):
        assert not is_advice_request(q), q
    for q in ("Should I subscribe to this IPO?", "Can I apply for this IPO?", "Is this a good IPO to apply?",
              "Would you recommend investing?", "Is it a good time to invest?", "How should I invest in this offer?"):
        assert is_advice_request(q), q
