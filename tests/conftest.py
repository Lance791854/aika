"""Shared test config.

The behavioral tests judge with Groq's free tier, which allows 12k tokens a
minute. Each behavioral test burns ~5k, so pause after each one to keep a
full suite run under the limit. Storage-only tests are unaffected.
"""

import time

import pytest

LLM_TEST_MODULES = (
    "test_unhandled",
    "test_reminders",
    "test_shift_summary",
    "test_stock_and_delete",
)


@pytest.fixture(autouse=True)
def _groq_free_tier_pacing(request):
    yield
    is_llm_module = request.node.module.__name__ in LLM_TEST_MODULES
    is_behavioral = request.node.get_closest_marker("asyncio") is not None
    if is_llm_module and is_behavioral:
        time.sleep(25)
