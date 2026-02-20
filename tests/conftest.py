"""
Shared pytest configuration for FinRAG tests.

Sets OPENAI_API_KEY to a dummy value so pydantic-settings does not fail
validation when running tests that don't call the OpenAI API.
"""

import os

# Provide a dummy key so Settings() can be imported during tests.
# Tests that need the real key are marked to skip or use a fake LLM.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")
