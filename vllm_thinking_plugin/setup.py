"""
Setup script for vLLM Thinking Mode Plugin.

To install in development mode (recommended for development):
    pip install -e .

This registers the plugin with vLLM so it can be loaded via VLLM_PLUGINS env var.
"""

from setuptools import setup

setup(
    name="vllm-thinking-plugin",
    version="0.1.0",
    packages=["vllm_thinking"],
    entry_points={
        "vllm.general_plugins": [
            "vllm_thinking = vllm_thinking:vllm_thinking_plugin"
        ]
    },
)
