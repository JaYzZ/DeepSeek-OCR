from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from llamafactory.data.mm_plugin import Qwen3VLPlugin, register_mm_plugin
from llamafactory.data.template import (
    FunctionFormatter,
    ReasoningTemplate,
    StringFormatter,
    ToolFormatter,
    register_template,
)
from llamafactory.extras.constants import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER


@dataclass
class OCRVLQwen3VLPlugin(Qwen3VLPlugin):
    """Qwen3-VL plugin that expands tokens without loading images at tokenization time.

    LlamaFactory's stock Qwen3VLPlugin calls `_get_mm_inputs()` inside `process_messages()`
    to infer `image_grid_thw`, which forces eager image loading during dataset preprocessing.

    For OCRVL, each image always maps to a fixed number of OCR tokens (e.g., 100 visual tokens when
    separators are removed), so we can expand placeholders deterministically and keep image IO
    exclusively in the data collator (training time).
    """

    def _infer_ocr_image_seqlen(self) -> int:
        # Keep consistent with DPSK encoder config: 100 tokens when separators removed, else 111.
        remove_separators = os.environ.get("OCRVL_DPSK_REMOVE_SEPARATORS", "1").strip() != "0"
        return 100 if remove_separators else 111

    def process_messages(  # type: ignore[override]
        self,
        messages,
        images,
        videos,
        audios,
        processor: Optional[object],
    ):
        self._validate_input(processor, images, videos, audios)
        # Skip `_validate_messages` since we expand <image> to multiple <|image_pad|> tokens
        # and images are loaded later in the data collator (not at tokenization time).
        #
        # IMPORTANT: Only treat placeholders in non-assistant messages as media markers.
        # Some alignment data contains literal "<image>" strings inside assistant text
        # (e.g. documentation snippets like "... image stream tag pointed by the <image> ...").
        # Expanding those would desync image tokens vs provided images and crash training.

        # Videos require timestamps/grid inference; keep the stock behavior.
        if len(videos) != 0 or len(audios) != 0:
            return super().process_messages(messages, images, videos, audios, processor)  # type: ignore[misc]

        # Align the images list with the number of placeholders we will actually expand.
        # Mutate in-place so downstream dataset processing/collation sees the corrected list.
        placeholder_budget = 0
        for message in messages:
            if message.get("role") == "assistant":
                continue
            placeholder_budget += message.get("content", "").count(IMAGE_PLACEHOLDER)

        if isinstance(images, list):
            if len(images) > placeholder_budget:
                images[:] = images[:placeholder_budget]

        remaining_images = len(images) if isinstance(images, list) else 0
        messages = [dict(m) for m in messages]

        image_seqlen = self._infer_ocr_image_seqlen() if self.expand_mm_tokens else 1

        for message in messages:
            if message.get("role") == "assistant":
                continue
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if remaining_images > 0:
                    content = content.replace(
                        IMAGE_PLACEHOLDER,
                        f"<|begin_of_image|>{self.image_token * image_seqlen}<|end_of_image|>",
                        1,
                    )
                    remaining_images -= 1
                else:
                    # Drop extra placeholders to avoid token/feature mismatch.
                    content = content.replace(IMAGE_PLACEHOLDER, "", 1)

            # No videos for OCRVL alignment datasets; keep placeholder for safety.
            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, self.video_token or VIDEO_PLACEHOLDER, 1)

            message["content"] = content

        return messages


def register_ocrvl_qwen3_vl_template() -> None:
    """Register OCRVL plugin + templates exactly once per process."""
    plugin_name = "ocrvl_qwen3_vl"
    try:
        register_mm_plugin(plugin_name, OCRVLQwen3VLPlugin)
    except ValueError:
        # Already registered (e.g., in dataloader worker).
        pass

    mm_plugin = OCRVLQwen3VLPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>", audio_token=None)

    # Mirror LlamaFactory's qwen3_vl templates but swap mm_plugin.
    try:
        register_template(
            name="ocrvl_qwen3_vl",
            format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
            format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
            format_observation=StringFormatter(
                slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
            ),
            format_tools=ToolFormatter(tool_format="qwen"),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=mm_plugin,
            template_class=ReasoningTemplate,
        )
    except ValueError:
        pass

    try:
        register_template(
            name="ocrvl_qwen3_vl_nothink",
            format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
            format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
            format_observation=StringFormatter(
                slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
            ),
            format_tools=ToolFormatter(tool_format="qwen"),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=mm_plugin,
        )
    except ValueError:
        pass
