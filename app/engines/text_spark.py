"""Text Generation Engine for Spark-X2.5-4B (1M Context) using Apple MLX."""

from __future__ import annotations

import asyncio
import logging
import json
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from app.config import settings
from app.engines.base import BaseModelEngine

logger = logging.getLogger("local-infra.text")

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate
    from mlx_lm.generate import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore
    mlx_load = None  # type: ignore
    stream_generate = None  # type: ignore
    make_sampler = None  # type: ignore
    MLX_AVAILABLE = False


class SparkTextEngine(BaseModelEngine):
    """Engine for XHToken/Spark-X2.5-4B (4.1B parameters, 1M context window)."""

    def __init__(self, model_id: str | None = None) -> None:
        selected_model = (
            model_id
            or (settings.text_model_bf16 if settings.prefer_bf16_text else settings.text_model)
        )
        super().__init__(model_id=selected_model, modality="text")
        self.is_simulated = False

    def estimate_vram_gb(self) -> float:
        if "4bit" in self.model_id.lower():
            return 2.5
        elif "8bit" in self.model_id.lower():
            return 4.5
        return 8.2  # Native BF16 unquantized

    def load(self) -> None:
        if not MLX_AVAILABLE:
            if settings.enable_simulation_fallback:
                logger.warning("MLX not available. Starting SparkTextEngine in simulation mode.")
                self.is_simulated = True
                self.is_loaded = True
                return
            raise RuntimeError("MLX is not installed in the current environment.")

        # Configure Metal wired memory safety limit
        if mx.metal.is_available():
            limit_bytes = int(settings.max_wired_memory_gb * 1024**3)
            try:
                mx.set_wired_limit(limit_bytes)
                logger.info(f"Set Apple Metal wired limit to {settings.max_wired_memory_gb} GB")
            except Exception:
                pass

        try:
            logger.info(f"Loading MLX model '{self.model_id}'...")
            self._raw_model, self._raw_tokenizer = mlx_load(self.model_id)
            self.is_loaded = True
            self.is_simulated = False
            logger.info(f"Model '{self.model_id}' successfully loaded into Metal VRAM.")
        except Exception as exc:
            if settings.enable_simulation_fallback:
                logger.warning(
                    f"Could not load weights for '{self.model_id}' ({exc}). "
                    f"Operating in simulation fallback mode."
                )
                self.is_simulated = True
                self.is_loaded = True
            else:
                raise exc

    def extract_tool_calls(
        self,
        content: str,
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Multi-format tool call extractor supporting:
        1. Spark native XML: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        2. Hermes / Nous JSON: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        3. XML with name attribute: <tool_call name="...">...</tool_call>
        4. Function tag: <function_call>...</function_call>
        5. Fenced markdown JSON: ```json {"name": "...", "arguments": {...}} ```
        """
        tool_calls: List[Dict[str, Any]] = []
        clean_parts: List[str] = []
        last_idx = 0

        # Pattern matches <tool_call> or <function_call> with optional attributes
        tag_pattern = re.compile(
            r"<(tool_call|function_call)(?:\s+name=[\"']([^\"']+)[\"'])?\s*>(.*?)(?:</\1>|$)",
            re.DOTALL | re.IGNORECASE,
        )

        matches = list(tag_pattern.finditer(content))
        if matches:
            for match in matches:
                clean_parts.append(content[last_idx : match.start()])
                last_idx = match.end()
                attr_name = match.group(2)
                raw_body = match.group(3).strip()
                if not raw_body and not attr_name:
                    continue

                fn_name = attr_name
                fn_args: Any = {}

                # 1. Hermes / OpenAI JSON format inside tag
                if raw_body.startswith("{") or '"name"' in raw_body or '"arguments"' in raw_body:
                    try:
                        json_start = raw_body.find("{")
                        json_end = raw_body.rfind("}") + 1
                        if json_start != -1 and json_end > json_start:
                            parsed_json = json.loads(raw_body[json_start:json_end])
                            if isinstance(parsed_json, dict):
                                if not fn_name:
                                    fn_name = parsed_json.get("name")
                                parsed_args = parsed_json.get("arguments") or parsed_json.get("parameters") or parsed_json
                                if "name" in parsed_args and ("arguments" in parsed_args or "parameters" in parsed_args):
                                    parsed_args = parsed_args.get("arguments") or parsed_args.get("parameters")
                                fn_args = parsed_args
                    except Exception:
                        pass

                # 2. Spark XML format: <tool_call>fn_name<arg_key>k</arg_key><arg_value>v</arg_value>
                if not fn_name:
                    m = re.match(r"^([a-zA-Z0-9_\-]+)(.*)$", raw_body, re.DOTALL)
                    if m:
                        fn_name = m.group(1).strip()
                        remainder = m.group(2)
                        args_dict: Dict[str, Any] = {}
                        arg_matches = re.finditer(
                            r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>",
                            remainder,
                            re.DOTALL,
                        )
                        for am in arg_matches:
                            k = am.group(1).strip()
                            v_raw = am.group(2).strip()
                            try:
                                v = json.loads(v_raw)
                            except Exception:
                                v = v_raw
                            args_dict[k] = v
                        if args_dict or not fn_args:
                            fn_args = args_dict

                if fn_name:
                    if isinstance(fn_args, (dict, list)):
                        args_str = json.dumps(fn_args)
                    elif isinstance(fn_args, str):
                        # Ensure valid JSON string
                        try:
                            json.loads(fn_args)
                            args_str = fn_args
                        except Exception:
                            args_str = json.dumps(fn_args)
                    else:
                        args_str = json.dumps(fn_args)

                    tool_calls.append({
                        "id": f"call_{len(tool_calls)+1}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str,
                        },
                    })

            clean_parts.append(content[last_idx:])
            clean_content = "".join(clean_parts).strip()
            return clean_content, tool_calls

        # Fallback: check for markdown code blocks if available_tools is present
        if available_tools:
            known_names = set()
            for t in available_tools:
                if isinstance(t, dict):
                    fn = t.get("function", t)
                    if "name" in fn:
                        known_names.add(fn["name"])

            cb_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
            cb_clean_parts = []
            cb_last = 0
            for cb in cb_pattern.finditer(content):
                cb_clean_parts.append(content[cb_last : cb.start()])
                cb_last = cb.end()
                try:
                    data = json.loads(cb.group(1))
                    if isinstance(data, dict) and data.get("name") in known_names:
                        raw_a = data.get("arguments") or data.get("parameters") or {}
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)+1}",
                            "type": "function",
                            "function": {
                                "name": data["name"],
                                "arguments": json.dumps(raw_a) if isinstance(raw_a, (dict, list)) else str(raw_a),
                            },
                        })
                except Exception:
                    pass
            if tool_calls:
                cb_clean_parts.append(content[cb_last:])
                return "".join(cb_clean_parts).strip(), tool_calls

        return content.strip(), []

    def _parse_spark_output(self, raw_text: str):
        """Extract reasoning (<think>) and tool calls from raw model output."""
        reasoning = None
        content = raw_text

        if "</think>" in raw_text:
            parts = raw_text.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
        elif "<think>" in raw_text:
            reasoning = raw_text.replace("<think>", "").strip()
            content = ""

        clean_content, tool_calls = self.extract_tool_calls(content)
        return reasoning, clean_content, tool_calls

    def _normalize_messages_for_template(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize messages so Jinja template receives dictionary arguments
        for tool_calls (preventing 'arguments must be a dictionary' error)
        and string content for tool messages."""
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            item = dict(msg)
            # Ensure content is string for tool messages
            if item.get("role") == "tool":
                c = item.get("content", "")
                if not isinstance(c, str):
                    item["content"] = json.dumps(c) if c is not None else ""

            # Ensure tool_call arguments are dictionaries for apply_chat_template
            if item.get("tool_calls"):
                norm_calls = []
                for tc in item["tool_calls"]:
                    tc_copy = dict(tc)
                    fn = dict(tc_copy.get("function", {}))
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except Exception:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    fn["arguments"] = args
                    tc_copy["function"] = fn
                    norm_calls.append(tc_copy)
                item["tool_calls"] = norm_calls

            normalized.append(item)
        return normalized

    def _format_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> str:
        """Format OpenAI-style chat messages into model prompt."""
        active_tools = tools
        if tool_choice == "none":
            active_tools = None

        norm_messages = self._normalize_messages_for_template(messages)

        # Inject tool_choice requirement if specified
        if active_tools and isinstance(tool_choice, dict):
            req_fn = tool_choice.get("function", {}).get("name")
            if req_fn:
                norm_messages[-1]["content"] += f"\n[Instruction: You must use the tool '{req_fn}' to respond.]"
        elif active_tools and tool_choice == "required":
            norm_messages[-1]["content"] += "\n[Instruction: You must call an available tool to respond.]"

        if self._raw_tokenizer and hasattr(self._raw_tokenizer, "apply_chat_template"):
            try:
                kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
                if active_tools:
                    kwargs["tools"] = active_tools
                return self._raw_tokenizer.apply_chat_template(norm_messages, **kwargs)
            except Exception as exc:
                logger.debug(f"apply_chat_template fallback: {exc}")

        # Native Spark fallback template
        lines: List[str] = []
        default_system = "you are a helpful assistant."
        initial_system = ""
        user_turns = []

        for msg in norm_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" and not initial_system:
                initial_system = content
            else:
                user_turns.append(msg)

        sys_block = f"<｜start▁of▁sentence｜><|System|>\n{default_system}"
        if active_tools:
            sys_block += "\n## Tools\nYou have access to the following functions:\n<tools>"
            for t in active_tools:
                fn = t.get("function", t)
                sys_block += f"\n{json.dumps(fn)}"
            sys_block += "\n</tools>"
        if initial_system:
            sys_block += f"\n\n{initial_system}"
        sys_block += "<｜end▁of▁sentence｜>"
        lines.append(sys_block)

        for msg in user_turns:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"<｜start▁of▁sentence｜><|User|>{content}<｜end▁of▁sentence｜>")
            elif role == "assistant":
                tc_str = ""
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        tc_str += f"<tool_call>{fn_name}"
                        if isinstance(args, dict):
                            for k, v in args.items():
                                val_str = v if isinstance(v, str) else json.dumps(v)
                                tc_str += f"<arg_key>{k}</arg_key><arg_value>{val_str}</arg_value>"
                        tc_str += "</tool_call>"
                reasoning = msg.get("reasoning_content", "")
                reasoning_str = f"<think>{reasoning}</think>" if reasoning else "</think>"
                lines.append(f"<｜start▁of▁sentence｜><|Bot|>{reasoning_str}{content}{tc_str}<｜end▁of▁sentence｜>")
            elif role == "tool":
                lines.append(f"<｜start▁of▁sentence｜><|Tool|><tool_response>{content}</tool_response><｜end▁of▁sentence｜>")

        lines.append("<｜start▁of▁sentence｜><|Bot|><think>")
        return "".join(lines)

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream generated tokens formatted as OpenAI chat completion chunks with
        support for reasoning_content, live content, and structured tool_calls."""
        prompt = self._format_prompt(messages, tools=tools, tool_choice=tool_choice)
        created = int(time.time())
        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"

        # Stop sequences normalization
        stop_list: List[str] = []
        if isinstance(stop, str):
            stop_list = [stop]
        elif isinstance(stop, list):
            stop_list = [s for s in stop if isinstance(s, str)]

        if self.is_simulated or not self._raw_model:
            sim_thinking = "Analyzing request and determining tool invocation in simulation mode..."
            for word in sim_thinking.split(" "):
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": word + " "},
                            "finish_reason": None,
                        }
                    ],
                }

            if tools and tool_choice != "none":
                first_tool = tools[0].get("function", tools[0])
                fn_name = first_tool.get("name", "simulated_tool")
                props = first_tool.get("parameters", {}).get("properties", {})
                mock_args: Dict[str, Any] = {}
                for p_name, p_spec in props.items():
                    p_type = p_spec.get("type", "string")
                    if p_type == "string":
                        mock_args[p_name] = "simulated_value"
                    elif p_type in ("integer", "number"):
                        mock_args[p_name] = 42
                    elif p_type == "boolean":
                        mock_args[p_name] = True
                    else:
                        mock_args[p_name] = "value"
                args_json = json.dumps(mock_args)
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_sim_1",
                                        "type": "function",
                                        "function": {
                                            "name": fn_name,
                                            "arguments": args_json,
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
                return

            sim_text = (
                f"Spark-X2.5-4B (1M Context) [Simulation Mode]: "
                f"Processed prompt with {len(prompt)} chars. "
                f"Production streaming pipeline active on port 8080."
            )
            for word in sim_text.split(" "):
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            return

        # True live real-time token streaming with background producer thread
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()

        def _produce_tokens():
            try:
                sampler = make_sampler(temp=temperature, top_p=top_p) if make_sampler else None
                accum_text = ""
                for res in stream_generate(
                    self._raw_model,
                    self._raw_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    accum_text += res.text
                    loop.call_soon_threadsafe(q.put_nowait, res.text)
                    if stop_list and any(st in accum_text for st in stop_list):
                        break
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        producer_thread = threading.Thread(target=_produce_tokens, daemon=True)
        producer_thread.start()

        in_thinking = True
        content_buffer = ""
        in_tool_call = False
        emitted_tool_calls: List[Dict[str, Any]] = []

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                logger.error(f"Error during token generation: {item}")
                break
            if not item:
                continue

            if in_thinking:
                if "</think>" in item:
                    in_thinking = False
                    parts = item.split("</think>", 1)
                    if parts[0]:
                        yield {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": parts[0]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    remainder = parts[1]
                    if remainder:
                        content_buffer += remainder
                elif "<tool_call" in item or "<function_call" in item:
                    # Model omitted </think> and immediately began tool call
                    in_thinking = False
                    content_buffer += item
                else:
                    yield {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": item},
                                "finish_reason": None,
                            }
                        ],
                    }
                    continue

            if not in_thinking:
                if item and "</think>" not in item:
                    content_buffer += item

                while True:
                    if not in_tool_call:
                        tc_start = content_buffer.find("<tool_call")
                        fc_start = content_buffer.find("<function_call")
                        candidates = [pos for pos in (tc_start, fc_start) if pos != -1]
                        earliest_start = min(candidates) if candidates else -1

                        if earliest_start != -1:
                            gt_pos = content_buffer.find(">", earliest_start)
                            if gt_pos != -1:
                                prefix = content_buffer[:earliest_start]
                                if prefix:
                                    yield {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": self.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": prefix},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                content_buffer = content_buffer[earliest_start:]
                                in_tool_call = True
                                continue
                            else:
                                prefix = content_buffer[:earliest_start]
                                if prefix:
                                    yield {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": self.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": prefix},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                content_buffer = content_buffer[earliest_start:]
                                break
                        else:
                            # Hold back partial tag prefix
                            held_len = 0
                            for pattern in ("<tool_call>", "<function_call>"):
                                for k in range(min(len(content_buffer), len(pattern) - 1), 0, -1):
                                    if pattern.startswith(content_buffer[-k:]):
                                        held_len = max(held_len, k)
                            if held_len > 0:
                                emit_text = content_buffer[:-held_len]
                                content_buffer = content_buffer[-held_len:]
                            else:
                                emit_text = content_buffer
                                content_buffer = ""
                            if emit_text:
                                yield {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": self.model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": emit_text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            break
                    else:
                        tc_end = content_buffer.find("</tool_call>")
                        fc_end = content_buffer.find("</function_call>")
                        end_candidates = []
                        if tc_end != -1:
                            end_candidates.append((tc_end, len("</tool_call>")))
                        if fc_end != -1:
                            end_candidates.append((fc_end, len("</function_call>")))

                        if end_candidates:
                            end_candidates.sort(key=lambda x: x[0])
                            earliest_end, tag_len = end_candidates[0]
                            end_pos = earliest_end + tag_len
                            full_tc = content_buffer[:end_pos]
                            content_buffer = content_buffer[end_pos:]
                            in_tool_call = False
                            _, parsed_calls = self.extract_tool_calls(full_tc, available_tools=tools)
                            for tc in parsed_calls:
                                tc_idx = len(emitted_tool_calls)
                                emitted_tool_calls.append(tc)
                                yield {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": self.model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "tool_calls": [
                                                    {
                                                        "index": tc_idx,
                                                        "id": tc["id"],
                                                        "type": "function",
                                                        "function": {
                                                            "name": tc["function"]["name"],
                                                            "arguments": tc["function"]["arguments"],
                                                        },
                                                    }
                                                ],
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            continue
                        else:
                            break

        # Flush any remaining buffer at end of generation
        if content_buffer:
            if in_tool_call or "<tool_call" in content_buffer or "<function_call" in content_buffer:
                _, parsed_calls = self.extract_tool_calls(content_buffer, available_tools=tools)
                if parsed_calls:
                    for tc in parsed_calls:
                        tc_idx = len(emitted_tool_calls)
                        emitted_tool_calls.append(tc)
                        yield {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "index": tc_idx,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": tc["function"]["arguments"],
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                else:
                    yield {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content_buffer},
                                "finish_reason": None,
                            }
                        ],
                    }
            elif tools:
                clean_txt, parsed_calls = self.extract_tool_calls(content_buffer, available_tools=tools)
                if parsed_calls:
                    if clean_txt:
                        yield {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": clean_txt},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    for tc in parsed_calls:
                        tc_idx = len(emitted_tool_calls)
                        emitted_tool_calls.append(tc)
                        yield {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "index": tc_idx,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": tc["function"]["arguments"],
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                else:
                    yield {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content_buffer},
                                "finish_reason": None,
                            }
                        ],
                    }
            else:
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content_buffer},
                            "finish_reason": None,
                        }
                    ],
                }

        finish_reason = "tool_calls" if emitted_tool_calls else "stop"
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    async def generate_non_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Generate non-streaming response."""
        reasoning_accum = ""
        content_accum = ""
        tool_calls_dict: Dict[int, Dict[str, Any]] = {}
        finish_reason = "stop"

        async for chunk in self.generate_stream(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            if "reasoning_content" in delta and delta["reasoning_content"]:
                reasoning_accum += delta["reasoning_content"]
            if "content" in delta and delta["content"]:
                content_accum += delta["content"]
            if "tool_calls" in delta and delta["tool_calls"]:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", len(tool_calls_dict))
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tc.get("id", f"call_{idx+1}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", ""),
                            },
                        }
                    else:
                        existing = tool_calls_dict[idx]
                        fn_info = tc.get("function", {})
                        if "name" in fn_info and fn_info["name"]:
                            existing["function"]["name"] = fn_info["name"]
                        if "arguments" in fn_info:
                            existing["function"]["arguments"] += fn_info["arguments"]

        tool_calls = [tool_calls_dict[k] for k in sorted(tool_calls_dict.keys())] if tool_calls_dict else None

        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": content_accum if content_accum or not tool_calls else None,
        }
        if reasoning_accum.strip():
            message_dict["reasoning_content"] = reasoning_accum.strip()
        if tool_calls:
            message_dict["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(messages),
                "completion_tokens": len(reasoning_accum.split()) + len(content_accum.split()),
                "total_tokens": len(messages) + len(reasoning_accum.split()) + len(content_accum.split()),
            },
        }
