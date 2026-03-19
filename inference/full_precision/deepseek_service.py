from __future__ import annotations

import os
import re
from typing import Optional

from openai import AsyncOpenAI


class DeepSeekService:
    """OpenAI-compatible DeepSeek refinement service."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-chat"
        self.client = None
        self.is_loaded = False

        print("DeepSeek API service initializing...")
        print(f"API Base URL: {self.base_url}")

    async def load(self):
        try:
            if not self.api_key:
                print("DeepSeek API key not provided")
                self.is_loaded = False
                return

            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            self.is_loaded = True
            print("DeepSeek API service is ready!")
        except Exception as exc:
            print(f"DeepSeek API service initialization failed: {exc}")
            self.is_loaded = False

    async def refine_diagnosis(
        self,
        raw_answer: str,
        raw_thinking: Optional[str] = None,
        language: str = "zh",
    ) -> dict:
        if not self.is_loaded or self.client is None:
            error_msg = (
                "API not initialized, cannot generate analysis"
                if language == "en"
                else "API未初始化，无法生成分析过程"
            )
            print("DeepSeek API not initialized, returning original result")
            return {
                "success": False,
                "description": "",
                "analysis_process": raw_thinking or error_msg,
                "diagnosis_result": raw_answer,
                "original_diagnosis": raw_answer,
                "error": "DeepSeek API not initialized",
            }

        try:
            prompt = self._build_refine_prompt(raw_answer, raw_thinking, language)
            system_content = (
                "You are a professional medical text editor. Your task is to polish and organize "
                "medical diagnostic text to make it flow smoothly while preserving the original "
                "meaning. Output ONLY the formatted result. Do NOT add any explanations, comments, "
                "or thoughts. Just follow the format exactly."
                if language == "en"
                else "你是医学文本整理专家，按照用户要求将用户输入的文本整理成用户想要的格式，不要改写或总结。"
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
                top_p=0.8,
            )

            generated_text = response.choices[0].message.content
            parsed = self._parse_refined_output(generated_text, raw_answer, raw_thinking, language)

            return {
                "success": True,
                "description": parsed["description"],
                "analysis_process": parsed["analysis_process"],
                "diagnosis_result": parsed["diagnosis_result"],
                "original_diagnosis": raw_answer,
                "raw_refined": generated_text,
            }
        except Exception as exc:
            print(f"DeepSeek API call failed: {exc}")
            error_msg = (
                "API call failed, cannot generate analysis"
                if language == "en"
                else "API调用失败，无法生成分析过程"
            )
            return {
                "success": False,
                "description": "",
                "analysis_process": raw_thinking or error_msg,
                "diagnosis_result": raw_answer,
                "original_diagnosis": raw_answer,
                "error": str(exc),
            }

    def _build_refine_prompt(
        self,
        raw_answer: str,
        raw_thinking: Optional[str] = None,
        language: str = "zh",
    ) -> str:
        thinking_text = raw_thinking if raw_thinking else "No analysis process available."
        if language == "en":
            return f"""You are a text organization expert. There are two texts that need to be organized. Text 1 is the thinking process of the SkinGPT model, and Text 2 is the diagnosis result given by SkinGPT.

【Requirements】
- Preserve the original tone and expression style
- Text 1 contains the thinking process, Text 2 contains the diagnosis result
- Extract the image observation part from the thinking process as Description. This should include all factual observations about what was seen in the image, not just a brief summary.
- For Diagnostic Reasoning: refine and condense the remaining thinking content. Remove redundancies, self-doubt, circular reasoning, and unnecessary repetition. Keep it concise and not too long. Keep the logical chain clear and enhance readability. IMPORTANT: DO NOT include any image description or visual observations in Diagnostic Reasoning. Only include reasoning, analysis, and diagnostic thought process.
- If [Text 1] content is NOT: No analysis process available. Then organize [Text 1] content accordingly, DO NOT confuse [Text 1] and [Text 2]
- If [Text 1] content IS: No analysis process available. Then extract the analysis process and description from [Text 2]
- DO NOT infer or add new medical information, DO NOT output any meta-commentary
- You may adjust unreasonable statements or remove redundant content to improve clarity

[Text 1]
{thinking_text}

[Text 2]
{raw_answer}

【Output】Only output three sections, do not output anything else:
## Description
(Extract all image observation content from the thinking process - include all factual descriptions of what was seen)

## Analysis Process
(Refined and condensed diagnostic reasoning: remove self-doubt, circular logic, and redundancies. Keep it concise and not too long. Keep logical flow clear. Do NOT include image observations)

## Diagnosis Result
(The organized diagnosis result from Text 2)
"""

        return f"""你是一个文本整理专家。有两段文本需要整理，文本1是SkinGPT模型的思考过程的文本，文本2是SkinGPT给出的诊断结果的文本。

【要求】
- 保留原文的语气和表达方式
- 文本1是思考过程，文本2是诊断结果
- 从思考过程中提取图像观察部分作为图像描述。需要包含所有关于图片中观察到的事实内容，不要简化或缩短。
- 对于分析过程：提炼并精简剩余的思考内容，去除冗余、自我怀疑、兜圈子的内容。保持简洁，不要太长。保持逻辑链条清晰，增强可读性。重要：分析过程中不要包含任何图像描述或视觉观察内容，只包含推理、分析和诊断思考过程。
- 如果【文本1】内容不是：No analysis process available.那么按要求整理【文本1】的内容，不要混淆【文本1】和【文本2】。
- 如果【文本1】内容是：No analysis process available.那么从【文本2】提炼分析过程和描述。
- 【文本1】和【文本2】需要翻译成简体中文
- 禁止推断或添加新的医学信息，禁止输出任何元评论
- 可以调整不合理的语句或去除冗余内容以提高清晰度

【文本1】
{thinking_text}

【文本2】
{raw_answer}

【输出】只输出三个部分，不要输出其他任何内容：
## 图像描述
（从思考过程中提取所有图像观察内容，包含所有关于图片的事实描述）

## 分析过程
（提炼并精简后的诊断推理：去除自我怀疑、兜圈逻辑和冗余内容。保持简洁，不要太长。保持逻辑流畅。不包含图像观察）

## 诊断结果
（整理后的诊断结果）
"""

    def _parse_refined_output(
        self,
        generated_text: str,
        raw_answer: str,
        raw_thinking: Optional[str] = None,
        language: str = "zh",
    ) -> dict:
        description = ""
        analysis_process = None
        diagnosis_result = None

        if language == "en":
            desc_match = re.search(
                r"##\s*Description\s*\n([\s\S]*?)(?=##\s*Analysis\s*Process|$)",
                generated_text,
                re.IGNORECASE,
            )
            analysis_match = re.search(
                r"##\s*Analysis\s*Process\s*\n([\s\S]*?)(?=##\s*Diagnosis\s*Result|$)",
                generated_text,
                re.IGNORECASE,
            )
            result_match = re.search(
                r"##\s*Diagnosis\s*Result\s*\n([\s\S]*?)$",
                generated_text,
                re.IGNORECASE,
            )
            desc_header = "## Description"
            analysis_header = "## Analysis Process"
            result_header = "## Diagnosis Result"
        else:
            desc_match = re.search(r"##\s*图像描述\s*\n([\s\S]*?)(?=##\s*分析过程|$)", generated_text)
            analysis_match = re.search(r"##\s*分析过程\s*\n([\s\S]*?)(?=##\s*诊断结果|$)", generated_text)
            result_match = re.search(r"##\s*诊断结果\s*\n([\s\S]*?)$", generated_text)
            desc_header = "## 图像描述"
            analysis_header = "## 分析过程"
            result_header = "## 诊断结果"

        if desc_match:
            description = desc_match.group(1).strip()
        else:
            description = ""

        if analysis_match:
            analysis_process = analysis_match.group(1).strip()
        else:
            result_pos = generated_text.find(result_header)
            if result_pos > 0:
                analysis_process = generated_text[:result_pos].strip()
                for header in [desc_header, analysis_header]:
                    analysis_process = re.sub(f"{re.escape(header)}\\s*\\n?", "", analysis_process).strip()
            else:
                analysis_process = generated_text[: len(generated_text) // 2].strip()
            if not analysis_process and raw_thinking:
                analysis_process = raw_thinking

        if result_match:
            diagnosis_result = result_match.group(1).strip()
        else:
            result_pos = generated_text.find(result_header)
            if result_pos > 0:
                diagnosis_result = generated_text[result_pos:].strip()
                diagnosis_result = re.sub(
                    f"^{re.escape(result_header)}\\s*\\n?",
                    "",
                    diagnosis_result,
                ).strip()
            else:
                diagnosis_result = generated_text[len(generated_text) // 2 :].strip()
            if not diagnosis_result:
                diagnosis_result = raw_answer

        return {
            "description": description,
            "analysis_process": analysis_process,
            "diagnosis_result": diagnosis_result,
        }


_deepseek_service: Optional[DeepSeekService] = None


async def get_deepseek_service(api_key: Optional[str] = None) -> Optional[DeepSeekService]:
    global _deepseek_service

    if _deepseek_service is None:
        try:
            _deepseek_service = DeepSeekService(api_key=api_key)
            await _deepseek_service.load()
            if not _deepseek_service.is_loaded:
                print("DeepSeek API service initialization failed, will use fallback mode")
                return _deepseek_service
        except Exception as exc:
            print(f"DeepSeek service initialization failed: {exc}")
            return None

    return _deepseek_service
