import os
import sys
import asyncio
import traceback
import subprocess
import importlib
import re
import io
from typing import Dict, List, Any, Optional, Tuple
from io import StringIO
import pandas as pd
from pathlib import Path

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from utils.environment_checker import EnvironmentChecker
from .base_agent import Agent

class CodeAgent(Agent):
    """處理代碼生成和執行的智能代理，支持多種程式語言和依賴管理，帶有自動錯誤修復和測試數據生成功能"""
    
    def __init__(self, name: str = "CodeAgent"):
        """
        初始化代碼代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name, skills=["代碼生成", "代碼執行", "代碼解釋", "代碼除錯", "多語言支持", "依賴管理", "自動錯誤修復", "測試數據生成"])
        self.code_gen_function = None
        self.code_fix_function = None
        self.test_data_gen_function = None  # 新增：測試數據生成功能
        self.supported_languages = ["python", "javascript", "java", "c++", "c#", "go", "ruby", "php", "rust", "typescript", "bash", "r", "sql"]
        self.allow_installs = True  # 是否允許安裝新的依賴
        self.max_fix_attempts = 3   # 最大修復嘗試次數
    
    def setup_kernel(self, kernel: Kernel):
        """
        設置 Semantic Kernel 並註冊代碼生成功能
        
        Args:
            kernel: Semantic Kernel 實例
        """
        super().setup_kernel(kernel)
        self._register_code_gen_function()
        self._register_code_fix_function()
        self._register_test_data_gen_function()  # 註冊測試數據生成功能
    
    def _register_code_gen_function(self):
        """註冊代碼生成功能"""
        
        # 代碼生成提示模板 - 增強版
        code_gen_prompt = """
        請根據以下任務生成可執行的程式碼。
        
        任務: {{$task}}
        
        你可以使用任何適合的程式語言來實現此任務。如果任務中沒有指定程式語言，請選擇最適合的語言。
    
        重要指引:
        - 除非用戶明確要求保存或產生檔案，否則不要在代碼中包含檔案寫入操作。
        - 普通的代碼示例不應該自動保存結果到檔案中。
        - 將計算結果儲存在 'result' 變數中，不要寫入檔案。
        
        僅在任務明確要求生成檔案時（例如明確包含「生成檔案」、「保存為」、「輸出到文件」等詞語），才實現檔案生成功能，並遵循以下原則：
        1. 將生成的檔案保存到 "downloads" 目錄（如不存在則創建）
        2. 使用適當的文件名
        3. 在執行完成後顯示檔案路徑
        
        如果是一般寫code的任務(非生成檔案的任務)你的回應必須包含以下部分:
        1. 程式語言: 你選擇使用的程式語言名稱
        2. 依賴項: 此代碼所需的所有依賴庫/模組列表，精確到版本號（如果重要）
        3. 完整代碼: 完整、可執行的代碼，包含所有必要的導入語句
        4. 説明: 代碼的簡要説明和使用方法
        
        確保代碼:
        - 完整且可立即執行
        - 包含必要的錯誤處理
        - 如果是 Python，將最終結果存儲在 'result' 變數中
        - 包含必要的註釋解釋關鍵邏輯
        
        按照以下格式回覆：
        
        LANGUAGE: [程式語言名稱]
        
        DEPENDENCIES:
        [依賴1]
        [依賴2]
        ...
        
        CODE:
        [完整代碼]
        
        EXPLANATION:
        [代碼説明和使用方法]
        
        只返回上述格式，不要有其他多餘的解釋或 markdown 標記。
        """
        
        # 代碼生成配置
        code_gen_config = PromptTemplateConfig(
            template=code_gen_prompt,
            name="generateSmartCode",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="task", description="需要通過代碼實現的任務", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=3000,
                temperature=0.2,
            )
        )
        
        # 添加代碼生成功能到 Kernel
        self.code_gen_function = self.kernel.add_function(
            function_name="generateSmartCode",
            plugin_name="codePlugin",
            prompt_template_config=code_gen_config,
        )
    
    def _register_code_fix_function(self):
        """註冊代碼修復功能"""
        
        # 代碼修復提示模板
        code_fix_prompt = """
        我需要你幫我修復下面的代碼，該代碼執行時遇到了錯誤。

        原始任務: {{$original_task}}
        
        程式語言: {{$language}}
        
        原始代碼:
        ```
        {{$code}}
        ```
        
        執行錯誤:
        ```
        {{$error_message}}
        ```
        
        請分析錯誤原因，然後提供修復後的完整代碼。你的回應必須包含以下部分:
        
        1. 錯誤分析: 簡要說明錯誤的原因
        2. 修復方案: 描述你的修復方法
        3. 完整代碼: 修復後的完整代碼，確保代碼可以執行
        
        按照以下格式回覆：
        
        ERROR_ANALYSIS:
        [錯誤原因分析]
        
        FIX_APPROACH:
        [修復方案描述]
        
        FIXED_CODE:
        [完整修復後的代碼]
        
        只返回上述格式，不要有其他多餘的解釋或 markdown 標記。
        """
        
        # 代碼修復配置
        code_fix_config = PromptTemplateConfig(
            template=code_fix_prompt,
            name="fixBrokenCode",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="original_task", description="原始任務描述", is_required=True),
                InputVariable(name="language", description="程式語言", is_required=True),
                InputVariable(name="code", description="需要修復的代碼", is_required=True),
                InputVariable(name="error_message", description="錯誤訊息", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=3000,
                temperature=0.2,
            )
        )
        
        # 添加代碼修復功能到 Kernel
        self.code_fix_function = self.kernel.add_function(
            function_name="fixBrokenCode",
            plugin_name="codePlugin",
            prompt_template_config=code_fix_config,
        )
    
    def _register_test_data_gen_function(self):
        """註冊測試數據生成功能"""
        
        # 測試數據生成提示模板
        test_data_gen_prompt = """
        我需要你為以下代碼生成測試數據，以便可以自動執行和測試。

        程式語言: {{$language}}
        
        代碼:
        ```
        {{$code}}
        ```
        
        請分析代碼並生成適當的測試數據。你的回應必須包含以下部分:
        
        1. 輸入分析: 分析代碼需要哪些輸入數據
        2. 測試數據: 提供2-3組有意義的測試數據
        3. 執行方法: 說明如何使用這些測試數據執行代碼
        4. 修改建議: 如果需要修改代碼來接受測試數據，提供修改後的代碼
        
        按照以下格式回覆：
        
        INPUT_ANALYSIS:
        [輸入數據分析]
        
        TEST_DATA:
        [測試數據詳情]
        
        EXECUTION_METHOD:
        [執行方法說明]
        
        MODIFIED_CODE:
        [修改後的代碼，如果需要]
        
        只返回上述格式，不要有其他多餘的解釋或 markdown 標記。
        """
        
        # 測試數據生成配置
        test_data_gen_config = PromptTemplateConfig(
            template=test_data_gen_prompt,
            name="generateTestData",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="language", description="程式語言", is_required=True),
                InputVariable(name="code", description="需要測試的代碼", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=3000,
                temperature=0.2,
            )
        )
        
        # 添加測試數據生成功能到 Kernel
        self.test_data_gen_function = self.kernel.add_function(
            function_name="generateTestData",
            plugin_name="codePlugin",
            prompt_template_config=test_data_gen_config,
        )
    
    async def process_message(self, message: str, sender: Optional[str] = None) -> str:
        """
        處理代碼相關請求
        
        Args:
            message: 用戶訊息
            sender: 訊息發送者
            
        Returns:
            回應訊息
        """
        # 確保代碼生成功能已註冊
        if self.code_gen_function is None and self.kernel is not None:
            self._register_code_gen_function()
            self._register_code_fix_function()
            self._register_test_data_gen_function()

        # 檢測是否是檔案生成模式
        is_file_generation_mode = False
        clean_message = message
        
        # 檢查特殊標記
        if "[FILE_GENERATION_MODE=True]" in message:
            is_file_generation_mode = True
            # 移除標記，獲取乾淨的消息內容
            clean_message = message.replace("[FILE_GENERATION_MODE=True]", "").strip()

         # 檔案生成模式：生成代碼並執行，但不顯示代碼
        if is_file_generation_mode:
            # 提取任務描述
            task = clean_message
            
            # 確定文件類型
            file_type = await self._detect_file_type_with_ai(task)
            
            # 生成檔案創建代碼
            code_result = await self.generate_file_creation_code(task, file_type)
            code = code_result.get("code", "")
            
            # 執行代碼並返回結果
            execution_result, fixed_code, fix_attempts, is_successful, fix_history = await self.execute_and_fix_code(code, "python", task)
            if fixed_code:
                code = fixed_code
            # 提取檔案路徑
            file_path = await self._extract_file_path_with_ai(execution_result)
            
            # 格式化響應，不包含代碼
            if file_path:
                response = f"✅ 已經為您生成檔案，保存在：`{file_path}`\n\n"
                # 檢查檔案是否存在
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    response += f"檔案大小: {self._format_file_size(file_size)}\n"
                    response += f"檔案類型: {os.path.splitext(file_path)[1]}\n\n"
                    response += "您可以從上述路徑獲取該檔案。"
                else:
                    response += "⚠️ 警告：檔案可能已生成，但無法在指定路徑找到。"
            else:
                # 如果沒有檔案路徑，顯示執行結果
                response = f"執行結果:\n\n{execution_result}"
            
            return response
        
        # 提取代碼任務 (移除前綴詞)
        task = message
        for prefix in ["請幫我寫代碼", "生成代碼", "寫一段程式", "代碼生成", "幫我寫", "實現", "開發"]:
            if prefix in message:
                task = message.split(prefix, 1)[1].strip()
                break
        
        try:
            # 生成智能代碼（包含語言、依賴和代碼）
            code_result = await self.generate_smart_code(task)
            language = code_result.get("language", "").lower()
            dependencies = code_result.get("dependencies", [])
            code = code_result.get("code", "")
            explanation = code_result.get("explanation", "")
            
            # 檢查環境
            env_checker = EnvironmentChecker()
            env_ready, env_message = await env_checker.check_environment(language)
            
            # 根據語言和環境狀態決定執行方式
            if env_ready:
                # 環境已準備好，可以嘗試執行
                
                # 檢查並安裝依賴（如果是Python）
                install_logs = ""
                if language == "python" and dependencies:
                    missing_deps = await self.check_dependencies(dependencies)
                    if missing_deps and self.allow_installs:
                        install_logs = await self.install_dependencies(missing_deps)
                
                # 檢查代碼是否需要輸入數據，如果需要則生成測試數據
                needs_input, test_data_result = await self.analyze_input_requirements(code, language)
                
                if needs_input:
                    # 代碼需要輸入，使用生成的測試數據
                    modified_code = test_data_result.get("modified_code", "")
                    if modified_code:
                        code = modified_code  # 使用修改後可自動執行的代碼
                
                # 執行代碼並自動修復錯誤
                execution_result, fixed_code, fix_attempts, is_successful, fix_history = await self.execute_and_fix_code(
                    code, language, task
                )
                
                # 更新最終代碼
                if fixed_code:
                    code = fixed_code
                
                # 構建響應
                response = f"# {language.capitalize()} 代碼\n\n"
                
                if dependencies:
                    response += f"## 依賴項\n\n"
                    response += "\n".join(dependencies) + "\n\n"
                    
                    if install_logs:
                        response += f"## 安裝日誌\n\n```\n{install_logs}\n```\n\n"
                
                response += f"## 代碼\n\n```{language}\n{code}\n```\n\n"
                
                # 添加測試數據分析（如果有）
                if needs_input:
                    response += f"## 輸入分析\n\n"
                    response += f"{test_data_result.get('input_analysis', '代碼需要輸入數據來執行')}\n\n"
                    
                    response += f"## 測試數據\n\n"
                    response += f"{test_data_result.get('test_data', '無法生成有效的測試數據')}\n\n"
                    
                    response += f"## 執行方法\n\n"
                    response += f"{test_data_result.get('execution_method', '無執行方法建議')}\n\n"
                
                # 添加詳細的修復過程視覺化
                if fix_attempts > 0:
                    response += f"## 自動修復過程 ({fix_attempts}次嘗試)\n\n"
                    response += f"最終執行狀態: {'✅ 成功' if is_successful else '❌ 仍有錯誤'}\n\n"
                    
                    # 顯示每次嘗試的詳細信息
                    for i, record in enumerate(fix_history):
                        if "attempt" in record:
                            if "has_error" in record:  # 這是執行記錄
                                attempt_num = record["attempt"]
                                response += f"### 嘗試 {attempt_num + 1} - 執行代碼\n\n"
                                
                                # 顯示代碼片段 (為了簡潔，只顯示前幾行)
                                code_preview = "\n".join(record["code"].split("\n")[:10])
                                if len(record["code"].split("\n")) > 10:
                                    code_preview += "\n# ... (省略部分代碼)"
                                    
                                response += f"```{language}\n{code_preview}\n```\n\n"
                                
                                # 顯示執行結果或錯誤
                                if record["has_error"]:
                                    response += f"#### ❌ 執行結果 (有錯誤)\n\n"
                                    error_text = record["execution_result"]
                                    # 提取關鍵錯誤信息
                                    if "執行代碼出錯" in error_text:
                                        error_lines = error_text.split("\n")
                                        for line in error_lines:
                                            if "執行代碼出錯" in line:
                                                response += f"`{line}`\n\n"
                                                break
                                    else:
                                        response += f"```\n{error_text[:200]}{'...' if len(error_text) > 200 else ''}\n```\n\n"
                                else:
                                    response += f"#### ✅ 執行成功\n\n"
                                    result_preview = record["execution_result"][:200]
                                    response += f"```\n{result_preview}{'...' if len(record['execution_result']) > 200 else ''}\n```\n\n"
                            
                            elif "error_analysis" in record:  # 這是修復嘗試記錄
                                response += f"### 修復分析與策略\n\n"
                                
                                # 顯示錯誤分析
                                response += "#### 錯誤分析\n\n"
                                response += f"{record['error_analysis']}\n\n"
                                
                                # 顯示修復策略
                                response += "#### 修復策略\n\n"
                                response += f"{record['fix_approach']}\n\n"
                                
                                # 顯示代碼變更 (如果有)
                                if record.get("status") == "代碼已修改":
                                    response += "#### 代碼修改\n\n"
                                    # 使用diff風格顯示變更
                                    response += "```diff\n"
                                    
                                    # 簡化的代碼diff顯示
                                    original_lines = record["original_code"].split("\n")
                                    fixed_lines = record["fixed_code"].split("\n")
                                    
                                    # 只展示前後有變化的幾行
                                    changes_found = False
                                    for i in range(min(len(original_lines), len(fixed_lines))):
                                        if original_lines[i] != fixed_lines[i]:
                                            response += f"- {original_lines[i]}\n+ {fixed_lines[i]}\n"
                                            changes_found = True
                                    
                                    # 處理長度不同的情況
                                    if len(original_lines) < len(fixed_lines):
                                        for i in range(len(original_lines), len(fixed_lines)):
                                            response += f"+ {fixed_lines[i]}\n"
                                            changes_found = True
                                    elif len(original_lines) > len(fixed_lines):
                                        for i in range(len(fixed_lines), len(original_lines)):
                                            response += f"- {original_lines[i]}\n"
                                            changes_found = True
                                    
                                    if not changes_found:
                                        response += "代碼結構發生變化，但無法顯示簡單的行差異。\n"
                                    
                                    response += "```\n\n"
                                else:
                                    response += f"#### 狀態: {record.get('status', '未知')}\n\n"
                
                response += f"## 最終執行結果\n\n{execution_result}\n\n"
                
                if explanation:
                    response += f"## 説明\n\n{explanation}\n"
                
                return response
            else:
                # 環境未準備好，只顯示代碼並提供安裝指南
                response = f"# {language.capitalize()} 代碼\n\n"
                
                if dependencies:
                    response += f"## 依賴項\n\n"
                    response += "\n".join(dependencies) + "\n\n"
                
                response += f"## 代碼\n\n```{language}\n{code}\n```\n\n"
                
                if explanation:
                    response += f"## 説明\n\n{explanation}\n"
                
                response += f"## 環境配置\n\n{env_message}\n\n"
                response += "請安裝所需環境後再執行此代碼。\n"
                
                return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return f"處理您的代碼請求時出錯:\n\n```\n{error_trace}\n```"
    
    async def analyze_input_requirements(self, code: str, language: str) -> Tuple[bool, Dict[str, Any]]:
        """
        分析代碼是否需要輸入數據，如需要則生成測試數據
        
        Args:
            code: 代碼
            language: 程式語言
            
        Returns:
            (是否需要輸入, 測試數據結果)
        """
        # 檢查是否包含輸入函數
        needs_input = False
        
        input_patterns = {
            "python": [r"input\s*\(", r"sys\.stdin", r"fileinput"],
            "javascript": [r"prompt\s*\(", r"readline", r"process\.stdin"],
            "java": [r"Scanner", r"System\.in", r"BufferedReader"],
            "c++": [r"cin\s*>>", r"getline", r"scanf"],
            "c#": [r"Console\.Read", r"Console\.ReadLine"]
        }
        
        # 獲取對應語言的輸入模式
        patterns = input_patterns.get(language.lower(), [])
        
        # 檢查代碼是否匹配任何輸入模式
        for pattern in patterns:
            if re.search(pattern, code):
                needs_input = True
                break
        
        # 如果需要輸入，生成測試數據
        if needs_input:
            try:
                test_data_result = await self.generate_test_data(code, language)
                return True, test_data_result
            except Exception as e:
                return True, {
                    "input_analysis": f"檢測到代碼需要輸入，但生成測試數據時出錯: {str(e)}",
                    "test_data": "無法生成測試數據",
                    "execution_method": "請手動提供輸入數據執行此代碼"
                }
        else:
            return False, {}
    
    async def generate_test_data(self, code: str, language: str) -> Dict[str, Any]:
        """
        為代碼生成測試數據
        
        Args:
            code: 代碼
            language: 程式語言
            
        Returns:
            測試數據結果
        """
        try:
            # 調用測試數據生成功能
            result = await self.kernel.invoke(
                self.test_data_gen_function,
                KernelArguments(
                    language=language,
                    code=code
                )
            )
            
            # 解析生成的結果
            response = str(result).strip()
            parsed_result = self._parse_test_data_response(response)
            return parsed_result
            
        except Exception as e:
            raise Exception(f"測試數據生成失敗: {str(e)}")
    
    def _parse_test_data_response(self, response: str) -> Dict[str, Any]:
        """
        解析 AI 生成的測試數據響應
        
        Args:
            response: AI 生成的響應
            
        Returns:
            包含解析後信息的字典
        """
        result = {
            "input_analysis": "",
            "test_data": "",
            "execution_method": "",
            "modified_code": ""
        }
        
        # 解析輸入分析
        if "INPUT_ANALYSIS:" in response:
            analysis_part = response.split("INPUT_ANALYSIS:", 1)[1]
            if "TEST_DATA:" in analysis_part:
                analysis_part = analysis_part.split("TEST_DATA:", 1)[0]
            result["input_analysis"] = analysis_part.strip()
        
        # 解析測試數據
        if "TEST_DATA:" in response:
            test_data_part = response.split("TEST_DATA:", 1)[1]
            if "EXECUTION_METHOD:" in test_data_part:
                test_data_part = test_data_part.split("EXECUTION_METHOD:", 1)[0]
            result["test_data"] = test_data_part.strip()
        
        # 解析執行方法
        if "EXECUTION_METHOD:" in response:
            method_part = response.split("EXECUTION_METHOD:", 1)[1]
            if "MODIFIED_CODE:" in method_part:
                method_part = method_part.split("MODIFIED_CODE:", 1)[0]
            result["execution_method"] = method_part.strip()
        
        # 解析修改後的代碼
        if "MODIFIED_CODE:" in response:
            code_part = response.split("MODIFIED_CODE:", 1)[1].strip()
            
            # 移除 markdown 格式標記
            code_text = code_part.strip()
            # 移除開頭的 ```語言名稱
            if code_text.startswith("```"):
                first_line_end = code_text.find("\n")
                if first_line_end != -1:
                    code_text = code_text[first_line_end+1:]
            
            # 移除結尾的 ```
            if code_text.endswith("```"):
                code_text = code_text[:-3].strip()
            
            result["modified_code"] = code_text
        
        return result
    
    async def _detect_file_type_with_ai(self, message: str) -> str:
        """使用 AI 從用戶消息中檢測所需的文件類型"""
        # 註冊文件類型判斷函數（如果尚未註冊）
        if not hasattr(self, "file_type_function"):
            from semantic_kernel.prompt_template import PromptTemplateConfig
            from semantic_kernel.prompt_template.input_variable import InputVariable
            from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
            
            prompt = """
            請從用戶的消息中確定他們想要生成的文件類型。

            文件類型可能包括：pdf, excel, xlsx, word, docx, csv, txt, ppt, pptx, json, html, 等。

            即使用戶沒有明確指定文件格式，也請根據他們的描述或需求推斷最合適的文件類型。
            如果消息中提到"表格"或數據分析，可能需要 excel 或 csv；
            如果是文本內容，可能需要 txt, word 或 pdf；
            如果是圖表或演示，可能需要 ppt。

            只返回判斷的文件類型，無需其他解釋。如果無法確定，請返回 "pdf"（作為默認選項）。

            用戶消息: {{$message}}
            """
            
            config = PromptTemplateConfig(
                template=prompt,
                name="detectFileType",
                template_format="semantic-kernel",
                input_variables=[
                    InputVariable(name="message", description="用戶消息", is_required=True),
                ],
                execution_settings=AzureChatPromptExecutionSettings(
                    service_id="default",
                    max_tokens=50,
                    temperature=0.0,  # 確定性輸出
                )
            )
            
            self.file_type_function = self.kernel.add_function(
                function_name="detectFileType",
                plugin_name="codePlugin",
                prompt_template_config=config,
            )
        
        try:
            # 調用 AI 判斷
            from semantic_kernel.functions import KernelArguments
            result = await self.kernel.invoke(
                self.file_type_function,
                KernelArguments(message=message)
            )
            
            file_type = str(result).strip().lower()
            
            # 確保返回的文件類型是有效的
            valid_types = ["pdf", "excel", "xlsx", "word", "docx", "csv", "txt", "ppt", "pptx", "json", "html"]
            if file_type in valid_types:
                return file_type
            
            # 處理一些常見的替代表達
            if file_type in ["excel表格", "表格", "電子表格"]:
                return "excel"
            elif file_type in ["文本", "純文本"]:
                return "txt"
            elif file_type in ["演示文稿", "簡報", "投影片"]:
                return "ppt"
            
            # 默認類型
            return "pdf"
        except Exception as e:
            print(f"AI文件類型檢測失敗: {str(e)}")
            # 失敗時使用默認類型
            return "pdf"

    async def _extract_file_path_with_ai(self, result_text: str) -> Optional[str]:
        """使用 AI 從執行結果中提取文件路徑"""
        # 註冊文件路徑提取函數（如果尚未註冊）
        if not hasattr(self, "file_path_function"):
            from semantic_kernel.prompt_template import PromptTemplateConfig
            from semantic_kernel.prompt_template.input_variable import InputVariable
            from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
            
            prompt = """
            請從以下執行結果中提取生成檔案的完整路徑。

            執行結果內容:
            {{$result_text}}

            檔案路徑通常出現在類似這樣的短語之後：
            - "檔案已成功生成並保存到:"
            - "保存在:"
            - "文件路徑:"
            - "已生成:"

            路徑可能是Windows格式(如 C:\\Users\\name\\downloads\\file.pdf)或
            Unix格式(如 /home/user/downloads/file.pdf)。

            請只返回完整的檔案路徑，不要包含任何其他文字或解釋。
            如果找不到路徑，請回答 "未找到檔案路徑"。
            """
            
            config = PromptTemplateConfig(
                template=prompt,
                name="extractFilePath",
                template_format="semantic-kernel",
                input_variables=[
                    InputVariable(name="result_text", description="執行結果文本", is_required=True),
                ],
                execution_settings=AzureChatPromptExecutionSettings(
                    service_id="default",
                    max_tokens=100,
                    temperature=0.0,  # 確定性輸出
                )
            )
            
            self.file_path_function = self.kernel.add_function(
                function_name="extractFilePath",
                plugin_name="codePlugin",
                prompt_template_config=config,
            )
        
        try:
            # 調用 AI 提取路徑
            from semantic_kernel.functions import KernelArguments
            result = await self.kernel.invoke(
                self.file_path_function,
                KernelArguments(result_text=result_text)
            )
            
            path = str(result).strip()
            
            # 檢查結果是否是有效路徑
            if path == "未找到檔案路徑":
                return None
                
            # 簡單驗證是否看起來像路徑
            if ('\\' in path or '/' in path) and '.' in path:
                return path
                
            return None
        except Exception as e:
            print(f"AI路徑提取失敗: {str(e)}")
            # 失敗時返回None
            return None
    
    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    async def generate_file_creation_code(self, task: str, file_type: str) -> Dict[str, Any]:
        """生成創建文件的代碼"""
        # 創建用於檔案生成的提示
        file_gen_prompt = f"""
        請生成一段Python代碼，用於創建一個{file_type}檔案，內容應該與以下任務相關：
        
        任務: {task}
        
        生成的代碼應該滿足以下要求：
        1. 使用適當的Python庫生成{file_type}檔案
        2. 將檔案保存在 'downloads' 目錄下（如果目錄不存在，應自動創建）
        3. 檔案名稱應該合理且有意義
        4. 代碼應處理所有可能的錯誤
        5. 代碼應該在執行結束時打印檔案的絕對路徑，格式為：'檔案已成功生成並保存到: [路徑]'
        
        如果任務需要數據，請生成合理的示例數據或使用適當的數據結構。
        """
        
        # 調用standard_code_gen來生成代碼
        code_gen_result = await self.generate_smart_code(file_gen_prompt)
        return code_gen_result
    
    async def execute_and_fix_code(self, code: str, language: str, original_task: str) -> Tuple[str, Optional[str], int, bool, List[Dict[str, Any]]]:
        """
        執行Python代碼並嘗試自動修復錯誤，記錄修復過程。非Python代碼僅返回提示不執行。
        
        Args:
            code: 原始代碼
            language: 程式語言
            original_task: 原始任務描述
            
        Returns:
            (執行結果, 修復後的代碼(如果有), 修復嘗試次數, 是否成功, 修復過程記錄)
        """
        # 記錄每次修復的詳細信息
        fix_history = []
        
        # 非 Python 代碼不執行
        if language != "python":
            message = f"已生成 {language} 代碼。目前系統僅支持執行 Python 代碼。"
            return message, None, 0, True, fix_history
        
        # 執行 Python 代碼
        current_code = code
        fix_attempts = 0
        is_successful = False
        execution_result = None
        
        # 嘗試執行代碼，如有錯誤則修復
        for attempt in range(self.max_fix_attempts + 1):
            # 執行當前版本的代碼
            execution_result = await self._execute_python(current_code)
            
            # 記錄此次執行結果
            execution_record = {
                "attempt": attempt,
                "code": current_code,
                "execution_result": execution_result,
                "has_error": "執行代碼出錯" in execution_result or "Error" in execution_result
            }
            
            # 檢查是否有錯誤
            if not execution_record["has_error"]:
                # 代碼執行成功
                is_successful = True
                fix_history.append(execution_record)
                break
            
            # 到達最大嘗試次數
            if fix_attempts >= self.max_fix_attempts:
                fix_history.append(execution_record)
                break
            
            # 嘗試修復代碼
            try:
                # 記錄修復嘗試
                fix_record = {
                    "attempt": attempt,
                    "error_message": execution_result
                }
                
                # 獲取修復方案
                fixed_code_result = await self.fix_code(current_code, language, original_task, execution_result)
                new_code = fixed_code_result.get("fixed_code")
                
                # 更新修復記錄
                fix_record.update({
                    "error_analysis": fixed_code_result.get("error_analysis", ""),
                    "fix_approach": fixed_code_result.get("fix_approach", ""),
                    "original_code": current_code,
                    "fixed_code": new_code if new_code else current_code
                })
                
                # 檢查修復的代碼是否與當前代碼相同
                if new_code and new_code != current_code:
                    current_code = new_code
                    fix_attempts += 1
                    fix_record["status"] = "代碼已修改"
                else:
                    # 代碼沒有變化，停止嘗試
                    fix_record["status"] = "無法修復"
                    fix_history.append(execution_record)
                    fix_history.append(fix_record)
                    break
                
                # 保存修復記錄
                fix_history.append(execution_record)
                fix_history.append(fix_record)
                
            except Exception as e:
                # 修復過程出錯，停止嘗試
                execution_result += f"\n\n自動修復過程中出錯: {str(e)}"
                execution_record["fix_error"] = str(e)
                fix_history.append(execution_record)
                break
        
        return execution_result, current_code if fix_attempts > 0 else None, fix_attempts, is_successful, fix_history
    
    async def generate_smart_code(self, task: str) -> Dict[str, Any]:
        """
        生成智能代碼，包含程式語言、依賴和代碼
        
        Args:
            task: 任務描述
            
        Returns:
            包含語言、依賴和代碼的字典
        """
        try:
            result = await self.kernel.invoke(
                self.code_gen_function,
                KernelArguments(task=task)
            )
            
            # 解析生成的結果
            response = str(result).strip()
            parsed_result = self._parse_code_response(response)
            return parsed_result
            
        except Exception as e:
            raise Exception(f"代碼生成失敗: {str(e)}")
    
    def _parse_code_response(self, response: str) -> Dict[str, Any]:
        """
        解析 AI 生成的代碼響應
        
        Args:
            response: AI 生成的響應
            
        Returns:
            包含解析後信息的字典
        """
        result = {
            "language": "",
            "dependencies": [],
            "code": "",
            "explanation": ""
        }
        
        # 解析程式語言
        if "LANGUAGE:" in response:
            language_part = response.split("LANGUAGE:", 1)[1].split("\n", 1)[0].strip()
            result["language"] = language_part
        
        # 解析依賴項
        if "DEPENDENCIES:" in response:
            deps_part = response.split("DEPENDENCIES:", 1)[1]
            if "CODE:" in deps_part:
                deps_part = deps_part.split("CODE:", 1)[0]
            deps_list = [d.strip() for d in deps_part.strip().split("\n") if d.strip()]
            result["dependencies"] = deps_list
        
        # 解析代碼
        if "CODE:" in response:
            code_part = response.split("CODE:", 1)[1]
            if "EXPLANATION:" in code_part:
                code_part = code_part.split("EXPLANATION:", 1)[0]
            
            # 移除 markdown 格式標記
            code_text = code_part.strip()
            # 移除開頭的 ```語言名稱
            if code_text.startswith("```"):
                first_line_end = code_text.find("\n")
                if first_line_end != -1:
                    code_text = code_text[first_line_end+1:]
            
            # 移除結尾的 ```
            if code_text.endswith("```"):
                code_text = code_text[:-3].strip()
            
            result["code"] = code_text
        
        # 解析説明
        if "EXPLANATION:" in response:
            explanation_part = response.split("EXPLANATION:", 1)[1].strip()
            result["explanation"] = explanation_part
        
        return result
    
    async def fix_code(self, code: str, language: str, original_task: str, error_message: str) -> Dict[str, Any]:
        """
        嘗試修復代碼
        
        Args:
            code: 需要修復的代碼
            language: 程式語言
            original_task: 原始任務描述
            error_message: 錯誤訊息
            
        Returns:
            包含修復分析和修復後代碼的字典
        """
        try:
            # 調用代碼修復功能
            result = await self.kernel.invoke(
                self.code_fix_function,
                KernelArguments(
                    original_task=original_task,
                    language=language,
                    code=code,
                    error_message=error_message
                )
            )
            
            # 解析生成的結果
            response = str(result).strip()
            parsed_result = self._parse_fix_response(response)
            return parsed_result
            
        except Exception as e:
            raise Exception(f"代碼修復失敗: {str(e)}")
    
    def _parse_fix_response(self, response: str) -> Dict[str, Any]:
        """
        解析 AI 生成的代碼修復響應
        
        Args:
            response: AI 生成的響應
            
        Returns:
            包含解析後信息的字典
        """
        result = {
            "error_analysis": "",
            "fix_approach": "",
            "fixed_code": ""
        }
        
        # 解析錯誤分析
        if "ERROR_ANALYSIS:" in response:
            analysis_part = response.split("ERROR_ANALYSIS:", 1)[1]
            if "FIX_APPROACH:" in analysis_part:
                analysis_part = analysis_part.split("FIX_APPROACH:", 1)[0]
            result["error_analysis"] = analysis_part.strip()
        
        # 解析修復方案
        if "FIX_APPROACH:" in response:
            approach_part = response.split("FIX_APPROACH:", 1)[1]
            if "FIXED_CODE:" in approach_part:
                approach_part = approach_part.split("FIXED_CODE:", 1)[0]
            result["fix_approach"] = approach_part.strip()
        
        # 解析修復後的代碼
        if "FIXED_CODE:" in response:
            code_part = response.split("FIXED_CODE:", 1)[1].strip()
            
            # 移除 markdown 格式標記
            code_text = code_part.strip()
            # 移除開頭的 ```語言名稱
            if code_text.startswith("```"):
                first_line_end = code_text.find("\n")
                if first_line_end != -1:
                    code_text = code_text[first_line_end+1:]
            
            # 移除結尾的 ```
            if code_text.endswith("```"):
                code_text = code_text[:-3].strip()
            
            result["fixed_code"] = code_text
        
        return result
    
    async def check_dependencies(self, dependencies: List[str]) -> List[str]:
        """
        檢查缺少的依賴項
        
        Args:
            dependencies: 依賴項列表
            
        Returns:
            缺少的依賴項列表
        """
        missing_deps = []
        
        for dep in dependencies:
            # 移除版本信息
            clean_dep = dep.split('==')[0].split('>=')[0].split('<=')[0].strip()
            
            if not clean_dep:
                continue
                
            try:
                importlib.import_module(clean_dep)
            except ImportError:
                missing_deps.append(dep)
        
        return missing_deps
    
    async def install_dependencies(self, dependencies: List[str]) -> str:
        """
        安裝缺少的依賴項
        
        Args:
            dependencies: 要安裝的依賴項列表
            
        Returns:
            安裝日誌
        """
        if not dependencies:
            return "沒有需要安裝的依賴項。"
            
        try:
            # 使用 pip 安裝依賴
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install"] + dependencies,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                return f"安裝依賴項時出錯:\n{stderr}"
            
            return f"安裝成功:\n{stdout}"
        except Exception as e:
            return f"安裝過程中出現錯誤: {str(e)}"

    def _remove_markdown_format(self, code: str) -> str:
        """移除 markdown 格式"""
        # 移除開頭的 ```語言名稱
        if code.startswith("```"):
            first_line_end = code.find("\n")
            if first_line_end != -1:
                code = code[first_line_end+1:]
        
        # 移除結尾的 ```
        if code.endswith("```"):
            code = code[:-3].strip()
        
        return code.strip()

    async def _execute_python(self, code: str) -> str:
        """
        執行 Python 代碼
        
        Args:
            code: 要執行的代碼
            
        Returns:
            執行結果
        """
        
        # 創建捕獲輸出的緩衝區
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        redirected_output = StringIO()
        redirected_error = StringIO()
        
        sys.stdout = redirected_output
        sys.stderr = redirected_error
        
        # 對代碼進行預處理，修復常見格式問題
        code = self._clean_code(code)
        
        # 創建初始化的執行環境
        exec_globals = {
            "os": os,
            "sys": sys,
            "Path": Path,
            "result": None
        }
        
        # 動態導入代碼中提到的模組
        imported_modules = self._extract_imports(code)
        for module_name in imported_modules:
            try:
                # 只導入代碼中明確 import 的模組
                module = importlib.import_module(module_name)
                exec_globals[module_name] = module
            except ImportError:
                # 如果模組無法導入，在執行時可能會引發錯誤
                pass
        
        # 清除可能存在的 __file__ 變數以避免意外存取
        if "__file__" in exec_globals:
            del exec_globals["__file__"]
        
        try:
            # 執行代碼
            exec(code, exec_globals)
            
            # 收集輸出
            stdout_output = redirected_output.getvalue()
            stderr_output = redirected_error.getvalue()
            
            # 檢查結果變數
            result_output = ""
            if "result" in exec_globals and exec_globals["result"] is not None:
                result_output = f"結果變數:\n{exec_globals['result']}\n\n"
            
            # 合併輸出
            output = result_output
            
            if stdout_output:
                output += f"標準輸出:\n{stdout_output}\n"
            
            if stderr_output:
                output += f"錯誤輸出:\n{stderr_output}\n"
            
            # 如果沒有任何輸出，提供一個默認訊息
            if not output.strip():
                output = "代碼執行成功，但沒有產生輸出。"
            
            return output
        except SyntaxError as e:
            # 特別處理語法錯誤
            line_number = e.lineno if hasattr(e, 'lineno') else 0
            offset = e.offset if hasattr(e, 'offset') else 0
            error_text = e.text if hasattr(e, 'text') else ""
            
            # 提供更詳細的錯誤信息和上下文
            code_lines = code.splitlines()
            start_line = max(0, line_number - 3)
            end_line = min(len(code_lines), line_number + 2)
            
            context = "\n代碼上下文:\n"
            for i in range(start_line, end_line):
                if i < len(code_lines):
                    prefix = ">> " if i == line_number - 1 else "   "
                    context += f"{prefix}第 {i+1} 行: {code_lines[i]}\n"
            
            return f"執行代碼出錯: SyntaxError 在第 {line_number} 行: {str(e)}\n{context}"
        except Exception as e:
            error_class = e.__class__.__name__
            detail = str(e)
            cl, exc, tb = sys.exc_info()
            
            try:
                tb_info = traceback.extract_tb(tb)
                _, line_number, _, _ = tb_info[-1]
            except:
                line_number = "未知"
            
            return f"執行代碼出錯: {error_class} 在第 {line_number} 行: {detail}"
        finally:
            # 恢復標準輸出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _clean_code(self, code: str) -> str:
        """
        清理和格式化代碼，修復常見格式問題
        
        Args:
            code: 原始代碼
            
        Returns:
            清理後的代碼
        """
        # 去除代碼中的特殊字符
        code = code.replace('\xa0', ' ')  # 替換不間斷空格
        
        # 確保行結束為標準換行符
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # 修正可能的縮進問題
        lines = code.split('\n')
        result_lines = []
        for line in lines:
            # 替換 Tab 為 4 個空格
            line = line.replace('\t', '    ')
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _extract_imports(self, code: str) -> List[str]:
        """
        從 Python 代碼中提取導入的模組名稱
        
        Args:
            code: Python 代碼
            
        Returns:
            導入的模組名稱列表
        """
        import re
        
        # 找出所有 import 語句
        import_pattern = r'^\s*import\s+(\w+)(?:\s*,\s*(\w+))*'
        from_import_pattern = r'^\s*from\s+(\w+)(?:\.\w+)*\s+import'
        
        modules = set()
        
        # 處理 import 語句
        for line in code.split('\n'):
            import_match = re.match(import_pattern, line)
            if import_match:
                # 處理 "import x, y, z" 形式
                for group in import_match.groups():
                    if group:
                        modules.add(group)
            
            # 處理 from x import y 形式
            from_match = re.match(from_import_pattern, line)
            if from_match:
                base_module = from_match.group(1)
                if base_module:
                    modules.add(base_module)
        
        # 添加常見的基礎庫別名
        aliases = {
            'pd': 'pandas',
            'np': 'numpy',
            'plt': 'matplotlib.pyplot',
            'sns': 'seaborn',
            'tf': 'tensorflow',
            'torch': 'torch'
        }
        
        # 檢查別名使用
        for alias, module in aliases.items():
            if alias in code and alias not in modules:
                base_module = module.split('.')[0]
                modules.add(base_module)
        
        return list(modules)