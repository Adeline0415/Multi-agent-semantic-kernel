import os
import sys
import asyncio
import traceback
import subprocess
import importlib
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
    """處理代碼生成和執行的智能代理，支持多種程式語言和依賴管理"""
    
    def __init__(self, name: str = "CodeAgent"):
        """
        初始化代碼代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name, skills=["代碼生成", "代碼執行", "代碼解釋", "代碼除錯", "多語言支持", "依賴管理"])
        self.code_gen_function = None
        self.supported_languages = ["python", "javascript", "java", "c++", "c#", "go", "ruby", "php", "rust", "typescript", "bash", "r", "sql"]
        self.allow_installs = True  # 是否允許安裝新的依賴
    
    def setup_kernel(self, kernel: Kernel):
        """
        設置 Semantic Kernel 並註冊代碼生成功能
        
        Args:
            kernel: Semantic Kernel 實例
        """
        super().setup_kernel(kernel)
        self._register_code_gen_function()
    
    def _register_code_gen_function(self):
        """註冊代碼生成功能"""
        
        # 代碼生成提示模板 - 增強版
        code_gen_prompt = """
        請根據以下任務生成可執行的程式碼。
        
        任務: {{$task}}
        
        你可以使用任何適合的程式語言來實現此任務。如果任務中沒有指定程式語言，請選擇最適合的語言。
        
        你的回應必須包含以下部分:
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
            
            # 根據語言決定執行方式
            if language == "python":
                # 檢查並安裝依賴
                missing_deps = await self.check_dependencies(dependencies)
                install_logs = ""
                
                if missing_deps and self.allow_installs:
                    install_logs = await self.install_dependencies(missing_deps)
                
                # 執行 Python 代碼
                execution_result = await self._execute_python(code)
                
                # 構建響應
                response = f"# {language.capitalize()} 代碼\n\n"
                
                if missing_deps:
                    response += f"## 缺少的依賴項\n\n"
                    response += ", ".join(missing_deps) + "\n\n"
                    
                    if install_logs:
                        response += f"## 安裝日誌\n\n```\n{install_logs}\n```\n\n"
                
                response += f"## 代碼\n\n```{language}\n{code}\n```\n\n"
                response += f"## 執行結果\n\n{execution_result}\n\n"
                
                if explanation:
                    response += f"## 説明\n\n{explanation}\n"
                
                return response
            else:
                # 對於非 Python 代碼，只返回代碼和説明
                response = f"# {language.capitalize()} 代碼\n\n"
                
                if dependencies:
                    response += f"## 依賴項\n\n"
                    response += "\n".join(dependencies) + "\n\n"
                
                response += f"## 代碼\n\n```{language}\n{code}\n```\n\n"
                
                if explanation:
                    response += f"## 説明\n\n{explanation}\n"
                
                response += "\n**注意**: 目前只支持 Python 代碼的直接執行。其他語言的代碼需要您手動執行。\n"
                return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return f"處理您的代碼請求時出錯:\n\n```\n{error_trace}\n```"
    
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
    
    async def execute_code(self, code: str, language: str) -> str:
        """
        執行各種程式語言的代碼，先檢查環境
        
        Args:
            code: 要執行的代碼
            language: 程式語言
            
        Returns:
            執行結果
        """
        # 標準化語言名稱
        language = language.lower().strip()
        
        # 移除可能的 markdown 格式標記
        code = self._remove_markdown_format(code)
        
        # 檢查執行環境
        env_checker = EnvironmentChecker()
        env_ready, env_message = await env_checker.check_environment(language)
        
        if not env_ready:
            return f"無法執行 {language} 代碼: 環境未準備好。\n\n{env_message}\n\n請安裝所需環境後再試。"
        
        # 根據不同語言選擇執行方法
        if language in ["python", "py"]:
            return await self._execute_python(code)
        elif language in ["javascript", "js", "node"]:
            return await self._execute_javascript(code)
        elif language in ["java"]:
            return await self._execute_java(code)
        elif language in ["c#", "csharp", "cs"]:
            return await self._execute_csharp(code)
        elif language in ["c", "c++"]:
            return await self._execute_cpp(code)
        elif language in ["php"]:
            return await self._execute_php(code)
        elif language in ["ruby", "rb"]:
            return await self._execute_ruby(code)
        elif language in ["r"]:
            return await self._execute_r(code)
        else:
            return f"不支持執行 {language} 語言。代碼已生成，但需要您在適當的環境中手動運行。"

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

    async def _execute_javascript(self, code: str) -> str:
        """執行 JavaScript 代碼"""
        try:
            # 使用 Node.js 執行 JavaScript
            # 先創建臨時文件
            temp_file = Path("temp_script.js")
            temp_file.write_text(code, encoding="utf-8")
            
            # 執行代碼
            process = await asyncio.create_subprocess_exec(
                "node", str(temp_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # 刪除臨時文件
            temp_file.unlink()
            
            # 處理輸出
            result = ""
            if stdout:
                result += f"標準輸出:\n{stdout.decode('utf-8')}\n"
            if stderr:
                result += f"錯誤輸出:\n{stderr.decode('utf-8')}\n"
            
            return result or "代碼執行成功，但沒有產生輸出。"
        except Exception as e:
            return f"執行 JavaScript 時出錯: {str(e)}"

    async def _handle_html(self, code: str) -> str:
        """處理 HTML/CSS 代碼"""
        try:
            # 為 HTML 創建一個臨時文件
            temp_file = Path("temp_page.html")
            temp_file.write_text(code, encoding="utf-8")
            
            return f"HTML 代碼已保存到 {temp_file.absolute()}。請在瀏覽器中打開此文件查看結果。"
        except Exception as e:
            return f"處理 HTML 時出錯: {str(e)}"

    async def _execute_java(self, code: str) -> str:
        """執行 Java 代碼"""
        try:
            # 從代碼中提取類名
            import re
            class_match = re.search(r'public\s+class\s+(\w+)', code)
            if not class_match:
                return "無法找到 Java 主類名稱。請確保代碼包含 'public class ClassName' 聲明。"
            
            class_name = class_match.group(1)
            
            # 創建臨時 Java 文件
            java_file = Path(f"{class_name}.java")
            java_file.write_text(code, encoding="utf-8")
            
            # 編譯 Java 代碼
            compile_process = await asyncio.create_subprocess_exec(
                "javac", str(java_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await compile_process.communicate()
            
            if compile_process.returncode != 0:
                return f"Java 編譯錯誤:\n{stderr.decode('utf-8')}"
            
            # 執行 Java 程序
            run_process = await asyncio.create_subprocess_exec(
                "java", class_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await run_process.communicate()
            
            # 清理臨時文件
            java_file.unlink(missing_ok=True)
            Path(f"{class_name}.class").unlink(missing_ok=True)
            
            # 處理輸出
            result = ""
            if stdout:
                result += f"標準輸出:\n{stdout.decode('utf-8')}\n"
            if stderr:
                result += f"錯誤輸出:\n{stderr.decode('utf-8')}\n"
            
            return result or "代碼執行成功，但沒有產生輸出。"
        except Exception as e:
            return f"執行 Java 時出錯: {str(e)}"
    
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