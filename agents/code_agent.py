import os
import sys
import asyncio
import traceback
from typing import Dict, List, Any, Optional
from io import StringIO
import pandas as pd
from pathlib import Path

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from .base_agent import Agent

class CodeAgent(Agent):
    """處理代碼生成和執行的代理"""
    
    def __init__(self, name: str = "CodeAgent"):
        """
        初始化代碼代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name, skills=["代碼生成", "代碼執行", "代碼解釋", "代碼除錯"])
        self.code_gen_function = None
    
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
        
        # 代碼生成提示模板
        code_gen_prompt = """
        請根據以下任務生成可執行的 Python 代碼。

        任務: {{$task}}

        生成的代碼應該:
        1. 處理所有可能的錯誤情況
        2. 包含必要的註釋
        3. 將最終結果存儲在 'result' 變數中
        4. 只使用標準庫和以下允許的庫: pandas, numpy, pathlib, os, sys, datetime, json, matplotlib

        僅返回代碼，不要包含任何解釋或 markdown 標記。
        """
        
        # 代碼生成配置
        code_gen_config = PromptTemplateConfig(
            template=code_gen_prompt,
            name="generateCode",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="task", description="需要通過代碼實現的任務", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=2000,
                temperature=0.2,
            )
        )
        
        # 添加代碼生成功能到 Kernel
        self.code_gen_function = self.kernel.add_function(
            function_name="generateCode",
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
        for prefix in ["請幫我寫代碼", "生成代碼", "寫一段程式", "代碼生成"]:
            if prefix in message:
                task = message.split(prefix, 1)[1].strip()
                break
        
        try:
            # 生成代碼
            code = await self.generate_code(task)
            
            # 執行代碼
            result = await self.execute_code(code)
            
            return f"根據您的請求，我生成了以下代碼：\n\n```python\n{code}\n```\n\n執行結果：\n\n{result}"
        except Exception as e:
            return f"處理您的代碼請求時出錯: {str(e)}"
    
    async def generate_code(self, task: str) -> str:
        """
        生成 Python 代碼
        
        Args:
            task: 任務描述
            
        Returns:
            生成的代碼
        """
        try:
            code = await self.kernel.invoke(
                self.code_gen_function,
                KernelArguments(task=task)
            )
            return str(code).strip()
        except Exception as e:
            raise Exception(f"代碼生成失敗: {str(e)}")
    
    async def execute_code(self, code: str) -> str:
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
        
        # 創建安全執行環境
        exec_globals = {
            "os": os,
            "sys": sys,
            "pandas": pd,
            "pd": pd,
            "Path": Path,
            "result": None
        }
        
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
            
            return output
        except Exception as e:
            error_class = e.__class__.__name__
            detail = e.args[0]
            cl, exc, tb = sys.exc_info()
            line_number = traceback.extract_tb(tb)[-1][1]
            return f"執行代碼出錯: {error_class} 在第 {line_number} 行: {detail}"
        finally:
            # 恢復標準輸出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    async def generate_and_execute_code(self, task: str) -> str:
        """
        生成並執行 Python 代碼
        
        Args:
            task: 任務描述
            
        Returns:
            代碼和執行結果
        """
        # 生成代碼
        code = await self.generate_code(task)
        
        # 執行代碼
        result = await self.execute_code(code)
        
        return f"生成的代碼:\n```python\n{code}\n```\n\n執行結果:\n{result}"