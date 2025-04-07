# code_executor.py
import os
import sys
import traceback
from io import StringIO
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class CodeExecutor:
    """執行 Python 代碼並返回結果"""
    
    def __init__(self):
        """初始化代碼執行器"""
        # 允許載入的模組列表
        self.allowed_modules = {
            "os": os,
            "sys": sys,
            "pandas": pd,
            "pd": pd,
            "Path": Path,
            "pathlib": __import__("pathlib"),
            "datetime": __import__("datetime"),
            "json": __import__("json"),
            "re": __import__("re"),
            "math": __import__("math"),
            "random": __import__("random"),
            "time": __import__("time"),
            "collections": __import__("collections"),
            "csv": __import__("csv"),
            "io": __import__("io"),
            "StringIO": StringIO,
            "numpy": __import__("numpy"),
            "matplotlib": __import__("matplotlib")
        }
    
    def execute_code(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """
        執行 Python 代碼
        
        Args:
            code: 要執行的代碼
            
        Returns:
            包含執行輸出和結果變數的元組
        """
        # 創建捕獲輸出的緩衝區
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        redirected_output = StringIO()
        redirected_error = StringIO()
        
        sys.stdout = redirected_output
        sys.stderr = redirected_error
        
        # 創建執行環境
        exec_globals = self.allowed_modules.copy()
        exec_locals = {"result": None}
        
        # 清除可能存在的 __file__ 變數以避免意外存取
        if "__file__" in exec_globals:
            del exec_globals["__file__"]
        
        try:
            # 執行代碼
            exec(code, exec_globals, exec_locals)
            
            # 收集輸出
            stdout_output = redirected_output.getvalue()
            stderr_output = redirected_error.getvalue()
            
            # 準備結果
            output = ""
            if stdout_output:
                output += f"標準輸出:\n{stdout_output}\n"
            
            if stderr_output:
                output += f"錯誤輸出:\n{stderr_output}\n"
            
            return output, exec_locals
        except Exception as e:
            error_class = e.__class__.__name__
            detail = e.args[0]
            cl, exc, tb = sys.exc_info()
            line_number = traceback.extract_tb(tb)[-1][1]
            error_msg = f"執行代碼出錯: {error_class} 在第 {line_number} 行: {detail}"
            return error_msg, {"result": None}
        finally:
            # 恢復標準輸出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def execute_code_block(self, code: str) -> str:
        """
        執行代碼塊並返回格式化結果
        
        Args:
            code: 要執行的代碼
            
        Returns:
            格式化的執行結果
        """
        output, locals_dict = self.execute_code(code)
        
        # 檢查結果變數
        result_output = ""
        if "result" in locals_dict and locals_dict["result"] is not None:
            result_output = f"結果變數:\n{locals_dict['result']}\n\n"
        
        return result_output + output
    
    def is_code_safe(self, code: str) -> Tuple[bool, str]:
        """
        檢查代碼是否安全
        
        Args:
            code: 要檢查的代碼
            
        Returns:
            包含安全標誌和原因的元組
        """
        # 危險函數列表
        dangerous_functions = [
            "eval(", "exec(",
            "os.system(", "subprocess", "os.popen(",
            "__import__", "importlib",
            "open(", "read(", "write(",
            "globals(", "locals("
        ]
        
        # 檢查危險函數
        for func in dangerous_functions:
            if func in code:
                return False, f"代碼包含不安全的函數: {func}"
        
        return True, "代碼安全"