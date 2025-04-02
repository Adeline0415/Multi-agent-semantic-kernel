import os
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import PyPDF2
import docx
import pandas as pd

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from .base_agent import Agent
from utils.document_processor import DocumentProcessor

class DocumentAgent(Agent):
    """處理文檔分析和問答的代理"""
    
    def __init__(self, name: str = "DocumentAgent"):
        """
        初始化文檔代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name, skills=["文檔載入", "文檔分析", "文檔摘要", "文檔問答"])
        self.documents = {}  # 儲存已載入的文檔
        self.document_processor = DocumentProcessor()  # 文檔處理器
        self.summarize_function = None
        self.qa_function = None
    
    def setup_kernel(self, kernel: Kernel):
        """
        設置 Semantic Kernel 並註冊文檔相關功能
        
        Args:
            kernel: Semantic Kernel 實例
        """
        super().setup_kernel(kernel)
        self._register_document_functions()
    
    def _register_document_functions(self):
        """註冊文檔相關功能"""
        
        # 文檔摘要功能
        summarize_prompt = """
        請為以下文檔提供全面的摘要:
        
        {{$input}}
        
        您的摘要應該:
        1. 捕捉主要思想和關鍵點
        2. 結構良好且連貫
        3. 保留重要細節和上下文
        4. 長度約為原文的15-20%
        """
        
        summarize_config = PromptTemplateConfig(
            template=summarize_prompt,
            name="summarize",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="input", description="要總結的文檔", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=2000,
                temperature=0.3,  # 較低的溫度，以獲得更準確的摘要
            )
        )
        
        self.summarize_function = self.kernel.add_function(
            function_name="summarize",
            plugin_name="documentPlugin",
            prompt_template_config=summarize_config,
        )
        
        # 文檔問答功能
        qa_prompt = """
        請僅根據以下文檔中提供的信息回答問題。
        如果在文檔中找不到答案，請說"我沒有足夠的信息來回答這個問題。"
        
        文檔:
        {{$document}}
        
        問題: {{$question}}
        
        回答:
        """
        
        qa_config = PromptTemplateConfig(
            template=qa_prompt,
            name="documentQA",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="document", description="文檔內容", is_required=True),
                InputVariable(name="question", description="要回答的問題", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=1000,
                temperature=0.2,  # 低溫度，以獲得準確的答案
            )
        )
        
        self.qa_function = self.kernel.add_function(
            function_name="documentQA",
            plugin_name="documentPlugin",
            prompt_template_config=qa_config,
        )
    
    async def process_message(self, message: str, sender: Optional[str] = None) -> str:
        """
        處理文檔相關請求
        
        Args:
            message: 用戶訊息
            sender: 訊息發送者
            
        Returns:
            回應訊息
        """
        # 確保文檔功能已註冊
        if (self.summarize_function is None or self.qa_function is None) and self.kernel is not None:
            self._register_document_functions()
        
        try:
            # 檢查請求類型
            message_lower = message.lower()
            
            # 載入文檔請求
            if "載入" in message_lower or "加載" in message_lower or "讀取" in message_lower or "load" in message_lower:
                # 注意：實際實現中，文件載入需要與 Streamlit 上傳機制整合
                # 這裡是 API 示例，實際應用會有所不同
                return "請使用上傳功能來載入文檔。"
            
            # 列出文檔請求
            elif "列出文檔" in message_lower or "list documents" in message_lower:
                return self._list_documents()
            
            # 文檔摘要請求
            elif "摘要" in message_lower or "總結" in message_lower or "summarize" in message_lower:
                doc_name = self._extract_document_name(message)
                if doc_name:
                    return await self.summarize_document(doc_name)
                else:
                    return "請指定要摘要的文檔名稱。"
            
            # 假設其他是文檔問答請求
            else:
                # 嘗試找出相關文檔
                doc_name = self._extract_document_name(message)
                
                if doc_name and doc_name in self.documents:
                    # 從特定文檔回答問題
                    return await self.answer_from_document(doc_name, message)
                elif len(self.documents) == 1:
                    # 如果只有一個文檔，使用它
                    doc_name = list(self.documents.keys())[0]
                    return await self.answer_from_document(doc_name, message)
                elif len(self.documents) > 1:
                    # 如果有多個文檔但未指定
                    return "您有多份文檔。請指定要使用的文檔名稱，或上傳新的文檔。"
                else:
                    return "沒有載入任何文檔。請先上傳文檔。"
        
        except Exception as e:
            return f"處理文檔請求時出錯: {str(e)}"
    
    def load_document(self, file_path: str, document_name: Optional[str] = None) -> str:
        """
        從文件路徑加載文檔
        
        Args:
            file_path: 文件路徑
            document_name: 文檔名稱 (可選)
            
        Returns:
            載入結果訊息
        """
        if not os.path.exists(file_path):
            return f"錯誤: 在 {file_path} 找不到文件"
        
        # 從文檔中提取文本
        try:
            text = self.document_processor.extract_text(file_path)
            
            # 用名稱或文件名保存文檔
            if document_name is None:
                document_name = os.path.basename(file_path)
            
            self.documents[document_name] = text
            
            return f"文檔 '{document_name}' 已成功加載。({len(text)} 字符)"
        except Exception as e:
            return f"載入文檔時出錯: {str(e)}"
    
    def get_document_names(self) -> List[str]:
        """
        獲取所有已加載文檔的名稱
        
        Returns:
            文檔名稱列表
        """
        return list(self.documents.keys())
    
    def get_document_content(self, document_name: str) -> Optional[str]:
        """
        通過名稱獲取特定文檔的內容
        
        Args:
            document_name: 文檔名稱
            
        Returns:
            文檔內容，如果不存在則返回 None
        """
        return self.documents.get(document_name)
    
    async def summarize_document(self, document_name_or_text: str) -> str:
        """
        生成文檔摘要
        
        Args:
            document_name_or_text: 文檔名稱或直接文本內容
            
        Returns:
            文檔摘要
        """
        # 檢查輸入是否為文檔名稱
        if document_name_or_text in self.documents:
            document_text = self.documents[document_name_or_text]
        else:
            document_text = document_name_or_text
        
        try:
            summary = await self.kernel.invoke(
                self.summarize_function,
                KernelArguments(input=document_text)
            )
            return str(summary)
        except Exception as e:
            return f"生成摘要時出錯: {str(e)}"
    
    async def answer_from_document(self, document_name_or_text: str, question: str) -> str:
        """
        根據文檔內容回答問題
        
        Args:
            document_name_or_text: 文檔名稱或直接文本內容
            question: 問題
            
        Returns:
            回答
        """
        # 檢查輸入是否為文檔名稱
        if document_name_or_text in self.documents:
            document_text = self.documents[document_name_or_text]
        else:
            document_text = document_name_or_text
        
        try:
            answer = await self.kernel.invoke(
                self.qa_function,
                KernelArguments(document=document_text, question=question)
            )
            return str(answer)
        except Exception as e:
            return f"回答問題時出錯: {str(e)}"
    
    def _extract_document_name(self, message: str) -> Optional[str]:
        """
        從訊息中提取文檔名稱
        
        Args:
            message: 用戶訊息
            
        Returns:
            提取的文檔名稱，如果找不到則返回 None
        """
        # 遍歷所有已載入的文檔名稱
        for doc_name in self.documents.keys():
            if doc_name.lower() in message.lower():
                return doc_name
        return None
    
    def _list_documents(self) -> str:
        """
        列出所有已載入的文檔
        
        Returns:
            文檔列表的格式化字符串
        """
        if not self.documents:
            return "沒有已載入的文檔。"
        
        result = "已載入的文檔:\n"
        for idx, (name, content) in enumerate(self.documents.items(), 1):
            size = len(content)
            result += f"{idx}. {name} ({size} 字符)\n"
        
        return result