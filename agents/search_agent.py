import os
import asyncio
import requests
import json
from typing import Dict, List, Any, Optional

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from .base_agent import Agent

class SearchAgent(Agent):
    """處理網絡搜索的代理"""
    
    def __init__(self, name: str = "SearchAgent"):
        """
        初始化搜索代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name, skills=["網絡搜索", "實時信息", "信息檢索"])
        self.search_function = None
        self.bing_api_key = os.getenv("BING_SEARCH_API_KEY", "")
    
    def setup_kernel(self, kernel: Kernel):
        """
        設置 Semantic Kernel 並註冊搜索功能
        
        Args:
            kernel: Semantic Kernel 實例
        """
        super().setup_kernel(kernel)
        self._register_search_function()
    
    def _register_search_function(self):
        """註冊搜索功能"""
        
        # 搜索結果處理提示模板
        search_prompt = """
        你是一個負責處理網絡搜索結果的智能助手。請根據以下搜索結果回答問題：
        
        搜索查詢: {{$query}}
        
        搜索結果:
        {{$search_results}}
        
        請基於搜索結果提供全面且有條理的回答。包括相關事實、數據和來源網站（如果適用）。
        如果搜索結果中沒有足夠的信息來回答問題，請誠實地說明。
        
        回答:
        """
        
        # 搜索功能配置
        search_config = PromptTemplateConfig(
            template=search_prompt,
            name="processSearchResults",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="query", description="搜索查詢", is_required=True),
                InputVariable(name="search_results", description="搜索結果", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=2000,
                temperature=0.3,
            )
        )
        
        # 添加搜索功能到 Kernel
        self.search_function = self.kernel.add_function(
            function_name="processSearchResults",
            plugin_name="searchPlugin",
            prompt_template_config=search_config,
        )
    
    async def process_message(self, message: str, sender: Optional[str] = None) -> str:
        """
        處理搜索請求
        
        Args:
            message: 用戶訊息
            sender: 訊息發送者
            
        Returns:
            搜索結果
        """
        # 確保搜索功能已註冊
        if self.search_function is None and self.kernel is not None:
            self._register_search_function()
        
        # 驗證 API 密鑰
        if not self.bing_api_key:
            return "搜索功能未配置。請設置 BING_SEARCH_API_KEY 環境變數。"
        
        try:
            # 執行搜索
            search_results = await self.bing_search(message)
            
            # 使用 AI 處理搜索結果
            response = await self.kernel.invoke(
                self.search_function,
                KernelArguments(query=message, search_results=search_results)
            )
            
            return str(response)
        except Exception as e:
            return f"搜索過程中出錯: {str(e)}"
    
    async def bing_search(self, query: str, count: int = 5) -> str:
        """
        執行 Bing 搜索
        
        Args:
            query: 搜索查詢
            count: 返回結果數量
            
        Returns:
            格式化的搜索結果
        """
        try:
            search_url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
            params = {
                "q": query,
                "count": count,
                "responseFilter": "Webpages",
                "textDecorations": True,
                "textFormat": "HTML"
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            search_results = response.json()
            
            # 檢查是否有搜索結果
            if "webPages" not in search_results:
                return "沒有找到相關結果。"
            
            web_pages = search_results["webPages"]["value"]
            
            # 格式化搜索結果
            formatted_results = ""
            for i, page in enumerate(web_pages, 1):
                name = page.get("name", "無標題")
                url = page.get("url", "無URL")
                snippet = page.get("snippet", "無摘要")
                
                formatted_results += f"{i}. {name}\n"
                formatted_results += f"   URL: {url}\n"
                formatted_results += f"   {snippet}\n\n"
            
            return formatted_results
        
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.ConnectionError):
                return "網絡連接錯誤。請檢查您的網絡連接。"
            elif isinstance(e, requests.exceptions.Timeout):
                return "搜索請求超時。請稍後再試。"
            elif isinstance(e, requests.exceptions.HTTPError):
                if e.response.status_code == 401:
                    return "API密鑰無效或未授權。"
                elif e.response.status_code == 403:
                    return "API密鑰權限不足。"
                elif e.response.status_code == 429:
                    return "超出API使用限制。請稍後再試。"
                else:
                    return f"HTTP錯誤: {e.response.status_code}"
            else:
                return f"搜索請求錯誤: {str(e)}"