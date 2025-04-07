#search_agent.py
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
        你是一個負責處理網絡搜索結果的智能助手。請根據以下搜索結果回答用戶的問題。

        用戶問題: {{$query}}

        搜索結果:
        {{$search_results}}

        請基於上述搜索結果提供具體且有用的回答。請確保：
        1. 如果搜索結果中有與用戶問題相關的信息，請直接提供這些信息
        2. 如果找到多個相關信息，請整合並總結這些信息
        3. 如果搜索結果中沒有與用戶問題直接相關的信息，請誠實地說明
        4. 提供相關建議幫助用戶獲取更多信息

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

    async def preprocess_search_query(self, message: str) -> str:
        """
        預處理搜索查詢，從增強的消息中提取核心搜索內容
        
        Args:
            message: 可能包含歷史記錄的增強消息
            
        Returns:
            優化後的搜索查詢
        """
        # 如果消息簡短，直接使用
        if len(message) < 100 and "[對話歷史]" not in message and "[新問題]" not in message:
            return message
            
        # 配置提取查詢的提示模板
        extract_query_prompt = """
        你是一個智能搜索助手。請分析以下對話歷史和新問題，然後提取出最適合用於網絡搜索的簡潔查詢。
        
        對話內容:
        {{$message}}
        
        請僅返回一個簡短的搜索查詢（不超過10個詞），無需其他解釋。這個查詢應該捕捉用戶真正想要搜索的核心內容。
        """
        
        # 創建臨時函數來提取搜索查詢
        extract_config = PromptTemplateConfig(
            template=extract_query_prompt,
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="message", description="包含對話歷史的消息", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=50,
                temperature=0.0,  # 使用確定性輸出
            )
        )
        
        # 添加提取函數到 Kernel
        extract_function = self.kernel.add_function(
            function_name="extractSearchQuery",
            plugin_name="searchPlugin",
            prompt_template_config=extract_config,
        )
        
        # 執行查詢提取
        try:
            optimized_query = await self.kernel.invoke(
                extract_function,
                KernelArguments(message=message)
            )
            
            # 確保結果是字符串並移除多餘空格
            result = str(optimized_query).strip()
            print(f"原始輸入: {message[:100]}...")
            print(f"優化後的搜索查詢: {result}")
            
            return result
        except Exception as e:
            print(f"提取搜索查詢時出錯: {str(e)}")
            
            # 如果出錯，嘗試簡單提取 [新問題] 後的內容
            if "[新問題]" in message:
                parts = message.split("[新問題]")
                if len(parts) > 1:
                    return parts[1].strip()
            
            # 如果上述方法都失敗，則返回原始消息
            return message
    
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
            # 預處理搜索查詢
            optimized_query = await self.preprocess_search_query(message)
            
            # 執行搜索
            search_results = await self.bing_search(optimized_query)
            
            # 記錄搜索信息
            print(f"原始消息: {message[:100]}...")
            print(f"優化查詢: {optimized_query}")
            print(f"搜索結果大小: {len(search_results)} 字符")
            
            # 使用 AI 處理搜索結果，但使用原始問題來保持上下文
            response = await self.kernel.invoke(
                self.search_function,
                KernelArguments(
                    query=message,  # 原始問題提供完整上下文
                    optimized_query=optimized_query,  # 提供優化後的查詢供參考
                    search_results=search_results  # 搜索結果
                )
            )
            
            return str(response)
        except Exception as e:
            import traceback
            print(f"搜索處理錯誤: {str(e)}")
            print(traceback.format_exc())
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