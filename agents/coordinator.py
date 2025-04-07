# coordinator.py
import os
import time
import asyncio
import json
from typing import Dict, List, Any, Optional

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from .base_agent import Agent

class CoordinatorAgent(Agent):
    """協調器代理，負責分配任務和整合結果"""
    
    def __init__(self, name: str = "Coordinator"):
        """
        初始化協調器代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name)
        self.agents = {}  # 註冊的代理
        self.decision_function = None  # 任務分配決策函數
    
    def register_agent(self, agent_name: str, agent: Agent):
        """
        註冊代理
        
        Args:
            agent_name: 代理名稱 (用於查找)
            agent: 代理實例
        """
        self.agents[agent_name] = agent
    
    def _register_decision_function(self):
        """註冊決策功能，用於將任務分配到合適的代理"""
        
        # 決策提示模板
        decision_prompt = """
        你是一個智能協調系統，負責將用戶請求路由到最合適的專業代理處理。
        請根據用戶的輸入，選擇最合適的代理。
        
        可用的代理:
        - conversation_agent: 處理一般對話、問候、閒聊、信息提供和一般推理
        - document_agent: 處理文檔分析、摘要、文檔問答
        - code_agent: 處理代碼生成、代碼分析、代碼執行、code generation、debug、code explanation
        - creative_agent: 處理創意內容生成、寫作、創意任務
        - search_agent: 處理網絡搜尋、實時信息、事實查詢
        
        用戶輸入: {{$input}}
        
        請以 JSON 格式回復，格式如下:
        {
          "agent": "選定的代理名稱",
          "reason": "選擇該代理的原因",
          "task": "給代理的具體任務描述"
        }
        
        只返回 JSON，不要有其他多餘的解釋。
        """
        
        # 決策功能配置
        decision_config = PromptTemplateConfig(
            template=decision_prompt,
            name="routeDecision",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="input", description="用戶輸入", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=500,
                temperature=0.0,  # 確定性輸出
            )
        )
        
        # 添加決策功能到 Kernel
        self.decision_function = self.kernel.add_function(
            function_name="routeDecision",
            plugin_name="coordinatorPlugin",
            prompt_template_config=decision_config,
        )
    
    async def process_message(self, message: str, sender: Optional[str] = None) -> str:
        """
        處理用戶請求，決定由哪個代理處理
        
        Args:
            message: 用戶訊息內容
            sender: 訊息發送者 (通常是 "user")
            
        Returns:
            處理結果
        """
        try:
            # 確保決策功能已註冊
            if self.decision_function is None and self.kernel is not None:
                self._register_decision_function()
            
            # 使用決策功能選擇代理
            decision_result = await self.kernel.invoke(
                self.decision_function,
                KernelArguments(input=message)
            )
            
            # 解析決策結果
            try:
                decision = json.loads(str(decision_result))
                selected_agent = decision.get("agent")
                task = decision.get("task", message)
            except (json.JSONDecodeError, AttributeError):
                # 如果決策結果無法解析，使用備用邏輯
                selected_agent = self._fallback_decision(message)
                task = message
            
            # 檢查選定的代理是否註冊
            if selected_agent in self.agents:
                # 委派任務給選定的代理
                response = await self.agents[selected_agent].receive_message(task, self.name)
                return response
            else:
                # 如果選定的代理未註冊，使用對話代理
                if "conversation_agent" in self.agents:
                    return await self.agents["conversation_agent"].receive_message(message, self.name)
                return f"無法處理您的請求。未找到合適的代理。"
                
        except Exception as e:
            # 出錯時的友善回應
            return f"處理您的請求時出現了問題。請稍後再試。"
    
    def _fallback_decision(self, message: str) -> str:
        """
        備用決策邏輯，當 AI 決策失敗時使用
        
        Args:
            message: 用戶訊息
            
        Returns:
            選定的代理名稱
        """
        message = message.lower()
        
        # 搜索相關關鍵詞
        search_keywords = [
            "搜索", "查詢", "查找", "找找", "搜尋", "網絡", "最新", "新聞", 
            "今天", "昨天", "最近", "search", "find", "lookup", "web", 
            "internet", "news", "recent", "latest"
        ]
        
        # 代碼相關關鍵詞
        code_keywords = [
            "代碼", "程式", "編程", "函數", "方法", "變數", "循環", "條件", 
            "算法", "code", "program", "function", "method", "variable", 
            "loop", "algorithm", "python", "javascript", "java", "c++"
        ]
        
        # 文檔相關關鍵詞
        document_keywords = [
            "文檔", "文件", "pdf", "word", "excel", "表格", "摘要", "總結", 
            "document", "file", "summarize", "summary", "extract"
        ]
        
        # 創意相關關鍵詞
        creative_keywords = [
            "寫", "創作", "故事", "文章", "創意", "設計", "廣告", "標語", 
            "write", "create", "story", "article", "creative", "design", 
            "advertisement", "slogan", "poem", "poetry"
        ]
        
        # 檢查關鍵詞匹配
        if any(keyword in message for keyword in search_keywords):
            return "search_agent"
        elif any(keyword in message for keyword in code_keywords):
            return "code_agent"
        elif any(keyword in message for keyword in document_keywords):
            return "document_agent"
        elif any(keyword in message for keyword in creative_keywords):
            return "creative_agent"
        else:
            return "conversation_agent"  # 默認使用對話代理
    
    async def route_task(self, task: str, source_agent: str) -> str:
        """
        路由任務給適當的代理 (代理間協作用)
        
        Args:
            task: 任務描述
            source_agent: 發起請求的代理名稱
            
        Returns:
            處理結果
        """
        # 使用決策邏輯選擇合適的代理
        selected_agent = self._fallback_decision(task)
        
        # 確保不會路由回發起請求的代理
        if selected_agent == source_agent and len(self.agents) > 1:
            # 選擇另一個代理
            for agent_name in self.agents:
                if agent_name != source_agent:
                    selected_agent = agent_name
                    break
        
        # 委派任務
        if selected_agent in self.agents:
            return await self.agents[selected_agent].receive_message(
                f"[Task from {source_agent}]: {task}", 
                self.name
            )
        else:
            return "無法找到合適的代理來處理此任務。"