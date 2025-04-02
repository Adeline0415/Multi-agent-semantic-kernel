import os
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from agents.base_agent import Agent
from agents.coordinator import CoordinatorAgent
from agents.conversation_agent import ConversationAgent
from agents.document_agent import DocumentAgent
from agents.code_agent import CodeAgent
from agents.creative_agent import CreativeAgent

class MultiAgentSystem:
    """多智能體系統主類"""
    
    def __init__(self):
        """初始化多智能體系統"""
        # 載入環境變數
        load_dotenv()
        
        # 檢查必要的環境變數
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"缺少必要的環境變數: {', '.join(missing_vars)}")
        
        # 初始化 Semantic Kernel
        self.kernel = Kernel()
        
        # 初始化協調器
        self.coordinator = CoordinatorAgent()
        
        # 初始化各專業代理
        self.conversation_agent = ConversationAgent()
        self.document_agent = DocumentAgent()
        self.code_agent = CodeAgent()
        self.creative_agent = CreativeAgent()
        
        # 記錄系統是否已設置
        self.is_setup = False
    
    async def setup(self):
        """設置系統 - 添加 AI 服務並配置代理"""
        if self.is_setup:
            return
        
        # 設置 Azure OpenAI 服務
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        try:
            # 添加Azure OpenAI Chat Completion服務
            self.kernel.add_service(
                AzureChatCompletion(
                    service_id="default",
                    deployment_name=deployment_name,
                    endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
            )
            
            # 設置 Bing 搜索服務 (如果有API密鑰)
            bing_key = os.getenv("BING_SEARCH_API_KEY")
            if bing_key:
                # 將在未來版本添加 Bing 搜索服務
                pass
            
            # 設置各代理的 Kernel
            self.coordinator.setup_kernel(self.kernel)
            self.conversation_agent.setup_kernel(self.kernel)
            self.document_agent.setup_kernel(self.kernel)
            self.code_agent.setup_kernel(self.kernel)
            self.creative_agent.setup_kernel(self.kernel)
            
            # 向協調器註冊所有代理
            self.coordinator.register_agent("conversation_agent", self.conversation_agent)
            self.coordinator.register_agent("document_agent", self.document_agent)
            self.coordinator.register_agent("code_agent", self.code_agent)
            self.coordinator.register_agent("creative_agent", self.creative_agent)
            
            # 設置完成
            self.is_setup = True
            
        except Exception as e:
            raise Exception(f"設置多智能體系統時出錯: {str(e)}")
    
    async def process_message(self, message: str) -> str:
        """
        處理用戶消息
        
        Args:
            message: 用戶消息內容
            
        Returns:
            系統回應
        """
        # 確保系統已設置
        if not self.is_setup:
            await self.setup()
        
        try:
            # 通過協調器處理消息
            response = await self.coordinator.process_message(message, "user")
            return response
        except Exception as e:
            return f"處理您的請求時出錯: {str(e)}"
    
    def upload_document(self, file_path: str, document_name: Optional[str] = None) -> str:
        """
        上傳文檔到文檔代理
        
        Args:
            file_path: 文件路徑
            document_name: 文檔名稱 (可選)
            
        Returns:
            上傳結果訊息
        """
        return self.document_agent.load_document(file_path, document_name)
    
    def get_document_names(self) -> List[str]:
        """
        獲取所有已加載文檔的名稱
        
        Returns:
            文檔名稱列表
        """
        return self.document_agent.get_document_names()
    
    def get_all_agents(self) -> Dict[str, Agent]:
        """
        獲取所有代理
        
        Returns:
            代理字典 {名稱: 代理實例}
        """
        return {
            "coordinator": self.coordinator,
            "conversation": self.conversation_agent,
            "document": self.document_agent,
            "code": self.code_agent,
            "creative": self.creative_agent
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        獲取所有代理的狀態
        
        Returns:
            代理狀態字典
        """
        agents = self.get_all_agents()
        status = {}
        
        for name, agent in agents.items():
            status[name] = {
                "name": agent.name,
                "skills": agent.skills,
                "messages_count": len(agent.messages)
            }
        
        return status
    
    def reset(self):
        """重置系統，清除所有代理的消息歷史"""
        agents = self.get_all_agents()
        for agent in agents.values():
            agent.clear_messages()
        
        # 清除對話代理的聊天歷史
        self.conversation_agent.clear_chat_history()