# system.py
import os
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import time

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from agents.base_agent import Agent
from agents.coordinator import CoordinatorAgent
from agents.conversation_agent import ConversationAgent
from agents.document_agent import DocumentAgent
from agents.code_agent import CodeAgent
from agents.creative_agent import CreativeAgent
from agents.search_agent import SearchAgent
from utils.memory_manager import MemoryManager

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
        self.search_agent = SearchAgent()
        
        # 記錄系統是否已設置
        self.is_setup = False
        # 系統記憶
        self.conversation_history = []
        # 使用記憶管理器
        self.memory_manager = MemoryManager(max_items=100)
        self.recent_document_context = None
    
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
            self.search_agent.setup_kernel(self.kernel)
            
            # 向協調器註冊所有代理
            self.coordinator.register_agent("conversation_agent", self.conversation_agent)
            self.coordinator.register_agent("document_agent", self.document_agent)
            self.coordinator.register_agent("code_agent", self.code_agent)
            self.coordinator.register_agent("creative_agent", self.creative_agent)
            self.coordinator.register_agent("search_agent", self.search_agent)
            
            # 設置完成
            self.is_setup = True
            
        except Exception as e:
            raise Exception(f"設置多智能體系統時出錯: {str(e)}")
        
    async def process_message(self, message: str, include_history: bool = True) -> str:
        """
        處理用戶消息，並可選擇性地包含對話歷史
        
        Args:
            message: 用戶消息內容
            include_history: 是否包含對話歷史
            
        Returns:
            系統回應
        """
        # 確保系統已設置
        if not self.is_setup:
            await self.setup()
        
        try:
            # 檢查消息是否涉及文檔
            doc_names = self.document_agent.get_document_names()
            if doc_names:
                # 檢查是否需要包含文檔內容
                include_doc = False
                doc_to_include = None
                
                # 1. 檢查是否有最近上傳的文檔上下文
                if hasattr(self, 'recent_document_context') and self.recent_document_context:
                    # 如果最近 10 分鐘有上傳文檔，將其包含在下一條消息處理中
                    time_diff = time.time() - self.recent_document_context["timestamp"]
                    if time_diff < 600:  # 10分鐘內
                        include_doc = True
                        doc_to_include = self.recent_document_context["name"]
                        # 使用後清除上下文，避免重複包含
                        self.recent_document_context = None
                
                # 2. 使用 AI 判斷消息是否與已有文檔相關
                if not include_doc:
                    is_doc_related = await self._is_message_document_related(message, doc_names)
                    if is_doc_related:
                        include_doc = True
                        # 找出明確提到的文檔或最後上傳的文檔
                        for doc_name in doc_names:
                            if doc_name.lower() in message.lower():
                                doc_to_include = doc_name
                                break
                        
                        # 如果沒有明確提到文檔，使用最後一個
                        if not doc_to_include:
                            doc_to_include = doc_names[-1]
                
                # 如果需要包含文檔，準備增強消息
                if include_doc and doc_to_include:
                    doc_content = self.document_agent.get_document_content(doc_to_include)
                    if doc_content:
                        enhanced_message = f"[文檔: {doc_to_include}]\n{doc_content}\n\n[用戶問題]\n{message}"
                        response = await self.document_agent.process_message(enhanced_message, "user")
                        
                        # 更新記憶
                        self.memory_manager.add_memory(message, "user")
                        self.memory_manager.add_memory(response, "assistant")
                        
                        return response
            
            # 如果沒有包含文檔或沒有文檔，使用標準流程
            if include_history:
                # 獲取最近的記憶並格式化
                history_text = self.memory_manager.format_as_text(
                    self.memory_manager.get_recent_memories(10)
                )
                enhanced_message = f"[對話歷史]\n{history_text}\n\n[新問題]\n{message}"
            else:
                enhanced_message = message
            
            # 通過協調器處理消息
            response = await self.coordinator.process_message(enhanced_message, "user")
            
            # 更新記憶
            self.memory_manager.add_memory(message, "user")
            self.memory_manager.add_memory(response, "assistant")
            
            return response
        except Exception as e:
            return f"處理您的請求時出錯: {str(e)}"
    
    async def _is_message_document_related(self, message: str, doc_names: List[str]) -> bool:
        """使用 AI 判斷消息是否與文檔相關"""
        # 註冊判斷函數（如果尚未註冊）
        if not hasattr(self, "doc_relevance_function"):
            from semantic_kernel.prompt_template import PromptTemplateConfig
            from semantic_kernel.prompt_template.input_variable import InputVariable
            from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
            
            prompt = """
            請判斷用戶的消息是否與文檔相關。用戶可能直接或間接地引用文檔，例如：
            1. 直接提及文檔（「請分析這個PDF」、「查看我上傳的文件」等）
            2. 間接引用（「文件裡說了什麼」、「裡面有什麼內容」等）
            3. 詢問文檔信息（「內容摘要是什麼」、「能總結一下嗎」等）
            4. 引用剛上傳的文件（「我剛剛上傳的檔案」、「這個文件」等）
            
            目前系統中的文檔: {{$document_names}}
            
            用戶消息: {{$message}}
            
            請僅回答 "是" 或 "否"。
            """
            
            config = PromptTemplateConfig(
                template=prompt,
                name="isDocumentRelated",
                template_format="semantic-kernel",
                input_variables=[
                    InputVariable(name="document_names", description="系統中的文檔名稱", is_required=True),
                    InputVariable(name="message", description="用戶消息", is_required=True),
                ],
                execution_settings=AzureChatPromptExecutionSettings(
                    service_id="default",
                    max_tokens=100,
                    temperature=0.0,  # 確定性輸出
                )
            )
            
            self.doc_relevance_function = self.kernel.add_function(
                function_name="isDocumentRelated",
                plugin_name="systemPlugin",
                prompt_template_config=config,
            )
        
        # 調用 AI 判斷
        from semantic_kernel.functions import KernelArguments
        doc_names_str = ", ".join(doc_names)
        result = await self.kernel.invoke(
            self.doc_relevance_function,
            KernelArguments(document_names=doc_names_str, message=message)
        )
        
        result_str = str(result).strip().lower()
        
        is_related = "是" in result_str or "yes" in result_str
        print(f"Document relevance for '{message[:50]}...': {result_str} -> {is_related}")
        return is_related

    async def search(self, query: str) -> str:
        """
        直接執行搜索，繞過協調器
        
        Args:
            query: 搜索查詢
            
        Returns:
            搜索結果
        """
        # 確保系統已設置
        if not self.is_setup:
            await self.setup()
        
        return await self.search_agent.process_message(query)
    
    def upload_document(self, file_path: str, document_name: Optional[str] = None) -> str:
        """
        上傳文檔到文檔代理
        
        Args:
            file_path: 文件路徑
            document_name: 文檔名稱 (可選)
            
        Returns:
            上傳結果訊息
        """
        result = self.document_agent.load_document(file_path, document_name)
        
        # 如果上傳成功，記錄文檔上下文
        if "已成功加載" in result:
            actual_name = document_name or os.path.basename(file_path)
            doc_content = self.document_agent.get_document_content(actual_name)
            if doc_content:
                # 更新最近的文檔上下文
                self.recent_document_context = {
                    "name": actual_name,
                    "timestamp": time.time(),
                    "preview": doc_content[:500] + "..." if len(doc_content) > 500 else doc_content
                }
                
                # 添加到系統記憶
                self.memory_manager.add_memory(
                    f"文檔 '{actual_name}' 已上傳並添加到對話上下文。",
                    "system"
                )
        print(f"Document context set: {self.recent_document_context}") #debug
        return result
    
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
            "creative": self.creative_agent,
            "search": self.search_agent
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
        # 清除系統記憶
        self.conversation_history = []

    def _format_conversation_history(self) -> str:
        """格式化對話歷史為文本"""
        history_text = ""
        for i, message in enumerate(self.conversation_history):
            role = "用戶" if message["role"] == "user" else "助手"
            history_text += f"{role}: {message['content']}\n"
        return history_text