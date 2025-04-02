import os
import asyncio
from typing import Dict, List, Any, Optional

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from .base_agent import Agent

class ConversationAgent(Agent):
    """處理一般對話的代理"""
    
    def __init__(self, name: str = "ConversationAgent"):
        """
        初始化對話代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name, skills=["一般對話", "問候", "閒聊", "信息提供"])
        self.chat_history = []
        self.chat_function = None
    
    def setup_kernel(self, kernel: Kernel):
        """
        設置 Semantic Kernel 並註冊對話功能
        
        Args:
            kernel: Semantic Kernel 實例
        """
        super().setup_kernel(kernel)
        self._register_chat_function()
    
    def _register_chat_function(self):
        """註冊聊天功能"""
        
        # 聊天提示模板
        chat_prompt = """
        你是一個智能助手，能夠提供有用的信息和建議。你的回答應該友善、有幫助且基於事實。
        
        {{$history}}
        用戶: {{$user_input}}
        助手: 
        """
        
        # 聊天功能配置
        chat_config = PromptTemplateConfig(
            template=chat_prompt,
            name="chat",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="user_input", description="用戶輸入", is_required=True),
                InputVariable(name="history", description="對話歷史", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=2000,
                temperature=0.7,
            )
        )
        
        # 添加聊天功能到 Kernel
        self.chat_function = self.kernel.add_function(
            function_name="chat",
            plugin_name="conversationPlugin",
            prompt_template_config=chat_config,
        )
    
    async def process_message(self, message: str, sender: Optional[str] = None) -> str:
        """
        處理一般對話
        
        Args:
            message: 用戶訊息
            sender: 訊息發送者
            
        Returns:
            回應訊息
        """
        # 確保聊天功能已註冊
        if self.chat_function is None and self.kernel is not None:
            self._register_chat_function()
        
        # 更新聊天歷史
        self.chat_history.append({"role": "user", "content": message})
        
        # 格式化歷史
        history = self._format_chat_history()
        
        # 生成回應
        try:
            answer = await self.kernel.invoke(
                self.chat_function,
                KernelArguments(user_input=message, history=history)
            )
            
            # 更新聊天歷史
            self.chat_history.append({"role": "assistant", "content": str(answer)})
            
            # 如果歷史太長，保留最近的部分
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
            return str(answer)
        except Exception as e:
            return "我現在無法處理您的請求。請稍後再試。"
    
    def _format_chat_history(self) -> str:
        """
        格式化聊天歷史
        
        Returns:
            格式化後的聊天歷史文本
        """
        history = ""
        for i in range(max(0, len(self.chat_history) - 10), len(self.chat_history)):
            msg = self.chat_history[i]
            if msg["role"] == "user":
                history += f"用戶: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history += f"助手: {msg['content']}\n"
        return history
    
    def clear_chat_history(self):
        """清除聊天歷史"""
        self.chat_history = []