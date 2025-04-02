import os
import asyncio
from typing import Dict, List, Any, Optional

from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from .base_agent import Agent

class CreativeAgent(Agent):
    """處理創意內容生成的代理"""
    
    def __init__(self, name: str = "CreativeAgent"):
        """
        初始化創意代理
        
        Args:
            name: 代理名稱
        """
        super().__init__(name, skills=["內容生成", "創意寫作", "故事創作", "廣告文案"])
        self.generate_function = None
    
    def setup_kernel(self, kernel: Kernel):
        """
        設置 Semantic Kernel 並註冊內容生成功能
        
        Args:
            kernel: Semantic Kernel 實例
        """
        super().setup_kernel(kernel)
        self._register_generate_function()
    
    def _register_generate_function(self):
        """註冊內容生成功能"""
        
        # 內容生成提示模板
        generate_prompt = """
        基於以下說明和上下文生成內容:
        
        說明: {{$instructions}}
        上下文: {{$context}}
        
        您生成的內容應該:
        1. 結構良好且連貫
        2. 與提供的上下文相關
        3. 嚴格遵循給定的說明
        4. 專業且精煉
        
        生成的內容:
        """
        
        # 內容生成配置
        generate_config = PromptTemplateConfig(
            template=generate_prompt,
            name="generateContent",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="instructions", description="內容生成的說明", is_required=True),
                InputVariable(name="context", description="內容生成的上下文", is_required=True),
            ],
            execution_settings=AzureChatPromptExecutionSettings(
                service_id="default",
                max_tokens=2000,
                temperature=0.7,
            )
        )
        
        # 添加內容生成功能到 Kernel
        self.generate_function = self.kernel.add_function(
            function_name="generateContent",
            plugin_name="creativePlugin",
            prompt_template_config=generate_config,
        )
    
    async def process_message(self, message: str, sender: Optional[str] = None) -> str:
        """
        處理創意內容生成請求
        
        Args:
            message: 用戶訊息
            sender: 訊息發送者
            
        Returns:
            生成的創意內容
        """
        # 確保生成功能已註冊
        if self.generate_function is None and self.kernel is not None:
            self._register_generate_function()
        
        try:
            # 提取指令和上下文
            instructions = self._extract_instructions(message)
            context = self._extract_context(message)
            
            # 生成內容
            content = await self.generate_content(instructions, context)
            
            return content
        except Exception as e:
            return f"生成創意內容時出錯: {str(e)}"
    
    async def generate_content(self, instructions: str, context: str) -> str:
        """
        生成創意內容
        
        Args:
            instructions: 內容生成指令
            context: 內容上下文
            
        Returns:
            生成的內容
        """
        try:
            content = await self.kernel.invoke(
                self.generate_function,
                KernelArguments(instructions=instructions, context=context)
            )
            return str(content)
        except Exception as e:
            raise Exception(f"內容生成失敗: {str(e)}")
    
    def _extract_instructions(self, message: str) -> str:
        """
        從訊息中提取指令
        
        Args:
            message: 用戶訊息
            
        Returns:
            提取的指令
        """
        # 簡單實現 - 使用整個訊息作為指令
        return message
    
    def _extract_context(self, message: str) -> str:
        """
        從訊息中提取上下文
        
        Args:
            message: 用戶訊息
            
        Returns:
            提取的上下文
        """
        # 簡單實現 - 返回空上下文
        # 在實際應用中，可以從對話歷史或其他源提取上下文
        return ""