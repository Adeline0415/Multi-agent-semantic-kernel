import time
import asyncio
from typing import Dict, List, Any, Optional, Callable

class Agent:
    """基礎智能體類，所有專業代理繼承自此類"""
    
    def __init__(self, name: str, skills: Optional[List[str]] = None):
        """
        初始化代理
        
        Args:
            name: 代理名稱
            skills: 代理擁有的技能列表
        """
        self.name = name
        self.skills = skills or []
        self.messages = []  # 訊息歷史
        self.kernel = None  # Semantic Kernel 實例
    
    async def receive_message(self, message: str, sender: Optional[str] = None) -> str:
        """
        接收並處理訊息
        
        Args:
            message: 接收到的訊息內容
            sender: 訊息發送者名稱
            
        Returns:
            回應訊息
        """
        # 記錄訊息
        self.messages.append({
            "content": message, 
            "sender": sender, 
            "timestamp": time.time()
        })
        
        # 處理訊息
        response = await self.process_message(message, sender)
        
        # 記錄回應
        self.messages.append({
            "content": response, 
            "sender": self.name, 
            "timestamp": time.time()
        })
        
        return response
    
    async def process_message(self, message: str, sender: Optional[str]) -> str:
        """
        處理訊息（由子類實現）
        
        Args:
            message: 訊息內容
            sender: 訊息發送者
        
        Returns:
            回應訊息
        """
        raise NotImplementedError("子類必須實現此方法")
    
    async def send_message(self, message: str, recipient: 'Agent') -> str:
        """
        發送訊息給另一個代理
        
        Args:
            message: 要發送的訊息內容
            recipient: 接收訊息的代理
            
        Returns:
            接收方的回應
        """
        if hasattr(recipient, 'receive_message'):
            return await recipient.receive_message(message, self.name)
        return None
    
    def setup_kernel(self, kernel):
        """
        設置 Semantic Kernel
        
        Args:
            kernel: Semantic Kernel 實例
        """
        self.kernel = kernel
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        獲取最近的訊息
        
        Args:
            count: 要獲取的訊息數量
            
        Returns:
            最近的訊息列表
        """
        return self.messages[-count:] if len(self.messages) >= count else self.messages
    
    def clear_messages(self):
        """清除所有訊息歷史"""
        self.messages = []