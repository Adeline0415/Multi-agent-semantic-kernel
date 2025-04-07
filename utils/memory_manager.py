#memory_manager.py
from typing import List, Dict, Any, Optional
import time

class MemoryManager:
    """管理對話記憶的工具類"""
    
    def __init__(self, max_items: int = 50):
        """
        初始化記憶管理器
        
        Args:
            max_items: 最大記憶項數量
        """
        self.memories = []
        self.max_items = max_items
    
    def add_memory(self, content: str, role: str, metadata: Optional[Dict[str, Any]] = None):
        """
        添加新的記憶項
        
        Args:
            content: 記憶內容
            role: 角色 (user/assistant/system)
            metadata: 額外元數據
        """
        memory_item = {
            "content": content,
            "role": role,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.memories.append(memory_item)
        
        # 如果超出最大數量，移除最舊的記憶
        if len(self.memories) > self.max_items:
            self.memories = self.memories[-self.max_items:]
    
    def get_recent_memories(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        獲取最近的記憶
        
        Args:
            count: 要獲取的記憶數量
            
        Returns:
            最近的記憶列表
        """
        return self.memories[-count:] if len(self.memories) >= count else self.memories
    
    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        搜索記憶
        
        Args:
            query: 搜索查詢
            
        Returns:
            匹配的記憶列表
        """
        # 簡單實現：關鍵詞匹配
        return [m for m in self.memories if query.lower() in m["content"].lower()]
    
    def format_as_text(self, memories: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        將記憶格式化為文本
        
        Args:
            memories: 要格式化的記憶，如果為None則使用所有記憶
            
        Returns:
            格式化的文本
        """
        if memories is None:
            memories = self.memories
        
        text = ""
        for m in memories:
            text += f"{m['role'].capitalize()}: {m['content']}\n"
        
        return text
    
    def clear(self):
        """清除所有記憶"""
        self.memories = []