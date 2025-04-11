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
        
        # 決策提示模板，更強調對用戶真實意圖的理解
        decision_prompt = """
        你是一個智能協調系統，負責將用戶請求路由到最合適的專業代理處理。
        請仔細分析用戶的輸入，理解其真實意圖，然後選擇最合適的代理。

        可用的代理:
        - conversation_agent: 處理一般對話、問候、閒聊、問答、概念解釋、推理分析
        - code_agent: 處理明確要求生成或分析代碼的請求，以及生成任何類型檔案的請求
        - document_agent: 處理閱讀和分析已上傳文檔的問題，但不負責生成新檔案
        - creative_agent: 處理創意內容生成、寫作、創意任務
        - search_agent: 處理明確需要最新網絡信息的搜尋請求

        重要提示:
        1. 任何涉及「生成檔案」、「整理成檔案」、「保存為檔案」、「下載」的請求，無論內容是什麼，都必須由 code_agent 處理
        2. 特別是 PDF、Excel、Word、CSV 等檔案格式的生成，一定要選擇 code_agent，而不是 document_agent
        3. document_agent 僅用於讀取和分析已上傳的文檔，它無法生成新的檔案
        4. 如果用戶要求介紹某個主題並整理成檔案，應該選擇 code_agent
        5. 如果用戶明確提到「下載」或「儲存」，這表示需要生成檔案，應選擇 code_agent
        6. 用戶如果只是提到程式相關概念但沒有要求生成代碼或檔案，應該使用 conversation_agent
        7. 只有當用戶明確需要網絡查詢或最新信息時才選擇 search_agent
        8. 若用戶請求涉及分析已上傳文檔的內容，應選擇 document_agent
        9. 當用戶剛上傳文件後提問，即使沒有明確提及文檔，也應該優先考慮document_agent
        10. 選擇題、測驗問題、理論性問題應該使用conversation_agent
        11. 若無法判斷用戶要使用何種agent則default使用conversation_agent
        12. 盡量使用用戶提問的語言作答
        
        用戶輸入: {{$input}}
        
        請以 JSON 格式回復，格式如下:
        {
        "agent": "選定的代理名稱",
        "reason": "選擇該代理的詳細原因，說明你如何理解用戶的真實意圖",
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
                max_tokens=800,  # 增加 token 數以允許更詳細的分析
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
            
            # 提取最新訊息進行所有決策
            latest_message = message
            if "[新問題]" in message:
                parts = message.split("[新問題]")
                if len(parts) > 1:
                    latest_message = parts[1].strip()
            
            # 1. 優先使用 AI 判斷是否為檔案生成請求
            if self.kernel is not None:
                is_file_gen_by_ai = await self._is_file_generation_request(latest_message)
                if is_file_gen_by_ai and "code_agent" in self.agents:
                    print(f"File generation detected by AI, routing to code_agent: '{latest_message[:50]}...'") #debug
                    task_with_marker = f"[FILE_GENERATION_MODE=True]\n{message}"
                    return await self.agents["code_agent"].receive_message(task_with_marker, self.name)
            
            # 2. 備用方案：使用關鍵字檢測
            latest_message_lower = latest_message.lower()
            
            # 檔案生成相關動詞
            file_verbs = [
                "生成", "產生", "創建", "建立", "做一個", "做個", "做成", "製作", 
                "存為", "儲存", "保存", "下載", "輸出", "轉成", "轉換", "轉為", 
                "整理成", "整理為", "彙整", "匯出", "輸出", "寫入", "寫成",
                "create", "generate", "make", "build", "produce", "download",
                "save", "store", "export", "output", "convert", "transform", 
                "organize", "compile", "write"
            ]
            
            # 檔案相關名詞
            file_nouns = [
                "檔案", "文件", "文檔", "表格", "報表", "報告", 
                "pdf", "excel", "xlsx", "word", "docx", "csv", "txt", "文本", 
                "試算表", "簡報", "投影片", "ppt", "pptx", "json", "xml", "yaml",
                "file", "document", "spreadsheet", "report", "presentation", "text"
            ]
            
            # 檢查是否同時包含動詞和名詞
            has_file_verb = any(verb in latest_message_lower for verb in file_verbs)
            has_file_noun = any(noun in latest_message_lower for noun in file_nouns)
            
            # 直接檔案擴展名檢測 (更寬鬆的匹配)
            has_file_extension = any(ext in latest_message_lower for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt", ".ppt", ".pptx"])
            
            # 檢查一些常見的完整短語
            common_phrases = [
                "存成檔案", "另存為", "輸出檔案", "製作報表", "生成報告", 
                "整理資料", "匯出數據", "save as", "output file"
            ]
            has_common_phrase = any(phrase in latest_message_lower for phrase in common_phrases)
            
            # 如果同時包含動詞和名詞，或者有檔案擴展名，或者有常見短語，判定為檔案生成請求
            is_file_gen_request = (has_file_verb and has_file_noun) or has_file_extension or has_common_phrase
            
            # 如果關鍵字檢測判定為檔案生成請求，路由到code_agent
            if is_file_gen_request and "code_agent" in self.agents:
                print(f"File generation detected by keywords, routing to code_agent: '{latest_message[:50]}...'") #debug
                task_with_marker = f"[FILE_GENERATION_MODE=True]\n{message}"
                return await self.agents["code_agent"].receive_message(task_with_marker, self.name)
            
            # 3. 常規 AI 決策流程
            decision_result = await self.kernel.invoke(
                self.decision_function,
                KernelArguments(input=latest_message)
            )
            
            # 解析決策結果
            try:
                decision = json.loads(str(decision_result))
                selected_agent = decision.get("agent")
                task = decision.get("task", message)
            except (json.JSONDecodeError, AttributeError):
                # 如果決策結果無法解析，使用備用邏輯
                selected_agent = self._fallback_decision(latest_message)
                task = message
            
            # 檢查選定的代理是否註冊
            if selected_agent in self.agents:
                # 委派任務給選定的代理
                response = await self.agents[selected_agent].receive_message(task, self.name)
                print(f"Routing decision for message: '{latest_message[:50]}...' -> {selected_agent}") #debug
                return response
            else:
                # 如果選定的代理未註冊，使用對話代理
                if "conversation_agent" in self.agents:
                    return await self.agents["conversation_agent"].receive_message(message, self.name)
                return f"無法處理您的請求。未找到合適的代理。"
                
        except Exception as e:
            # 出錯時的友善回應
            import traceback
            print(f"處理請求時出錯: {str(e)}\n{traceback.format_exc()}")
            return f"處理您的請求時出現了問題。請稍後再試。"
        
    async def _is_file_generation_request(self, message: str) -> bool:
        """使用 AI 判斷是否為檔案生成請求"""
        # 註冊判斷函數（如果尚未註冊）
        if not hasattr(self, "file_gen_function"):
            from semantic_kernel.prompt_template import PromptTemplateConfig
            from semantic_kernel.prompt_template.input_variable import InputVariable
            from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
            
            prompt = """
            請判斷用戶的消息是否要求生成或創建任何類型的文件或檔案。這包括：
            1. 直接請求建立文件（如「做一個PDF」、「生成報表」）
            2. 間接要求輸出到檔案（如「將資料整理好」並暗示需要檔案形式）
            3. 任何提到將信息或數據轉換為檔案格式的請求
            4. 要求下載或保存內容的請求
            
            即使用戶用詞不精確或不完整，只要意圖是生成檔案，也請判斷為「是」。
            
            用戶消息: {{$message}}
            
            請僅回答「是」或「否」。
            """
            
            config = PromptTemplateConfig(
                template=prompt,
                name="isFileGenerationRequest",
                template_format="semantic-kernel",
                input_variables=[
                    InputVariable(name="message", description="用戶消息", is_required=True),
                ],
                execution_settings=AzureChatPromptExecutionSettings(
                    service_id="default",
                    max_tokens=100,
                    temperature=0.0,  # 確定性輸出
                )
            )
            
            self.file_gen_function = self.kernel.add_function(
                function_name="isFileGenerationRequest",
                plugin_name="coordinatorPlugin",
                prompt_template_config=config,
            )
        
        # 調用 AI 判斷
        from semantic_kernel.functions import KernelArguments
        result = await self.kernel.invoke(
            self.file_gen_function,
            KernelArguments(message=message)
        )
        
        result_str = str(result).strip().lower()
        return "是" in result_str or "yes" in result_str
    
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
            "loop", "algorithm", "python", "javascript", "java", "c++", "generate",
            "create", "file", "save", "download", "整理成檔案", "生成檔案","檔案", "文件生成", "下載", "儲存", "生成"
        ]
        
        # 文檔相關關鍵詞
        document_keywords = [
            "文檔", "文件分析", "摘要", "總結", "上傳的", "已上傳", 
            "document", "uploaded", "summarize", "summary", "extract"
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