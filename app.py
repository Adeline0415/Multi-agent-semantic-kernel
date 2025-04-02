import os
import asyncio
import streamlit as st
import tempfile
import io
import base64
import requests
from PIL import Image
from dotenv import load_dotenv

from system import MultiAgentSystem

# 設置頁面標題和布局
st.set_page_config(
    page_title="Multi-Agent AI System",
    page_icon="🤖",
    layout="wide"
)

# 添加自定义CSS
st.markdown("""
<style>
/* 修改主容器样式，确保内容有足够的底部间距 */
.main .block-container {
    padding-bottom: 80px;
}

/* 固定输入框在底部 */
.stChatInputContainer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: white;
    padding: 1rem;
    z-index: 999;
    border-top: 1px solid #e6e6e6;
}

/* 调整聊天消息容器的样式 */
.stChatMessageContent {
    border-radius: 10px;
}

/* 确保页脚不被输入框覆盖 */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# 初始化會話狀態
if 'multi_agent_system' not in st.session_state:
    st.session_state.multi_agent_system = MultiAgentSystem()
    asyncio.run(st.session_state.multi_agent_system.setup())

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# 檢查網絡連接
def is_connected():
    try:
        requests.get("https://www.bing.com", timeout=4)
        return True
    except requests.exceptions.RequestException:
        return False

# 處理圖像
class ImageProcessor:
    """处理图像文件并转换为可在聊天中显示的格式"""
    
    @staticmethod
    def process_image(file_path):
        """处理图像文件并返回base64编码"""
        try:
            with Image.open(file_path) as img:
                # 调整大小以优化显示
                max_width = 800
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.LANCZOS)
                
                # 转换为bytes
                img_byte_arr = io.BytesIO()
                img_format = os.path.splitext(file_path)[1].upper().replace('.', '')
                
                if img_format.upper() in ['JPG', 'JPEG']:
                    img.save(img_byte_arr, format='JPEG')
                    mime_type = "image/jpeg"
                else:
                    img.save(img_byte_arr, format='PNG')
                    mime_type = "image/png"
                
                img_byte_arr = img_byte_arr.getvalue()
                
                # 转换为base64用于显示
                encoded = base64.b64encode(img_byte_arr).decode('utf-8')
                
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None

# 側邊欄
with st.sidebar:
    st.header("System Controls")
    
    # 上傳文件
    with st.sidebar.expander("Upload Documents", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["txt", "pdf", "docx", "csv", "py", "ipynb"], 
            key="sidebar_uploader"
        )
        doc_name = st.text_input("Document name (optional)")
        # 在 app.py 中，確保文檔上傳後正確通知系統
        if uploaded_file is not None and st.button("Upload"):
            # 保存臨時文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 上傳文件到系統並顯示結果
            result = st.session_state.multi_agent_system.upload_document(tmp_path, doc_name or uploaded_file.name)
            st.sidebar.success(result)
            
            # 確保將文檔信息添加到聊天上下文
            st.session_state.chat_history.append({
                "role": "system", 
                "content": f"文檔 '{doc_name or uploaded_file.name}' 已上傳並準備好供分析。"
            })
    
    
    
    # 顯示代理狀態
    with st.sidebar.expander("Agent Status", expanded=False):
        if st.button("Refresh Status"):
            agent_status = st.session_state.multi_agent_system.get_agent_status()
            
            for agent_name, status in agent_status.items():
                st.subheader(f"{status['name']}")
                st.write(f"Skills: {', '.join(status['skills'])}")
                st.write(f"Messages: {status['messages_count']}")
                st.write("---")
    
    # 系統重置
    with st.sidebar.expander("System Reset", expanded=False):
        if st.button("Reset System"):
            # 重置聊天歷史
            st.session_state.chat_history = []
            
            # 重置系統
            st.session_state.multi_agent_system.reset()
            
            st.sidebar.success("System has been reset!")
    
    # 顯示已上傳的文件
    st.subheader("Uploaded Documents")
    doc_names = st.session_state.multi_agent_system.get_document_names()
    if doc_names:
        for doc in doc_names:
            st.text(f"• {doc}")
    else:
        st.text("No documents uploaded yet.")

# 主界面 - 聊天
st.header("Multi-Agent AI System")

# 顯示聊天歷史
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

# 聊天輸入
user_input = st.chat_input("Enter your message...")

# 處理用戶輸入
if user_input:
    # 添加用戶消息到界面
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 添加到歷史記錄
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 處理響應
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 使用多智能體系統處理消息，並包含歷史
            response = asyncio.run(st.session_state.multi_agent_system.process_message(user_input, include_history=True))
            
            
            # 顯示響應
            st.markdown(response)
    
    # 添加響應到歷史記錄
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # 重新加載頁面以清除輸入框並刷新UI
    st.rerun()

# 在應用結束時清理臨時文件
def cleanup_temp_files():
    for file_info in st.session_state.uploaded_files:
        if "path" in file_info and os.path.exists(file_info["path"]):
            try:
                os.unlink(file_info["path"])
            except:
                pass

# 註冊清理函數
import atexit
atexit.register(cleanup_temp_files)