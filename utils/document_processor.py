import os
from pathlib import Path
import PyPDF2
import docx
import pandas as pd
import nbformat
from typing import Dict, List, Any, Optional

class DocumentProcessor:
    """處理各種文件格式並提取文本"""
    
    def extract_text(self, file_path: str) -> str:
        """
        從各種文件類型提取文本
        
        Args:
            file_path: 文件路徑
            
        Returns:
            提取的文本
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        elif file_extension == '.csv':
            return self.extract_text_from_csv(file_path)
        elif file_extension == '.ipynb':
            return self.extract_text_from_notebook(file_path)
        else:
            return f"不支持的文件格式: {file_extension}"
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        從PDF文件提取文本
        
        Args:
            file_path: PDF文件路徑
            
        Returns:
            提取的文本
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            text = f"從PDF提取文本時出錯: {str(e)}"
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        從DOCX文件提取文本
        
        Args:
            file_path: DOCX文件路徑
            
        Returns:
            提取的文本
        """
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            text = f"從DOCX提取文本時出錯: {str(e)}"
        return text
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        從TXT文件提取文本
        
        Args:
            file_path: TXT文件路徑
            
        Returns:
            提取的文本
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # 如果 UTF-8 解碼失敗，嘗試其他編碼
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                return f"從TXT提取文本時出錯: {str(e)}"
        except Exception as e:
            return f"從TXT提取文本時出錯: {str(e)}"
    
    def extract_text_from_csv(self, file_path: str) -> str:
        """
        從CSV文件提取文本為結構化字符串
        
        Args:
            file_path: CSV文件路徑
            
        Returns:
            提取的文本
        """
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception as e:
            try:
                # 嘗試不同的編碼
                df = pd.read_csv(file_path, encoding='latin-1')
                return df.to_string(index=False)
            except Exception as e2:
                return f"從CSV提取文本時出錯: {str(e)}, {str(e2)}"
    
    def extract_text_from_notebook(self, file_path: str) -> str:
        """
        從Jupyter Notebook文件提取文本
        
        Args:
            file_path: Notebook文件路徑
            
        Returns:
            提取的文本
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                notebook = nbformat.read(file, as_version=4)
            
            text = ""
            for cell in notebook.cells:
                if cell.cell_type == 'markdown':
                    text += cell.source + "\n\n"
                elif cell.cell_type == 'code':
                    text += f"```python\n{cell.source}\n```\n\n"
            
            return text
        except Exception as e:
            return f"從Notebook提取文本時出錯: {str(e)}"