import os
import sys
import shutil
import platform
import asyncio
import subprocess
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class EnvironmentChecker:
    """檢查系統環境並提供安裝指南"""
    
    def __init__(self):
        """初始化環境檢查器"""
        self.os_type = platform.system().lower()  # 'windows', 'darwin' (macOS), 'linux'
        self.language_environments = {
            "python": self._check_python,
            "javascript": self._check_javascript,
            "java": self._check_java,
            "csharp": self._check_csharp,
            "cpp": self._check_cpp,
            "php": self._check_php,
            "ruby": self._check_ruby,
            "r": self._check_r
        }
    
    async def check_environment(self, language: str) -> Tuple[bool, str]:
        """
        檢查特定語言的執行環境
        
        Args:
            language: 程式語言
            
        Returns:
            (是否支持, 資訊/指南)
        """
        language = self._normalize_language(language)
        
        if language in self.language_environments:
            return await self.language_environments[language]()
        else:
            return False, f"不支持檢查 {language} 語言環境。"
    
    def _normalize_language(self, language: str) -> str:
        """標準化語言名稱"""
        language = language.lower().strip()
        
        # 映射語言別名到標準名稱
        language_map = {
            "py": "python",
            "js": "javascript",
            "node": "javascript",
            "c#": "csharp",
            "cs": "csharp",
            "c++": "cpp",
            "c": "cpp",
            "rb": "ruby"
        }
        
        return language_map.get(language, language)
    
    async def _check_python(self) -> Tuple[bool, str]:
        """檢查 Python 環境"""
        # Python 已經在運行，所以必定可用
        version = platform.python_version()
        return True, f"Python {version} 已安裝並可用。"
    
    async def _check_javascript(self) -> Tuple[bool, str]:
        """檢查 JavaScript (Node.js) 環境"""
        node_path = shutil.which("node")
        npm_path = shutil.which("npm")
        
        if node_path and npm_path:
            # 取得 Node.js 版本
            try:
                process = await asyncio.create_subprocess_exec(
                    "node", "--version",
                    stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                version = stdout.decode().strip()
                return True, f"Node.js {version} 與 npm 已安裝並可用。"
            except:
                return True, "Node.js 與 npm 已安裝並可用。"
        else:
            # 提供安裝指南
            guide = "### 安裝 Node.js 與 npm\n\n"
            
            if self.os_type == "windows":
                guide += "1. 訪問 Node.js 官方網站: https://nodejs.org/\n"
                guide += "2. 下載並運行 Windows 安裝程序 (.msi 檔案)\n"
                guide += "3. 依照安裝嚮導的指示完成安裝\n"
            elif self.os_type == "darwin":  # macOS
                guide += "方法 1: 使用官方安裝程序\n"
                guide += "1. 訪問 Node.js 官方網站: https://nodejs.org/\n"
                guide += "2. 下載並運行 macOS 安裝程序 (.pkg 檔案)\n\n"
                guide += "方法 2: 使用 Homebrew\n"
                guide += "1. 開啟終端機\n"
                guide += "2. 執行: `brew install node`\n"
            elif self.os_type == "linux":
                guide += "使用套件管理器:\n"
                guide += "- Ubuntu/Debian: `sudo apt update && sudo apt install nodejs npm`\n"
                guide += "- Fedora: `sudo dnf install nodejs`\n"
                guide += "- Arch Linux: `sudo pacman -S nodejs npm`\n\n"
                guide += "或者訪問 Node.js 官方網站獲取更多安裝選項: https://nodejs.org/\n"
            
            return False, guide
    
    async def _check_java(self) -> Tuple[bool, str]:
        """檢查 Java 環境"""
        java_path = shutil.which("java")
        javac_path = shutil.which("javac")
        
        if java_path and javac_path:
            # 取得 Java 版本
            try:
                process = await asyncio.create_subprocess_exec(
                    "java", "--version",
                    stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                version_info = stdout.decode().strip().split("\n")[0]
                return True, f"Java 開發環境已安裝並可用: {version_info}"
            except:
                return True, "Java 開發環境已安裝並可用。"
        else:
            # 提供安裝指南
            guide = "### 安裝 Java 開發環境 (JDK)\n\n"
            
            if self.os_type == "windows":
                guide += "1. 訪問 Oracle JDK 下載頁面: https://www.oracle.com/java/technologies/downloads/\n"
                guide += "   或 OpenJDK: https://adoptium.net/\n"
                guide += "2. 下載並運行適用於 Windows 的安裝程序\n"
                guide += "3. 依照安裝嚮導的指示完成安裝\n"
                guide += "4. 設定 JAVA_HOME 環境變數指向 JDK 安裝目錄\n"
            elif self.os_type == "darwin":  # macOS
                guide += "方法 1: 使用官方安裝程序\n"
                guide += "1. 訪問 Oracle JDK 下載頁面: https://www.oracle.com/java/technologies/downloads/\n"
                guide += "   或 OpenJDK: https://adoptium.net/\n"
                guide += "2. 下載並運行 macOS 安裝程序\n\n"
                guide += "方法 2: 使用 Homebrew\n"
                guide += "1. 開啟終端機\n"
                guide += "2. 執行: `brew install openjdk@17`\n"
            elif self.os_type == "linux":
                guide += "使用套件管理器:\n"
                guide += "- Ubuntu/Debian: `sudo apt update && sudo apt install default-jdk`\n"
                guide += "- Fedora: `sudo dnf install java-latest-openjdk java-latest-openjdk-devel`\n"
                guide += "- Arch Linux: `sudo pacman -S jdk-openjdk`\n"
            
            return False, guide
    
    # 實現其他語言的環境檢查...
    async def _check_csharp(self) -> Tuple[bool, str]:
        """檢查 C# (.NET) 環境"""
        dotnet_path = shutil.which("dotnet")
        
        if dotnet_path:
            try:
                process = await asyncio.create_subprocess_exec(
                    "dotnet", "--version",
                    stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                version = stdout.decode().strip()
                return True, f".NET SDK {version} 已安裝並可用。"
            except:
                return True, ".NET SDK 已安裝並可用。"
        else:
            guide = "### 安裝 .NET SDK (用於 C# 開發)\n\n"
            guide += "1. 訪問 .NET 下載頁面: https://dotnet.microsoft.com/download\n"
            guide += "2. 下載並安裝 .NET SDK\n"
            
            if self.os_type == "windows":
                guide += "3. 運行下載的安裝程序 (.exe 檔案)\n"
            elif self.os_type == "darwin":  # macOS
                guide += "3. 運行下載的安裝程序 (.pkg 檔案)\n"
                guide += "或使用 Homebrew: `brew install dotnet-sdk`\n"
            elif self.os_type == "linux":
                guide += "3. 按照頁面上的 Linux 安裝指南進行操作\n"
                guide += "Ubuntu 示例: `sudo apt-get update && sudo apt-get install -y dotnet-sdk-6.0`\n"
            
            return False, guide
    
    async def _check_cpp(self) -> Tuple[bool, str]:
        """檢查 C/C++ 環境"""
        if self.os_type == "windows":
            # 在 Windows 上檢查是否安裝了 MSVC 或 MinGW
            msvc_cl = shutil.which("cl")
            gcc = shutil.which("g++")
            
            if msvc_cl:
                return True, "Microsoft Visual C++ 已安裝並可用。"
            elif gcc:
                return True, "MinGW/GCC C++ 編譯器已安裝並可用。"
            else:
                guide = "### 安裝 C/C++ 開發環境 (Windows)\n\n"
                guide += "選項 1: 安裝 Visual Studio (推薦)\n"
                guide += "1. 訪問 Visual Studio 下載頁面: https://visualstudio.microsoft.com/downloads/\n"
                guide += "2. 下載 Visual Studio Community (免費版本)\n"
                guide += "3. 在安裝過程中，選擇 'C++ 桌面開發' 工作負載\n\n"
                guide += "選項 2: 安裝 MinGW-w64\n"
                guide += "1. 訪問 MinGW-w64 下載頁面: https://www.mingw-w64.org/downloads/\n"
                guide += "2. 下載並安裝 MinGW-w64\n"
                guide += "3. 將 MinGW bin 目錄添加到 PATH 環境變數\n"
                return False, guide
        else:
            # 在 macOS/Linux 上檢查 GCC 或 Clang
            gcc = shutil.which("g++")
            clang = shutil.which("clang++")
            
            if gcc:
                try:
                    process = await asyncio.create_subprocess_exec(
                        "g++", "--version",
                        stdout=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await process.communicate()
                    version_info = stdout.decode().strip().split("\n")[0]
                    return True, f"GCC C++ 編譯器已安裝並可用: {version_info}"
                except:
                    return True, "GCC C++ 編譯器已安裝並可用。"
            elif clang:
                try:
                    process = await asyncio.create_subprocess_exec(
                        "clang++", "--version",
                        stdout=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await process.communicate()
                    version_info = stdout.decode().strip().split("\n")[0]
                    return True, f"Clang C++ 編譯器已安裝並可用: {version_info}"
                except:
                    return True, "Clang C++ 編譯器已安裝並可用。"
            else:
                # 提供安裝指南
                guide = "### 安裝 C/C++ 開發環境\n\n"
                
                if self.os_type == "darwin":  # macOS
                    guide += "安裝 Xcode Command Line Tools:\n"
                    guide += "1. 開啟終端機\n"
                    guide += "2. 執行: `xcode-select --install`\n"
                    guide += "3. 按照彈出窗口的指示進行安裝\n"
                elif self.os_type == "linux":
                    guide += "使用套件管理器:\n"
                    guide += "- Ubuntu/Debian: `sudo apt update && sudo apt install build-essential`\n"
                    guide += "- Fedora: `sudo dnf install gcc-c++ make`\n"
                    guide += "- Arch Linux: `sudo pacman -S base-devel`\n"
                
                return False, guide
    
    # 其他檢查方法的實現...
    async def _check_php(self) -> Tuple[bool, str]:
        """檢查 PHP 環境"""
        php_path = shutil.which("php")
        
        if php_path:
            try:
                process = await asyncio.create_subprocess_exec(
                    "php", "--version",
                    stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                version_info = stdout.decode().strip().split("\n")[0]
                return True, f"PHP 已安裝並可用: {version_info}"
            except:
                return True, "PHP 已安裝並可用。"
        else:
            guide = "### 安裝 PHP\n\n"
            
            if self.os_type == "windows":
                guide += "選項 1: 使用獨立安裝程序\n"
                guide += "1. 訪問 PHP 官方網站: https://www.php.net/downloads.php\n"
                guide += "2. 下載 Windows 版本的 PHP\n"
                guide += "3. 解壓文件並將 PHP 目錄添加到 PATH 環境變數\n\n"
                guide += "選項 2: 使用 XAMPP 或 WAMP（包含 PHP、MySQL、Apache）\n"
                guide += "- XAMPP: https://www.apachefriends.org/\n"
                guide += "- WAMP: https://www.wampserver.com/\n"
            elif self.os_type == "darwin":  # macOS
                guide += "方法 1: 使用 Homebrew\n"
                guide += "1. 開啟終端機\n"
                guide += "2. 執行: `brew install php`\n\n"
                guide += "方法 2: 使用 MAMP (包含 PHP、MySQL、Apache)\n"
                guide += "- 訪問 MAMP 官方網站: https://www.mamp.info/\n"
            elif self.os_type == "linux":
                guide += "使用套件管理器:\n"
                guide += "- Ubuntu/Debian: `sudo apt update && sudo apt install php php-cli php-fpm`\n"
                guide += "- Fedora: `sudo dnf install php php-cli`\n"
                guide += "- Arch Linux: `sudo pacman -S php php-fpm`\n"
            
            return False, guide
    
    async def _check_ruby(self) -> Tuple[bool, str]:
        """檢查 Ruby 環境"""
        ruby_path = shutil.which("ruby")
        
        if ruby_path:
            try:
                process = await asyncio.create_subprocess_exec(
                    "ruby", "--version",
                    stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                version = stdout.decode().strip()
                return True, f"Ruby {version} 已安裝並可用。"
            except:
                return True, "Ruby 已安裝並可用。"
        else:
            guide = "### 安裝 Ruby\n\n"
            
            if self.os_type == "windows":
                guide += "1. 訪問 RubyInstaller 官方網站: https://rubyinstaller.org/\n"
                guide += "2. 下載並運行 RubyInstaller\n"
                guide += "3. 依照安裝嚮導的指示完成安裝\n"
            elif self.os_type == "darwin":  # macOS
                guide += "使用 Homebrew:\n"
                guide += "1. 開啟終端機\n"
                guide += "2. 執行: `brew install ruby`\n"
            elif self.os_type == "linux":
                guide += "使用套件管理器:\n"
                guide += "- Ubuntu/Debian: `sudo apt update && sudo apt install ruby-full`\n"
                guide += "- Fedora: `sudo dnf install ruby`\n"
                guide += "- Arch Linux: `sudo pacman -S ruby`\n"
            
            return False, guide
    
    async def _check_r(self) -> Tuple[bool, str]:
        """檢查 R 環境"""
        r_path = shutil.which("R") or shutil.which("Rscript")
        
        if r_path:
            try:
                process = await asyncio.create_subprocess_exec(
                    r_path, "--version",
                    stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                version_info = stdout.decode().strip()
                return True, f"R {version_info} 已安裝並可用。"
            except:
                return True, "R 已安裝並可用。"
        else:
            guide = "### 安裝 R\n\n"
            
            if self.os_type == "windows":
                guide += "1. 訪問 R 官方網站: https://cran.r-project.org/bin/windows/base/\n"
                guide += "2. 下載並運行 R 安裝程序\n"
                guide += "3. 依照安裝嚮導的指示完成安裝\n"
                guide += "4. 如需 IDE，建議安裝 RStudio: https://www.rstudio.com/products/rstudio/download/\n"
            elif self.os_type == "darwin":  # macOS
                guide += "方法 1: 使用官方安裝程序\n"
                guide += "1. 訪問 R 官方網站: https://cran.r-project.org/bin/macosx/\n"
                guide += "2. 下載並安裝 R for macOS\n\n"
                guide += "方法 2: 使用 Homebrew\n"
                guide += "1. 開啟終端機\n"
                guide += "2. 執行: `brew install r`\n"
            elif self.os_type == "linux":
                guide += "使用套件管理器:\n"
                guide += "- Ubuntu/Debian:\n"
                guide += "  ```\n"
                guide += "  sudo apt update\n"
                guide += "  sudo apt install r-base r-base-dev\n"
                guide += "  ```\n"
                guide += "- Fedora: `sudo dnf install R`\n"
                guide += "- Arch Linux: `sudo pacman -S r`\n"
            
            return False, guide