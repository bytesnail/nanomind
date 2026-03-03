"""项目级常量定义。

集中管理所有跨模块共享的常量，避免重复定义。
"""

from __future__ import annotations

# Tokenizer 特殊 tokens
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
BASE_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
THINK_TOKENS = ["<think>", "</think>"]

# GitHub Code 语言扩展名映射
LANGUAGE_EXTENSIONS: dict[str, str] = {
    # C
    ".c": "c",
    ".h": "c",
    # C++
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hxx": "cpp",
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    # Rust
    ".rs": "rust",
    # HTML
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    # CSS
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".less": "css",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    ".mkd": "markdown",
    # JSON
    ".json": "json",
    ".jsonc": "json",
    ".jsonl": "json",
    # XML
    ".xml": "xml",
    ".xsl": "xml",
    ".xslt": "xml",
    ".svg": "xml",
    ".wsdl": "xml",
    # TOML
    ".toml": "toml",
}

ALLOWED_LANGUAGE_EXTENSIONS: frozenset[str] = frozenset(LANGUAGE_EXTENSIONS)
