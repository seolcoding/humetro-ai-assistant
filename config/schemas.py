"""
Data schemas for Seoul Traffic News Crawler

Comprehensive Pydantic models for:
- Page metadata with tree structure
- Attached documents (PDF, HWP, XLSX, etc.)
- Link contexts for Knowledge Graph preparation
- Breadcrumb navigation
"""

from typing import List, Dict, Optional, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class PageType(str, Enum):
    """Classification of page types in the website hierarchy"""
    ROOT = "root"
    CATEGORY = "category"
    LIST = "list"
    ARTICLE = "article"
    ATTACHMENT = "attachment"
    OTHER = "other"


class AttachmentType(str, Enum):
    """Types of attached documents"""
    PDF = "pdf"
    HWP = "hwp"
    XLSX = "xlsx"
    XLS = "xls"
    DOCX = "docx"
    DOC = "doc"
    OTHER = "other"


class AttachmentSource(str, Enum):
    """How the attachment was accessed"""
    POPUP_WINDOW = "popup_window"
    DIRECT_LINK = "direct_link"
    IFRAME = "iframe"
    JAVASCRIPT = "javascript"


class AttachedDocument(BaseModel):
    """Comprehensive metadata for attached documents"""

    attachment_id: str
    attachment_type: AttachmentType
    original_filename: str
    file_size: int
    source_type: AttachmentSource
    source_url: str
    trigger_element: Optional[str] = None
    trigger_context: Optional[str] = None
    popup_window_title: Optional[str] = None
    popup_window_url: Optional[str] = None
    extracted_text: str
    extracted_tables: List[Dict] = Field(default_factory=list)
    extracted_images: List[str] = Field(default_factory=list)
    page_count: Optional[int] = None
    sheet_names: Optional[List[str]] = None
    local_path: str
    markdown_path: Optional[str] = None
    extraction_method: Literal['pypdf', 'pdfplumber', 'openpyxl', 'olefile', 'hwp5txt', 'python-docx', 'manual']
    extraction_success: bool
    word_count: int
    entities_preview: List[str] = Field(default_factory=list)
    named_entities: Dict[str, List[str]] = Field(default_factory=dict)


class BreadcrumbItem(BaseModel):
    """Single item in breadcrumb navigation path"""
    url: str
    title: str
    depth: int
    page_type: PageType


class LinkContext(BaseModel):
    """Rich context around a link for Knowledge Graph generation"""
    target_url: str
    anchor_text: str
    surrounding_text: str
    parent_element: str
    link_position: int
    is_navigation: bool


class PageMetadata(BaseModel):
    """Comprehensive metadata for a crawled page"""

    url: str
    title: str
    page_type: PageType
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    parent_url: Optional[str] = None
    depth: int
    breadcrumb: List[BreadcrumbItem] = Field(default_factory=list)
    siblings: List[str] = Field(default_factory=list)
    outgoing_links: List[str] = Field(default_factory=list)
    incoming_links: List[str] = Field(default_factory=list)
    link_contexts: List[LinkContext] = Field(default_factory=list)
    word_count: int
    entities_preview: List[str] = Field(default_factory=list)
    named_entities: Dict[str, List[str]] = Field(default_factory=dict)
    attached_documents: List[AttachedDocument] = Field(default_factory=list)
    has_attachments: bool = False
    attachment_types: List[AttachmentType] = Field(default_factory=list)
    total_attachment_size: int = 0
    attachment_count: int = 0

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


__all__ = [
    'PageType', 'AttachmentType', 'AttachmentSource',
    'AttachedDocument', 'BreadcrumbItem', 'LinkContext', 'PageMetadata'
]
