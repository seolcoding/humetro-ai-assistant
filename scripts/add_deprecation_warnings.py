#!/usr/bin/env python3
"""Add deprecation warnings to deprecated files"""

import sys
from pathlib import Path

DEPRECATION_HEADER = '''"""
⚠️ DEPRECATED - DO NOT USE ⚠️
================================

이 파일은 더 이상 사용되지 않습니다.

Deprecated Date: 2025-11-11
Reason: 실제 Q&A 데이터 사용으로 질문 생성 불필요

대체 사용:
  python src/rag_pipeline/unified_benchmark_v4_real_qa.py
  from src.data_loader.dasan_qa_sampler import DasanQASampler

자세한 내용: src/rag_pipeline/deprecated/README.md

================================
"""

import warnings
warnings.warn(
    "⚠️ DEPRECATED: This file is no longer maintained. "
    "Use 'unified_benchmark_v4_real_qa.py' or 'DasanQASampler' instead. "
    "See deprecated/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

'''

def add_deprecation_warning(file_path: Path):
    """Add deprecation warning to a Python file"""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    try:
        content = file_path.read_text(encoding='utf-8')

        # Check if already has deprecation warning
        if "DEPRECATED - DO NOT USE" in content:
            print(f"✅ Already deprecated: {file_path.name}")
            return True

        # Find the first docstring
        lines = content.split('\n')
        shebang = lines[0] if lines[0].startswith('#!') else None

        # Find first triple quote
        docstring_start = -1
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                docstring_start = i
                break

        if docstring_start == -1:
            print(f"⚠️ No docstring found: {file_path.name}")
            return False

        # Insert deprecation header after shebang
        if shebang:
            new_content = shebang + '\n' + DEPRECATION_HEADER + '\n'.join(lines[1:])
        else:
            new_content = DEPRECATION_HEADER + content

        file_path.write_text(new_content, encoding='utf-8')
        print(f"✅ Added warning: {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ Error: {file_path.name} - {e}")
        return False


def main():
    deprecated_dir = Path("src/rag_pipeline/deprecated")

    if not deprecated_dir.exists():
        print(f"❌ Deprecated directory not found: {deprecated_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Adding Deprecation Warnings to Files")
    print("=" * 60)

    # Find all Python files
    py_files = list(deprecated_dir.rglob("*.py"))

    print(f"\nFound {len(py_files)} Python files\n")

    success_count = 0
    for py_file in py_files:
        if add_deprecation_warning(py_file):
            success_count += 1

    print()
    print("=" * 60)
    print(f"✅ Updated {success_count}/{len(py_files)} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
