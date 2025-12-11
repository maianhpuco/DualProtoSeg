#!/usr/bin/env python3
"""
Strip Python comments (tokens of type COMMENT) from files.

This preserves encoding declarations and docstrings (STRING tokens).
It removes line and inline comments starting with '#'.

Usage: python3 scripts/strip_comments.py [root_dir]
"""
import sys
import pathlib
import tokenize


def strip_comments_in_file(path: pathlib.Path) -> bool:
    """Return True if file was modified."""
    try:
        with path.open('rb') as f:
            tokens = list(tokenize.tokenize(f.readline))
    except Exception:
        return False

    # Rebuild token stream without COMMENT tokens
    new_tokens = []
    for t in tokens:
        tok_type = t.type
        tok_string = t.string
        if tok_type == tokenize.COMMENT:
            # skip comment
            continue
        new_tokens.append((tok_type, tok_string))

    new_src = tokenize.untokenize(new_tokens)

    # Read original as text to compare
    try:
        orig_text = path.read_text(encoding='utf-8')
    except Exception:
        # fallback: don't modify binary/non-utf8 files
        return False

    if new_src == orig_text:
        return False

    # Write back
    path.write_text(new_src, encoding='utf-8')
    return True


def main(root_dir: str):
    root = pathlib.Path(root_dir)
    py_files = list(root.rglob('*.py'))
    modified = []
    for p in py_files:
        # Skip virtualenvs or hidden folders common patterns
        parts = set(p.parts)
        if '.venv' in parts or 'venv' in parts or 'site-packages' in parts:
            continue
        if strip_comments_in_file(p):
            modified.append(str(p))

    print(f"Processed {len(py_files)} .py files; modified: {len(modified)}")
    for m in modified:
        print(m)


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(root)
