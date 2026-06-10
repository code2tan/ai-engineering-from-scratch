#!/usr/bin/env python3
"""
从 en-zh.md 双语文档中提取中文内容，输出到 phase-zh 目录。

用法:
  # 处理单个文件
  python scripts/extract-zh.py phases/03-deep-learning-core/01-the-perceptron/docs/en-zh.md

  # 批量处理某阶段下所有文件
  python scripts/extract-zh.py phases/03-deep-learning-core/

  # 指定输出目录
  python scripts/extract-zh.py phases/03-deep-learning-core/ -o phases/03-deep-learning-core-zh/

  # 仅列出将处理的文件，不实际执行
  python scripts/extract-zh.py phases/03-deep-learning-core/ --dry-run
"""

import argparse
import glob
import os
import re
import sys


def extract_chinese_content(text: str) -> str:
    """
    从 en-zh.md 内容中提取中文部分。

    保留规则:
    - 含中文汉字或中文标点的行
    - 代码块（```）完整保留
    - 数学公式块（$$）完整保留
    - Markdown 标题行（#）
    - 含中文的表格行
    - 空行和分隔线
    - 跳过纯英文段落
    """
    lines = text.split("\n")
    result = []
    in_code_block = False
    in_math_block = False

    for line in lines:
        stripped = line.strip()

        # 代码块开关
        if stripped.startswith("```") and not stripped.startswith("```math"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        # 数学公式块开关
        if stripped.startswith("$$"):
            in_math_block = not in_math_block
            result.append(line)
            continue

        # 代码/公式块内部：完整保留
        if in_code_block or in_math_block:
            result.append(line)
            continue

        has_chinese = bool(re.search(r"[一-鿿]", line))
        has_chinese_punct = bool(re.search(r"[：；，。？、（）【】「」『』]", line))

        # 含中文的行保留
        if has_chinese or has_chinese_punct:
            result.append(line)
        # 空行
        elif stripped == "":
            result.append(line)
        # 分隔线
        elif stripped.startswith("---"):
            result.append(line)
        # 标题
        elif stripped.startswith("#"):
            result.append(line)
        # 含中文的表格行
        elif stripped.startswith("|") and stripped.endswith("|"):
            if has_chinese or has_chinese_punct:
                result.append(line)

    output = "\n".join(result)
    # 合并连续过多的空行
    output = re.sub(r"\n{4,}", "\n\n\n", output)
    return output


def process_file(src_path: str, out_dir: str, dry_run: bool = False) -> bool:
    """
    处理单个 en-zh.md 文件，输出到指定目录。

    输出文件名规则: 从源路径提取编号-主题，如 01-the-perceptron.md
    """
    # 从路径中推断输出文件名
    # 期望路径: .../XX-XXX/docs/en-zh.md
    parts = src_path.replace(os.sep, "/").split("/")
    try:
        # 找父目录名，如 "01-the-perceptron"
        parent_dir = parts[-3]
    except IndexError:
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(src_path)))

    out_name = parent_dir + ".md"
    out_path = os.path.join(out_dir, out_name)

    if dry_run:
        print(f"  [DRY-RUN] {src_path}  ->  {out_path}")
        return True

    try:
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  [ERR] 文件不存在: {src_path}", file=sys.stderr)
        return False

    extracted = extract_chinese_content(content)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(extracted)

    # 统计中文保留率
    src_cn = len(re.findall(r"[一-鿿]", content))
    out_cn = len(re.findall(r"[一-鿿]", extracted))
    ratio = out_cn / src_cn * 100 if src_cn > 0 else 0

    status = "✓" if ratio >= 100 else "⚠"
    print(f"  {status} {out_name}: {out_cn}/{src_cn} = {ratio:.1f}% ({len(extracted)} chars)")
    return True


def find_en_zh_files(root: str) -> list[str]:
    """递归查找所有 en-zh.md 文件。"""
    path = os.path.join(root, "**", "en-zh.md")
    return sorted(glob.glob(path, recursive=True))


def main():
    parser = argparse.ArgumentParser(
        description="从 en-zh.md 双语文档中提取中文内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  %(prog)s phases/03-deep-learning-core/01-the-perceptron/docs/en-zh.md

  # 批量处理某个阶段所有子目录
  %(prog)s phases/03-deep-learning-core/

  # 指定输出目录
  %(prog)s phases/03-deep-learning-core/ -o phases/03-deep-learning-core-zh/

  # 仅列出要处理的文件
  %(prog)s phases/03-deep-learning-core/ --dry-run
        """,
    )
    parser.add_argument("target", help="en-zh.md 文件路径，或包含 en-zh.md 的目录")
    parser.add_argument("-o", "--output", default=None,
                        help="输出目录（默认: 在 target 同级创建 {阶段名}-zh）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出将处理的文件，不实际执行")

    args = parser.parse_args()

    # 确定输入文件列表
    if os.path.isfile(args.target):
        files = [args.target]
    elif os.path.isdir(args.target):
        files = find_en_zh_files(args.target)
        if not files:
            print(f"[ERR] 在 {args.target} 下未找到任何 en-zh.md 文件", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[ERR] 路径不存在: {args.target}", file=sys.stderr)
        sys.exit(1)

    # 确定输出目录
    if args.output:
        out_dir = args.output
    else:
        # 默认: target 父目录名去掉最后一个后缀，加 -zh
        base = os.path.dirname(os.path.abspath(args.target.rstrip("/")))
        base_name = os.path.basename(base)
        out_dir = os.path.join(os.path.dirname(base), base_name + "-zh")

    print(f"输入: {args.target}")
    print(f"输出: {out_dir}")
    print(f"文件数: {len(files)}")
    print()

    success = 0
    for f in files:
        if process_file(f, out_dir, args.dry_run):
            success += 1

    if not args.dry_run:
        print(f"\n完成: {success}/{len(files)} 个文件处理成功")
    else:
        print(f"\nDRY-RUN: 共 {len(files)} 个文件待处理")


if __name__ == "__main__":
    main()
