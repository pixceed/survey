import os
import re
import subprocess

def image2text_ymtk(image_path, output_dir):

    # コマンドの構築
    command = [
        "yomitoku",       # コマンド名
        image_path,     # 入力ファイル
        "-f", "md",       # 出力フォーマット
        "-o", output_dir,  # 出力ディレクトリ
        "-v",             # 詳細モード
        "--figure",
    ]

    # コマンドを実行
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("\nCommand executed successfully!\n")

    except subprocess.CalledProcessError as e:
        # エラー発生時の処理
        print("An error occurred while executing the command.")

        # コマンドの構築
        command = [
            "yomitoku",       # コマンド名
            image_path,     # 入力ファイル
            "-f", "md",       # 出力フォーマット
            "-o", output_dir,  # 出力ディレクトリ
            "-v",             # 詳細モード
        ]
        # コマンドを実行
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)

            # 実行結果を表示
            print("\nCommand executed successfully!\n")

        except subprocess.CalledProcessError as e:
            # エラー発生時の処理
            print("An error occurred while executing the command.")

    # 出力ファイル名は "{output_dirの出力ディレクトリ名}_{image_pathの写真名}_p1.md" となる
    base_output_dir = os.path.basename(output_dir)
    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_output_dir}_{base_image_name}_p1.md"
    output_filepath = os.path.join(output_dir, output_filename)

     # 出力ファイルのテキストを読み取り、返す
    try:
        with open(output_filepath, "r", encoding="utf-8") as f:
            file_text = f.read()
            file_text = file_text.replace("<br>", "").replace("\\", "")

    except Exception as read_error:
        print("An error occurred while reading the output file:", read_error)
        file_text = None

    # figures ディレクトリ内の画像ファイルを検索
    figures_dir = os.path.join(output_dir, "figures")
    image_paths = []
    if os.path.exists(figures_dir):
        # 画像ファイルのパターン:
        # {output_dirの出力ディレクトリ名}_{image_pathの写真名}_p1_figure_[n].png
        pattern = re.compile(re.escape(f"{base_output_dir}_{base_image_name}_p1_figure_") + r"(\d+)\.png")
        for filename in os.listdir(figures_dir):
            if pattern.match(filename):
                image_paths.append(os.path.join(figures_dir, filename))
        # n の値でソート（0から昇順）
        def extract_num(filepath):
            match = pattern.match(os.path.basename(filepath))
            return int(match.group(1)) if match else -1
        image_paths.sort(key=extract_num)

    return file_text, image_paths