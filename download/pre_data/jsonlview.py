import json

def split_json_file(input_file_path, output_file_path, num_lines=5000):
    """
    切分JSON文件的前指定行数。

    Args:
        input_file_path (str): 输入的JSON文件路径。
        output_file_path (str): 输出的JSON文件路径。
        num_lines (int): 需要切分的前行数。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            # 对于可能的不同JSON结构进行处理
            # 重要的是按行读取并计数
            line_count = 0
            for line in infile:
                if line_count < num_lines:
                    outfile.write(line)
                    line_count += 1
                else:
                    # 已经达到指定的行数，停止写入
                    break

        print(f"成功将文件 '{input_file_path}' 的前 {num_lines} 行写入到 '{output_file_path}' 中。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file_path}'。")
    except Exception as e:
        print(f"发生错误：{e}")

# 请修改这里的路径和文件名
input_file = './download/pre_data/mobvoi_seq_monkey_general_open_corpus.jsonl'  # 替换成您的JSON文件路径
output_file = 'output_first_5000_lines.json' # 替换成您希望输出的文件路径

# 调用函数进行切分
split_json_file(input_file, output_file)
