import os
import re
import jieba
import thulac
import sys


def documentation(source_dir, target_dir):
    """
    Rename documents from subfolders and move them to a new folder.

    Parameters:
    source_dir (str): The source directory where the files are organized.
    target_dir (str): The target directory where the files will be outputted after organization.

    Returns:
    None: This function does not return anything but prints the results.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Initialization file path
    path_train = os.path.join(target_dir, 'classical_train.txt')
    path_test = os.path.join(target_dir, 'classical_test.txt')

    n_files = 0  # Number of total text files
    p = 0

    # Traverse the source directory
    for _, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):
                n_files += 1

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):
                p += 1
                target_file_path = path_train if p/n_files < 0.9 else path_test

                with open(target_file_path, 'a', encoding='utf-8') as text_output:
                    with open(os.path.join(root, file), 'r') as text_input:
                        for line in text_input:
                            text_output.write(line)

                progress_percentage = round(p / n_files * 100)
                print('\rWork done by: {}%'.format(progress_percentage), '▓' * (progress_percentage // 2), end='')
                sys.stdout.flush()

    print(" Files have been combined successfully.")


def preprocessing_keep_Chinese(text):
    cleaned_text = re.sub(r'[^\u4e00-\u9fff，。？！；：]', '', text)
    return cleaned_text


def classical_token(input_file, output_file):
    thu = thulac.thulac(seg_only=True)
    thu.cut_f(input_file, output_file)
    return


def preprocess_corpus(source_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt'):
                output_file_path = os.path.join(output_dir, file)

                print(f'Preprocessing: {file}')

                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    processed_content_non_token = preprocessing_keep_Chinese(content)

                with open(file_path, 'w') as w:
                    w.write(processed_content_non_token)

                if 'Modern' in root:
                    processed_content = jieba.cut(processed_content_non_token, cut_all=True)

                    # Convert to string
                    processed_content = ' '.join(processed_content)

                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(processed_content)
                else:
                    classical_token(file_path, output_file_path)

                print(f'Preprocessing completed: {file}')


path_modernRaw = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/raw/ModernChinese'
path_classicalRaw = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/raw/ClassicalChineseCombined'
path_modernPro = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/processed/modern'
path_classicalPro = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/processed/classical'

source_dir = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/raw/ClassicalChinese'
target_dir = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/raw/ClassicalChineseCombined'


def main():
    documentation(source_dir, target_dir)
    preprocess_corpus(path_modernRaw, path_modernPro)
    preprocess_corpus(path_classicalRaw, path_classicalPro)
    print("Preprocessing is complete.")


if __name__ == '__main__':
    main()
