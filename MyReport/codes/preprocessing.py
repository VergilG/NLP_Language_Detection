import os
import jieba


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

                    # Save the processed text
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(processed_content)
                else:
                    classical_token(file_path, output_file_path)

                print(f'Preprocessing completed: {file}')


path_modernRaw = '/path/to/ModernChinese'
path_classicalRaw = '/path/to/ClassicalChinese'
path_modernPro = '/path/to/processed/modern'
path_classicalPro = '/path/to/processed/classical'


def main():
    preprocess_corpus(path_modernRaw, path_modernPro)
    preprocess_corpus(path_classicalRaw, path_classicalPro)


if __name__ == '__main__':
    main()
