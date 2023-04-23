import codecs


def read_data(file_name):
    data = []
    current_words = []
    current_tags = []

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue  # skip comments
            tok = line.split('\t')
            word = tok[0]
            tag = tok[1]

            current_words.append(word)
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    return data


if __name__ == "__main__":
    music_train = read_data("dataset/music_train.txt")
    music_test = read_data("dataset/music_test.txt")
    music_dev = read_data("dataset/music_dev.txt")

    science_train = read_data("dataset/science_train.txt")
    science_test = read_data("dataset/science_test.txt")
    science_dev = read_data("dataset/science_dev.txt")