import random
import codecs

entity_to_file_map = {'B-scientist': 'musicartist.txt', 'I-scientist': 'musicartist.txt',
                      'B-university': 'band.txt', 'I-university': 'band.txt',
                      'B-astronomicalobject': 'musicinstrument.txt', 'I-astronomicalobject': 'musicinstrument.txt',
                      'B-protein': 'song.txt', 'I-protein': 'song.txt',
                      'B-chemicalcompound': 'musicgenre.txt', 'I-chemicalcompound': 'musicgenre.txt',
                      'B-academicjournal': 'album.txt', 'I-academicjournal': 'album.txt',
                      'B-enzyme': 'song.txt', 'I-enzyme': 'song.txt',
                      'B-discipline': 'musicgenre.txt', 'I-discipline': 'musicgenre.txt',
                      'B-theory': 'musicgenre.txt', 'I-theory': 'musicgenre.txt',
                      'B-chemicalelement': 'musicgenre.txt', 'I-chemicalelement': 'musicgenre.txt'}

for line in codecs.open("entities/science_train.txt", encoding='utf-8'):
    if not line.strip():
        continue

    word, label = line.strip().split("\t")

    if label in entity_to_file_map:
        with open("entities/" + entity_to_file_map[label], "r") as f:
            words = f.read().splitlines()
            new_word = random.choice(words)
            new_sentence = word.replace(word, new_word)

            with open("tmp.txt", "a") as outfile:
                outfile.write(new_sentence + "\t" + label + "\n")
    else:
        with open("tmp.txt", "a") as outfile:
            outfile.write(word + "\t" + label + "\n")