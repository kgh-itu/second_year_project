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
                      'B-chemicalelement': 'musicgenre.txt', 'I-chemicalelement': 'musicgenre.txt',
                      'B-location': 'location.txt', "I-location": 'location.txt'}

with open("entities/science_train.txt", "r") as infile, open("tmp.txt", "w") as outfile:
    for line in infile:
        if not line.strip():
            outfile.write("\n")
            continue

        word, label = line.strip().split("\t")

        if label in entity_to_file_map:
            with open("entities/" + entity_to_file_map[label], "r") as f:
                words = f.read().splitlines()
                new_word = random.choice(words)
                new_sentence = word.replace(word, new_word)
                outfile.write(new_sentence + "\t" + label + "\n")
        else:
            outfile.write(word + "\t" + label + "\n")