import random
from collections import defaultdict

mapping = {'B-scientist': 'B-musicalartist', 'I-scientist': 'I-musicalartist',
           'B-university': 'B-band', 'I-university': 'I-band',
           'B-astronomicalobject': 'B-musicalinstrument', 'I-astronomicalobject': 'I-musicalinstrument',
           'B-protein': 'B-song', 'I-protein': 'I-song',
           'B-chemicalcompound': 'B-musicgenre', 'I-chemicalcompound': 'I-musicgenre',
           'B-academicjournal': 'B-album', 'I-academicjournal': 'I-album',
           'B-enzyme': 'B-song', 'I-enzyme': 'I-song',
           'B-discipline': 'B-musicgenre', 'I-discipline': 'I-musicgenre',
           'B-theory': 'B-musicgenre', 'I-theory': 'I-musicgenre',
           'B-chemicalelement': 'B-musicgenre', 'I-chemicalelement': 'I-musicgenre',
           'B-person': 'B-person', 'I-person': 'I-person',
           'B-misc': 'B-misc', 'I-misc': 'I-misc',
           'B-location': 'B-location', 'I-location': 'I-location',
           'B-organisation': 'B-organisation', 'I-organisation': 'I-organisation',
           'B-award': 'B-award', 'I-award': 'I-award',
           'B-event': 'B-event', 'I-event': 'I-event'}


def extract_entities(file_path):
    entities = defaultdict(lambda: defaultdict(list))
    with open(file_path, 'r', encoding="UTF-8") as f:
        entity = []
        for line in f:
            if line.strip():
                word, tag = line.strip().split()
                if tag.startswith('B-'):
                    if entity:
                        tag_key = entity[0][1]
                        length = len(entity)
                        entities[tag_key][length].append(entity)
                    entity = [(word, tag)]
                elif tag.startswith('I-'):
                    entity.append((word, tag))
                else:
                    if entity:
                        tag_key = entity[0][1]
                        length = len(entity)
                        entities[tag_key][length].append(entity)
                        entity = []
            else:
                if entity:
                    tag_key = entity[0][1]
                    length = len(entity)
                    entities[tag_key][length].append(entity)
                    entity = []
    return entities


def swap_entities(input_file, output_file, entities, token_map):
    with open(input_file, 'r', encoding="UTF-8") as in_file, open(output_file, 'w', encoding="UTF-8") as out_file:
        entity = []
        for line in in_file:
            if line.strip():
                word, tag = line.strip().split()

                # Check if it's the beginning of an entity
                if tag.startswith('B-'):
                    if entity:
                        # Get the new tag based on the mapping or use the existing tag
                        new_tag = token_map.get(entity[0][1], entity[0][1])

                        # Check if the new tag and entity length exist in the entities dictionary
                        if new_tag in entities and len(entity) in entities[new_tag]:
                            # Choose a random replacement for the entity
                            replacement = random.choice(entities[new_tag][len(entity)])
                            for i, (_, t) in enumerate(replacement):
                                out_file.write(f"{replacement[i][0]}\t{t}\n")
                        else:
                            # If no replacement is available, write the original entity
                            for w, t in entity:
                                out_file.write(f"{w}\t{token_map[t]}\n")
                        entity = [(word, tag)]  # Start a new entity
                    else:
                        entity = [(word, tag)]  # Start a new entity
                elif tag.startswith('I-'):
                    entity.append((word, tag))  # Continue building the entity
                else:
                    if entity:
                        new_tag = token_map.get(entity[0][1], entity[0][1])
                        if new_tag in entities and len(entity) in entities[new_tag]:
                            replacement = random.choice(entities[new_tag][len(entity)])
                            for i, (_, t) in enumerate(replacement):
                                out_file.write(f"{replacement[i][0]}\t{t}\n")
                        else:
                            for w, t in entity:
                                print(w, t)
                                out_file.write(f"{w}\t{token_map[t]}\n")
                        entity = []  # Reset the entity list
                    out_file.write(line)
            else:  # Empty line, end of a sentence
                if entity:
                    new_tag = token_map.get(entity[0][1], entity[0][1])
                    if new_tag in entities and len(entity) in entities[new_tag]:
                        replacement = random.choice(entities[new_tag][len(entity)])
                        for i, (_, t) in enumerate(replacement):
                            out_file.write(f"{replacement[i][0]}\t{t}\n")
                    else:
                        for w, t in entity:
                            out_file.write(f"{w}\t{token_map[t]}\n")
                    entity = []  # Reset the entity list
                out_file.write(line)


if __name__ == '__main__':
    music_entities = extract_entities("./datasets/music_dev.txt")
    swap_entities("./datasets/science_train.txt", "./datasets/entity_swapped_datasets/science_randomly_replaced.txt",
                  music_entities, mapping)
    print("done")
