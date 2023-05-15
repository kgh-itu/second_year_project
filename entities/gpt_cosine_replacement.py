import numpy as np
import spacy
import os
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


class BatchVectorizer:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, texts):
        return np.array([doc.vector for doc in self.nlp.pipe(texts)])


def extract_entities(directory):
    entities = defaultdict(lambda: defaultdict(list))
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
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


def cosine_similarity(v1, v2):
    return np.matmul(v1, v2.T) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))


nlp = spacy.load('en_core_web_sm')
vectorizer = BatchVectorizer(nlp)


def get_entity_embedding(entity):
    text = " ".join([word for word, _ in entity])
    return vectorizer([text])[0]


def find_most_similar_entity(entity, candidates):
    if not candidates:
        return None

    entity_embedding = np.mean([nlp(w).vector for w, _ in entity], axis=0)
    candidate_embeddings = np.array([get_entity_embedding(candidate) for candidate in candidates])

    similarities = cosine_similarity(entity_embedding.reshape(1, -1), candidate_embeddings.reshape(len(candidates), -1))
    most_similar_index = np.argmax(similarities)

    return candidates[most_similar_index]


def swap_entities(input_file, output_file, entities, token_map):
    with open(input_file, 'r', encoding='utf-8') as in_file, open(output_file, 'w', encoding='utf-8') as out_file:
        entity = []
        for line in in_file:
            if line.strip():
                word, tag = line.strip().split()
                if tag.startswith('B-'):
                    if entity:
                        new_tag = token_map.get(entity[0][1], entity[0][1])
                        if new_tag in entities and len(entity) in entities[new_tag]:
                            replacement = find_most_similar_entity(entity, entities[new_tag][len(entity)])
                            if replacement:
                                for i, (_, t) in enumerate(replacement):
                                    out_file.write(f"{replacement[i][0]}\t{t}\n")
                        else:
                            for w, t in entity:
                                out_file.write(f"{w}\t{token_map.get(t, t)}\n")
                        entity = [(word, tag)]
                    else:
                        entity = [(word, tag)]
                elif tag.startswith('I-'):
                    entity.append((word, tag))
                else:
                    if entity:
                        new_tag = token_map.get(entity[0][1], entity[0][1])
                        if new_tag in entities and len(entity) in entities[new_tag]:
                            replacement = find_most_similar_entity(entity, entities[new_tag][len(entity)])
                            if replacement:
                                for i, (_, t) in enumerate(replacement):
                                    out_file.write(f"{replacement[i][0]}\t{t}\n")
                        else:
                            for w, t in entity:
                                out_file.write(f"{w}\t{token_map.get(t, t)}\n")
                        entity = []
                    out_file.write(line)
            else:
                if entity:
                    new_tag = token_map.get(entity[0][1], entity[0][1])
                    if new_tag in entities and len(entity) in entities[new_tag]:
                        replacement = find_most_similar_entity(entity, entities[new_tag][len(entity)])
                        if replacement:
                            for i, (_, t) in enumerate(replacement):
                                out_file.write(f"{replacement[i][0]}\t{t}\n")
                        else:
                            for w, t in entity:
                                out_file.write(f"{w}\t{token_map.get(t, t)}\n")
                    entity = []
                out_file.write(line)


if __name__ == '__main__':
    music_entities = extract_entities("./reformatted_generated_entities/")
    swap_entities("./datasets/science_train.txt", "./datasets/entity_swapped_datasets/science_gpt_cosine_replaced.txt",
                  music_entities, mapping)