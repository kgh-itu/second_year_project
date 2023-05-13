
def readBIO(path):
    ents = []
    curEnts = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            curEnts = []
        else:
            word, tag = line.split()
            curEnts.append(tag)
    return ents


def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg + 1, len(tags)):
                if tags[beg][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans


def getInstanceScores(predPath, goldPath):
    goldEnts = readBIO(goldPath)
    predEnts = readBIO(predPath)
    entScores = []
    tp = 0
    fp = 0
    fn = 0
    entities = 0
    for goldEnt, predEnt in zip(goldEnts, predEnts):
        entities += 1
        goldSpans = toSpans(goldEnt)
        predSpans = toSpans(predEnt)
        overlap = len(goldSpans.intersection(predSpans))
        tp += overlap
        fp += len(predSpans) - overlap
        fn += len(goldSpans) - overlap

    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    print(f'Precision: {prec} Recall: {rec} Entities: {entities}')
    return f1


if __name__ == "__main__":
    score = getInstanceScores("predictions/random_music_test_preds.txt", "datasets/music_test.txt")
    print(score)