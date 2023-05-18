import sys

from seqeval.metrics import f1_score, classification_report


def readBIO(path):
    ents = []
    curEnts = []
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            curEnts = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            curEnts.append(line.split('\t')[1])
    return ents


def score(y_true, y_pred):
    score = f1_score(y_true, y_pred)
    print(' - f1: {:04.2f}'.format(score * 100))
    print(classification_report(y_true, y_pred, digits=4))
    return score


if __name__ == "__main__":
    true, preds = readBIO("datasets/music_test.txt"), readBIO("predictions/science_baseline_replaced_500_preds0.txt")
    print(score(true, preds))
