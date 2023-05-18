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
    # print(' - f1: {:04.2f}'.format(score * 100))
    # print(classification_report(y_true, y_pred, digits=4))
    return score


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    files_randomly = [f"science_randomly_replaced_{x}_preds{y}.txt" for x in [50, 100, 150, 200, 250, 300, 350] for y in
                      [0, 1, 2, 3, 4]]
    tmp = [50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 150, 150, 150, 150, 150, 200, 200, 200, 200, 200, 250, 250, 250,
           250, 250, 300, 300, 300, 300, 300, 350, 350, 350, 350, 350, 400, 400, 450, 450, 500, 500]

    files_randomly.append("science_randomly_replaced_400_preds0.txt")
    files_randomly.append("science_randomly_replaced_400_preds1.txt")
    files_randomly.append("science_randomly_replaced_450_preds0.txt")
    files_randomly.append("science_randomly_replaced_450_preds1.txt")
    files_randomly.append("science_randomly_replaced_500_preds0.txt")
    files_randomly.append("science_randomly_replaced_500_preds1.txt")
    files_randomly = list(zip(files_randomly, tmp))

    files_cosine = [f"science_cosine_replaced_{x}_preds{y}.txt" for x in
                    [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] for y in [0, 1]]
    tmp1 = [50, 50, 100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400, 450, 450, 500, 500]
    files_cosine = list(zip(files_cosine, tmp1))

    files_baseline = [f"science_baseline_replaced_{x}_preds{y}.txt" for x in
                      [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] for y in [0, 1]]
    tmp2 = [50, 50, 100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400, 450, 450, 500, 500]
    files_baseline = list(zip(files_baseline, tmp1))

    all_f1_randomly = {50: [], 100: [], 150: [], 200: [], 250: [], 300: [], 350: [], 400: [], 450: [], 500: []}
    all_f1_cosine = {50: [], 100: [], 150: [], 200: [], 250: [], 300: [], 350: [], 400: [], 450: [], 500: []}
    all_f1_baseline = {50: [], 100: [], 150: [], 200: [], 250: [], 300: [], 350: [], 400: [], 450: [], 500: []}

    true = readBIO("datasets/music_test.txt")
    for file, training_samples in files_randomly:
        preds = readBIO(f"predictions/{file}")
        all_f1_randomly[training_samples].append(score(true, preds))

    for file, training_samples in files_cosine:
        preds = readBIO(f"predictions/{file}")
        all_f1_cosine[training_samples].append(score(true, preds))

    for file, training_samples in files_baseline:
        preds = readBIO(f"predictions/{file}")
        all_f1_baseline[training_samples].append(score(true, preds))

    all_f1_randomly = {k: np.mean(v) for k, v in all_f1_randomly.items()}
    all_f1_cosine = {k: np.mean(v) for k, v in all_f1_cosine.items()}
    all_f1_baseline = {k: np.mean(v) for k, v in all_f1_baseline.items()}
    #
    fig, ax = plt.subplots()
    ax.scatter(all_f1_randomly.keys(), all_f1_randomly.values(), label='Randomly Replaced')
    ax.plot(all_f1_randomly.keys(), all_f1_randomly.values())
    ax.scatter(all_f1_cosine.keys(), all_f1_cosine.values(), label='Cosine Replaced')
    ax.plot(all_f1_cosine.keys(), all_f1_cosine.values())
    ax.scatter(all_f1_baseline.keys(), all_f1_baseline.values(), label='Baseline')
    ax.plot(all_f1_baseline.keys(), all_f1_baseline.values())
    ax.legend()
    ax.set_title("F1-Score based on size of training data")
    ax.set_xlabel("Number of training sentences")
    ax.set_ylabel("F1-Score")
    plt.show()
