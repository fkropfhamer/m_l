def accuracy(feature_set, labels, predict_fn):
    r = 0

    for features, label in zip(feature_set, labels):
        prediction = predict_fn(features)

        if prediction == label:
            r += 1

    return float(r) / len(feature_set)


def precision(feature_set, labels, predict_fn):
    return accuracy(feature_set, labels, predict_fn)


def recall(feature_set, labels, predict_fn, correct_label=1):
    r = 0

    for features, label in zip(feature_set, labels):
        prediction = predict_fn(features)

        if prediction == label:
            r += 1

    number_of_correct_labels = len(
        filter(lambda x: x == correct_label, labels))

    return float(r) / number_of_correct_labels


def f1_score(feature_set, labels, predict_fn, correct_label=1):
    p = precision(feature_set, labels, predict_fn)
    r = recall(feature_set, labels, predict_fn, correct_label=correct_label)

    return 2 * (p * r) / (p + r)
