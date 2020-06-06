def accuracy(feature_set, labels, predict_fn):
    r = 0

    for features, label in zip(feature_set, labels):
        prediction = predict_fn(features)

        if prediction == label: 
            r += 1 
    
    return float(r) / len(feature_set)
    