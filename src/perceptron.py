import random
from evaluate import accuracy

class Perceptron:
    def __init__(self):
        self.is_trained = False
        self.bias = 0

    def fit(self, training_set, labels, epochs=10, validation_fn=accuracy):
        self.is_trained = True
        self.weights = [1 for i in range(len(training_set[0]))]

        trainings_data = zip(training_set, labels)

        for i in range(epochs):
            random.shuffle(trainings_data)

            for features, label in trainings_data:
                self.update(features, label)
            
            traings_score=validation_fn(training_set, labels, self.predict)

            print('epoch: {}, score: {}'.format(i, traings_score))


        return self

    def update(self, features, label):
        predicition = self.predict(features)

        if predicition and not label:
            error = 1
        elif not predicition and label:
            error = -1
        else:
            return

        self.weights = [w - x * error for w, x in zip(self.weights, features)]
        

        


    def predict(self, features):
        score = sum([x * y for x, y in zip(features, self.weights)])

        if score > self.bias:
            return 1
        
        return 0


if __name__ == "__main__":
    print("test")
    with open("data/simple/points_linear_origin.txt") as file:
        x = filter(lambda x: x != '', file.read().split('\n'))
        y = [[int(b) for b in a.split(",")] for a in x]
        features = [a[:2] for a in y]
        labels = [a[2] for a in y] 

        # print(features, labels)

        p = Perceptron().fit(features, labels)
        print(p.predict([0,1]))

        #with open("data/simple/points_linear.txt", "w+") as f:
         #   for i in [",".join([str(a[0]), str(a[1] + 5), str(a[2])]) + "\n" for a in y]:
          #      print(i)
           #     f.write(i)

    