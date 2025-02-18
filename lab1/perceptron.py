import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Perceptron:
     # Inicjalizator, ustawiający atrybut self.w oraz self.b jako wektor losowych wag, n ilość sygnałów wejściowych (bias)
    def __init__(self, n, bias=True):
        self.w = np.random.rand(n)
        self.b = 1.0 if bias else 0.0

    # Metoda obliczająca odpowiedz modelu dla zadanego sygnału wejściowego x=[x1,x2,...,xN]
    def predict(self, x):
        activation = np.dot(self.w, x) + self.b
        return 1 if activation >= 0 else 0

    # Metoda uczenia według reguły perceptronu, xx - zbiór danych uczących, d - odpowiedzi,
    # eta - współczynnik uczenia,
    # tol - tolerancja (czyli jak duży błąd jesteśmy w stanie zaakceptować)
    def train(self, xx, d, eta, tol):
        t = 0
        while True:
            errors = 0
            for x, desired in zip(xx, d):
                prediction = self.predict(x)


                if prediction != desired:
                    error = desired - prediction

                    if prediction == 0 and desired == 1:
                        self.w += eta * x
                        self.b += eta
                    elif prediction == 1 and desired == 0:
                        self.w -= eta * x
                        self.b -= eta

                    errors += 1
            if errors <= tol:
                break
            t += 1
        print(f"Algorytm zatrzymał się po {t} epokach")

    # Metoda obliczająca błąd dla danych testowych xx
    # zwraca błąd oraz wektor odpowiedzi perceptronu dla danych testowych
    def evaluate_test(self, xx, d):
        errors = 0
        predictions = []
        for x, desired in zip(xx, d):
            prediction = self.predict(x)
            predictions.append(prediction)
            errors += abs(desired - prediction)
        return errors, predictions

if __name__ == "__main__":
    data = pd.read_csv("lab1/2D.csv", skiprows=1, delimiter=';', names=['X1', 'X2', 'L'], decimal=',', dtype={"X1": float, "X2": float, "L": int})
    xx = data[['X1', 'X2']].values
    d = data['L']

    # pick up random 80% of values for training and 20% for testing
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    p = Perceptron(2)
    p.train(train_data[['X1', 'X2']].values, train_data['L'], 0.1, 0)
    errors, predictions = p.evaluate_test(test_data[['X1', 'X2']].values, test_data['L'])

    print(f"Błąd: {errors}")
    print(f"Odpowiedzi: {predictions}")

    # plot the data
    plt.scatter(train_data['X1'], train_data['X2'], c=train_data['L'], cmap='viridis')
    plt.savefig('lab1/plot.png')

    

