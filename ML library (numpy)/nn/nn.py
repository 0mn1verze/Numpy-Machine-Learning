from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from nn.layers import Layer
from nn.optimisers import Optimiser
from nn.cost import Cost
from nn.decay import Constant

class NN:
    def __init__(self, layers: Layer=None):
        """Sequential Neural Network

        Args:
            layers (Layer, optional): Layers of the model. Defaults to None.
        """
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.compiled = False
    
    def add(self, layer: Layer):
        """Add layer to model

        Args:
            layer (Layer): Next layer of the model
        """
        self.layers.append(layer)

    def Input(self, in_shape: tuple[int] | int):
        """Input Layer

        Args:
            i (tuple[int] | int): input shape
        """
        self.in_shape = in_shape

    def compile(self, cost_func: Cost, optimiser: Optimiser):
        """Compile Model

        Args:
            cost_func (Cost): Cost function for training
            optimiser (Optimiser): Optimiser for training
        """
        in_shape = self.in_shape
        self.optimiser = optimiser
        for layer in self.layers:
            name = layer.__class__.__name__
            match name:
                case "Pool2D" | "Flatten":
                    layer.setup(in_shape)
                case _:
                    layer.setup(in_shape, optimiser)
                
            in_shape = layer.out_shape

        self.cost_func = cost_func
        self.compiled = True

    def summary(self):
        """Summarises the model

        Raises:
            ValueError: Cannot print summary when the model has not been compiled
        """
        if not self.compiled:
            raise ValueError("Model has not been compiled yet!")
        
        col_names = ['Layer (type)', 'Input Shape', 'Output Shape', "Activation", "Initialiser", "# of parameters"]

        pad = 5

        print('Model Summary:')
        print('='*sum([len(name) + pad for name in col_names]))

        header = ''.join(f"{name:^{len(name) + pad}}" for name in col_names)
        print(header)

        total = 0
        trainable = 0
        non_trainable = 0

        counter = {layer_type.__name__: 1 for layer_type in Layer.__subclasses__()}

        print('-'*sum([len(name) + pad for name in col_names]))

        col_data = ['Input', self.in_shape, self.in_shape, 'N/A', 'N/A', 0]
        information = ''.join(f"{str(data):^{len(name) + pad}}" for name, data in zip(col_names, col_data))
        print(information)

        for i, layer in enumerate(self.layers):

            col_data = []

            name = layer.__class__.__name__
            act_func = "N/A"
            if name in counter.keys():
                if i < len(self.layers) and self.layers[i+1].__class__.__name__ not in counter.keys():
                    act_func = self.layers[i+1].__class__.__name__
            else:
                continue

            print('-'*sum([len(name) + pad for name in col_names]))

            col_data.append(f"{name}_{counter[name]}")
            counter[name] += 1

            if name == "Conv2D":
                col_data.append(layer.in_shape_x)
            else:
                col_data.append(layer.in_shape)
            col_data.append(layer.out_shape)
            col_data.append(act_func)

            if name in ("Dense", "Conv2D"):
                col_data.append(layer.initialiser.__class__.__name__)
            else:
                col_data.append('N/A')

            tr, ntr = layer.params()

            total += tr + ntr
            trainable += tr
            non_trainable += ntr

            col_data.append(tr + ntr)

            information = ''.join(f"{str(data):^{len(name) + pad}}" for name, data in zip(col_names, col_data))
            print(information)

        print()

        print(f"Cost function: {self.cost_func.__class__.__name__}")
        print(f"Optimiser: {self.optimiser.__name__}")

        print(f"Total params: {total}")
        print(f"Trainable params: {trainable}")
        print(f"Non-trainable params: {non_trainable}")

    def forward(self, X, mode="train"):
        Z = X.copy()
        for layer in self.layers:
            if layer.__class__.__name__ == "BatchNorm":
                Z = layer.forward(Z, mode)
            else:
                Z = layer.forward(Z)
        return Z
    
    def backward(self, dZ):
        for layer in self.layers[::-1]:
            dZ = layer.backward(dZ)
    
    def fit(self, X, y, epochs=10, batch_size=5, lr=1, X_val=None, y_val=None, lr_decay=Constant(), **kwargs):

        self.history = {
            'Training Loss': [],
            'Validation Loss': [],
            'Training Accuracy': [],
            'Validation Accuracy': []
        }

        iterations = 0
        self.m = batch_size
        
        for epoch in range(epochs):
            cost_train = 0
            num_batches = 0
            y_pred_train = []
            y_train = []

            print()
            print(f"Epoch: {epoch+1}/{epochs}")

            for i in tqdm(range(0, len(X), batch_size)):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                Z = self.forward(X_batch)

                pred, ans = self.cost_func.predictions(Z, y_batch)
                y_pred_train += pred
                y_train += ans

                cost_train += self.cost_func.cost(Z, y_batch) / self.m

                dZ = self.cost_func.backward(Z, y_batch)

                self.backward(dZ)

                for layer in self.layers:
                    if layer.__class__.__name__ in ("Dense", "BatchNorm", "Conv2D"):
                        layer.update(lr, self.m, iterations)

                lr = lr_decay.update(iterations, **kwargs)

                num_batches += 1
                iterations += 1

            cost_train /= num_batches

            accuracy_train = np.mean(np.array(y_pred_train) == np.array(y_train))

            self.history["Training Loss"].append(cost_train)
            self.history["Training Accuracy"].append(accuracy_train)

            print(f"Training Loss: {cost_train:.4f}")
            print(f"Training Accuracy: {accuracy_train:.4f}")

            if X_val is None or y_val is None:
                continue

            cost_val, accuracy_val = self.evaluate(X_val, y_val, batch_size)

            self.history["Validation Loss"].append(cost_val)
            self.history["Validation Accuracy"].append(accuracy_train)

            print(f"Validation Loss: {cost_val:.4f}")
            print(f"Validation Accuracy: {accuracy_val:.4f}")
            
    def evaluate(self, X, y, batch_size=None):
        if batch_size is None:
            batch_size = len(X)

        cost = 0
        correct = 0
        num_batches = 0

        for i in tqdm(range(0, len(X), batch_size)):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            Z = self.forward(X_batch, "test")

            cost += self.cost_func.cost(Z, y_batch) / len(y_batch)
            pred, ans = self.cost_func.predictions(Z, y_batch)
            correct += np.sum(np.array(pred) == np.array(ans))

            num_batches += 1

        accuracy = correct / len(y)
        cost /= num_batches

        return cost, accuracy

    def accuracy_plot(self):
        plt.plot(self.history['Training Accuracy'], 'k')
        if len(self.history['Validation Accuracy'])>0:
            plt.plot(self.history['Validation Accuracy'], 'r')
            plt.legend(['Train', 'Validation'], loc='lower right')
            plt.title('Model Accuracy')
        else:
            plt.title('Training Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

    def loss_plot(self):
        plt.plot(self.history['Training Loss'], 'k')
        if len(self.history['Validation Loss'])>0:
            plt.plot(self.history['Validation Loss'], 'r')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.title('Model Loss')
        else:
            plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    def cm_plot(self, X_test, y_test, batch_size=None):
        y_pred = self.predict(X_test, batch_size)

        cm = np.array(confusion_matrix(y_test, y_pred))

        sns.heatmap(cm, annot=True, fmt="g", xticklabels=np.arange(10), yticklabels=np.arange(10))
        plt.ylabel('Prediction', fontsize=13)
        plt.xlabel('Actual', fontsize=13)
        plt.title('Confusion Matrix', fontsize=17)
        plt.show()

    def predict(self, X, batch_size=None): 
        if batch_size is None:
            batch_size = len(X)

        y_pred = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            Z = self.forward(X_batch, "test")

            y_pred += np.argmax(Z, axis=1).tolist()

        return np.array(y_pred)
    
    def accuracy(self, X_test, y_test, batch_size=None):
        y_pred = self.predict(X_test, batch_size)

        acc = accuracy_score(y_test, y_pred)

        print(f"Error Rate: {(1-acc)*100:2}")
        print(f"Accuracy: {acc*100:2}")

        


        
        

