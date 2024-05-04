import tkinter as tk
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import os.path
from Network import Network

class MNIST(tk.Tk):
    def __init__(self):
        super().__init__()
        self.net = None
        self.canvas_reduce = None
        self.btnPredict = None
        self.btnClear = None
        self.lblNumber = None
        self.panel = None
        self.canvas = None
        self.height_canvas = None
        self.width_canvas = None
        self.array_canvas = None
        self.array_canvas_scaled = None
        self.window_width = None
        self.window_height = None

        self.title('MNIST NUMBER DETECTOR')
        self.set_window(500, 300)
        self.create_widgets()
        self.setup_canvas()
        self.setup_network()

    def create_widgets(self):
        self.width_canvas = 280
        self.height_canvas = 280
        self.canvas = tk.Canvas(self, width=self.width_canvas, height=self.height_canvas, bg='white')
        self.canvas.pack(side=tk.LEFT, padx=30, pady=30)
        self.canvas.bind("<B1-Motion>", self.draw_square)

        self.lblNumber = tk.Label(self.panel, text=" ", font=("Consolas", 20), bg='white')
        self.lblNumber.pack(side=tk.BOTTOM, pady=10, padx=10)

        self.btnClear = tk.Button(self.panel, text="CLEAR", command=self.clear_canvas, bg='green')
        self.btnClear.pack(side=tk.BOTTOM, pady=10, padx=30)

        self.btnPredict = tk.Button(self.panel, text="PREDICT", command=self.predict, bg='green')
        self.btnPredict.pack(side=tk.BOTTOM, pady=10, padx=30)

    def draw_square(self, event):
        side = 16
        x1, y1 = int(event.x - side / 2), int(event.y - side / 2)
        x2, y2 = int(event.x + side / 2), int(event.y + side / 2)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black")
        self.update_canvas_array(x1, y1, x2, y2)

    def update_canvas_array(self, x1, y1, x2, y2):
        for i in range(max(0, y1), min(self.height_canvas, y2)):
            for j in range(max(0, x1), min(self.width_canvas, x2)):
                self.array_canvas[i][j] = 1
        r = self.canvas_reduce
        for i, row in enumerate(self.array_canvas[::r, ::r]):
            for j, value in enumerate(row):
                self.array_canvas_scaled[i][j] = np.mean(self.array_canvas[i*r:(i*r)+r, j*r:(j*r)+r])

    def predict(self):
        array_num = np.reshape(self.array_canvas_scaled, (784, 1))
        res = self.net.feedforward(array_num)
        self.lblNumber["text"] = f"{np.argmax(res)}"
        self.canvas.delete("all")
        self.setup_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.lblNumber["text"] = " "
        self.setup_canvas()

    def setup_network(self):
        self.net = Network([784, 30, 10])

        if os.path.isfile("data/models.pkl"):
            with open('data/models.pkl', 'rb') as f:
                self.net.weights, self.net.biases = pickle.load(f)
        else:
            training_data, validation_data, test_data = wrap_data()
            self.net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
            with open('data/models.pkl', 'wb') as f:
                pickle.dump((self.net.weights, self.net.biases), f)

    def setup_canvas(self):
        self.array_canvas = np.zeros((self.height_canvas, self.width_canvas), dtype=np.float32)
        self.canvas_reduce = 10
        self.array_canvas_scaled = np.zeros((self.height_canvas//self.canvas_reduce, self.width_canvas//self.canvas_reduce), dtype=np.float32)

    def set_window(self, width, height):
        self.window_width = width
        self.window_height = height

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        center_x = int(screen_width / 2 - self.window_width / 2)
        center_y = int(screen_height / 2 - self.window_height / 2)
        self.geometry(f'{self.window_width}x{self.window_height}+{center_x}-{center_y}')
        self.resizable(False, False)

def load_data():
    mnist = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, classification_data, test_data = pickle.load(mnist, encoding='latin1')
    mnist.close()
    return (training_data, classification_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def wrap_data():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def display_images(data):
    for i in range(10):
        for p in data[0][i]:
            print(p)
        image = data[0][i].reshape((28, 28))
        label = data[1][i]
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()

if __name__ == "__main__":
    mnist = MNIST()
    mnist.mainloop()