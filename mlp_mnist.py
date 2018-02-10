from MLP.Perceptron import Perceptron
from Utility.Read import Read


if __name__ == "__main__":
    print("Reading in data...")
    #reader = Read(784, 30000, 255, "train_labels", "train_data")
    #t_labels, t_data = reader.run()
    #p = Perceptron()
    #p.train(t_data, t_labels, 0.25, 10)

    #reader = Read(784, 10000, 255, "validation_labels", "validation_data")
    #v_labels, v_data = reader.run()
    #p.predict(v_data, v_labels)

    reader = Read(784, 2000, 255, "test_labels", "test_data")
    test_label, test_data = reader.run()
    p = Perceptron()
    p.load_model("/Users/kaseystowell/Documents/workspace/mlp-mnist/weighth.csv",
                 "/Users/kaseystowell/Documents/workspace/mlp-mnist/weighto.csv")
    p.predict(test_data, test_label)