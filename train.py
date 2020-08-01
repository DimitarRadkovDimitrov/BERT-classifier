import sys
import json
import pickle
import matplotlib.pyplot as plt
from bert_classifier import BERTClassifier


if __name__ == "__main__":
    path_to_config = sys.argv[1]

    with open(path_to_config, "r") as f:
        config = json.load(f)

    dataset = pickle.load(open(config['data']['output'], 'rb'))
    train_x = dataset['train'][0]
    train_y = dataset['train'][1]
    dev_x = dataset['dev'][0]
    dev_y = dataset['dev'][1]
    test_x = dataset['test'][0]
    test_y = dataset['test'][1]

    bert = BERTClassifier(len(train_x[0]), config)
    bert.model.summary()
    
    history = bert.train(train_x, train_y, dev_x, dev_y)
    report = bert.evaluate(test_x, test_y)
    print(report)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
