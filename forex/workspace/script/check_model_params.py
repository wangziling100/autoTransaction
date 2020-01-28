from analyse import Predictor


def check_model_params(name, level):
    pred = Predictor(name, level)
    pred.load()
    print(list(pred.model.parameters()))


if __name__ == '__main__':
    name = "EURUSD"
    levels = ['1s', '5s', '10s', '30s', '1m', '5m', '10m', '30m',
            '1h']

    for level in levels:
        print('------------------')
        print(level)
        check_model_params(name, level)
