from neural import NeuralNet

xor_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [1]),


]

orn = NeuralNet(2, 3, 1)

orn.train(or_data)
print(orn.test_with_expected(or_data))