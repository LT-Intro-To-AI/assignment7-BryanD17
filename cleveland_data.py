from typing import Tuple
from neural import *


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")

    print()
    print(tokens)
    if "?" not in tokens: 
        out = int(tokens[13])
        output = [1 if out == 1 else 0.5 if out == 2 else 1]

        inpt = [float(x) for x in tokens[1:13]]
        print((inpt, [out]))
        
        return (inpt, [out])
    else: 
        out = int(tokens[13])
        output = [1 if out == 1 else 0.5 if out == 2 else 1]
        print("found")

        inpt = [float(x) for x in tokens[1:12]]
        inpt.append(1)
        print((inpt, [out]))
        
        return (inpt, [out])


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("processed.cleveland.data", "r") as f:
        
    # # Read in the file contents as a string
    # contents = f.read()

    # # Split the contents of the file into individual words
    # words = contents.split()

    # # Loop through each word and replace it with a random integer
    # for i in range(len(words)):
    #     # Check if the current word is composed entirely of letters
    #     if words[i].isalpha():
    #         # Generate a random integer between 1 and 4
    #         random_int = random.randint(1, 4)
    #         # Replace the current word with the random integer
    #         words[i] = str(random_int)

    # # Join the words back together into a string with spaces between them
    # new_contents = ' '.join(words)

    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

for line in training_data: 
    print (line)

# # Open the input file for reading
# with open("processed.cleveland.data", "r") as f:
    
#     # Read in the file contents as a string
#     contents = f.read()

#     # Split the contents of the file into individual words
#     words = contents.split()

#     # Loop through each word and replace it with a random integer
#     for i in range(len(words)):
#         # Check if the current word is composed entirely of letters
#         if words[i].isalpha():
#             # Generate a random integer between 1 and 50
#             random_int = random.randint(1, 50)
#             # Replace the current word with the random integer
#             words[i] = str(random_int)

#     # Join the words back together into a string with spaces between them
#     new_contents = ' '.join(words)

# Open the output file for writing and write the new contents to it
# with open("processed.cleveland.data", "r") as f:
#     f.write(new_contents)




td = normalize(training_data)

nn = NeuralNet(13, 3, 1)
nn.train(td, iters=1000, print_interval=1000, learning_rate=0.1)

for i in nn.test_with_expected(td):
    print(f"desired: {i[1]}, actual: {i[2]}")

