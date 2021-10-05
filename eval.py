import json
from tqdm import tqdm
import numpy as np
import torch
max_word_len = 35


def zero_list():
    array = np.zeros((300), dtype = np.float)
    return array.tolist()


with open('data/slot/eval.json', 'r', encoding='UTF-8') as fp:
    word_list = []
    class_list = []
    temp = []

    data = json.load(fp)
    for i in range(len(data)):
        tokens = data[i]["tokens"]
        word_list.append(tokens)
    for j in range(len(word_list)):
        if len(word_list[j]) <= max_word_len:
            for l in range(max_word_len - len(word_list[j])):
                word_list[j].append('<none>')

    for a in range(len(data)):
        tags = data[a]['tags']
        class_list.append(tags)
    for b in range(len(class_list)):
        if len(class_list[b]) <= max_word_len:
            for l in range(max_word_len - len(class_list[b])):
                class_list[b].append('no')

def eval_input_array():
    with open('data/slot/eval.json', 'r', encoding='UTF-8') as fp:
        word_list = []
        data = json.load(fp)
        for i in range(len(data)):
            tokens = data[i]["tokens"]
            word_list.append(tokens)

        for j in range(len(word_list)):
            if len(word_list[j]) <= max_word_len:
                for l in range(max_word_len - len(word_list[j])):
                    word_list[j].append('<none>')
        # print(word_list)
    with open('dictionary.json', 'r', encoding='UTF-8') as dictionary_file:
        num_list = word_list
        dictionary = json.load(dictionary_file)
        for i in tqdm(range(len(num_list))):
            for j in range(max_word_len):
                if num_list[i][j] in dictionary.keys():
                    num_list[i][j] = dictionary[num_list[i][j]]
                else:
                    num_list[i][j] = zero_list()
    input_test_data_tensor = torch.Tensor(num_list).type(torch.FloatTensor)
    return (input_test_data_tensor)

print('processing test input data, which shape is ', eval_input_array().numpy().shape)


##################train output data#####################
def eval_output_array():
    with open('cache/slot/tag2idx.json', 'r', encoding='UTF-8') as fp:
        number = class_list
        tags_num_dictionary = json.load(fp)
        for i in tqdm(range(len(number))):
            for j in range(max_word_len):
                if number[i][j] in tags_num_dictionary.keys():
                    number[i][j] = tags_num_dictionary[number[i][j]]
    # print(number)
    output_train_data_tensor = torch.Tensor(number).type(torch.LongTensor)
    return(output_train_data_tensor)
print('processing train output data, which shape is ',eval_output_array().numpy().shape)

# print(eval_input_array())
# print(eval_output_array())

def eval_answer():
    with open('data/slot/eval.json', 'r', encoding='UTF-8') as fp:
        ans = []
        tags = json.load(fp)
        for i in range(len(tags)):
            tag = tags[i]["tags"]
            ans.append(tag)
        # print('call')
        return ans

