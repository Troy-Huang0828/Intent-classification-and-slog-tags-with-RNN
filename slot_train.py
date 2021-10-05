import json
import numpy as np
from tqdm import tqdm
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import pandas as pd
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import torch.utils.data as Data
from tqdm import trange

# from dataset import SeqClsDataset
# from utils import Vocab
import eval
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


EPOCH = 120                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 12
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 300        # rnn input size / image width
LR = 0.001
max_word_len = 35


def main(args):


    def zero_list():
        array = np.zeros((300), dtype=np.float)
        return array.tolist()

    with open('data/slot/train.json', 'r', encoding='UTF-8') as fp:
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

    #################train input data#######################
    def train_input_array():
        unfindword = 0
        dic_add = {}
        with open('dictionary.json', 'r', encoding='UTF-8') as dictionary_file:
            num_list = word_list
            dictionary = json.load(dictionary_file)
            for i in tqdm(range(len(num_list))):
                for j in range(max_word_len):
                    if num_list[i][j] in dictionary.keys():
                        num_list[i][j] = dictionary[num_list[i][j]]
                    else:
                        unfindword = unfindword + 1
                        num_list[i][j] = zero_list()
                #         dic_add = {f"{num_list[i][j]}":f"{dictionary[num_list[i][j]]}"}
                #         print(dic_add)
                # dictionary.update(dic_add)
        input_train_data_tensor = torch.Tensor(num_list).type(torch.FloatTensor)
        return input_train_data_tensor

    # print('processing train input data, which shape is ',train_input_array().numpy().shape)

    ##################train output data#####################
    def train_output_array():
        with open('cache/slot/tag2idx.json', 'r', encoding='UTF-8') as fp:
            number = class_list
            tags_num_dictionary = json.load(fp)
            for i in tqdm(range(len(number))):
                for j in range(max_word_len):
                    if number[i][j] in tags_num_dictionary.keys():
                        number[i][j] = tags_num_dictionary[number[i][j]]
        # print(number)
        output_train_data_tensor = torch.Tensor(number).type(torch.LongTensor)
        return (output_train_data_tensor)

    print('processing train output data, which shape is ', train_output_array().numpy().shape)

    ##################test output data#####################
    def test_input_array():
        unfindword = 0
        with open(args.test, 'r', encoding='UTF-8') as fp:
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
                        unfindword = unfindword + 1
                        num_list[i][j] = zero_list()
            print(f'testinputerror : {unfindword}')
        input_test_data_tensor = torch.Tensor(num_list).type(torch.FloatTensor)
        return (input_test_data_tensor)

    print('processing test input data, which shape is ', test_input_array().numpy().shape)

    ######################建立model#####################

    DataLoader = Data.TensorDataset(train_input_array(), train_output_array())
    train_loader = torch.utils.data.DataLoader(dataset=DataLoader, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()

            # self.fc1 = nn.Linear(300, 300)

            self.rnn = torch.nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )

            # self.relu = nn.ReLU()
            self.h1 = nn.Linear(128*2, 10)
            # self.h2 = nn.Linear(64, 10)

        def forward(self, x):
            r_out, h_state = self.rnn(x, None)  # x (batch, time_step, input_size)
            out = self.h1(r_out)  # (batch, time_step, input_size)
            # out = self.h2(out)
            return out

    rnn = RNN().cuda()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters())
    loss_func = nn.CrossEntropyLoss()
    flag = 0
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            # print(np.array(b_x).size)
            b_x = b_x.view(-1, 35, 300)
            # print(np.array(b_x).size)
            output = rnn(b_x.to('cuda'))
            # print(output)
            # print(output.data.cpu().numpy().shape)
            # print(b_y.numpy().shape)
            loss = loss_func(output.view(BATCH_SIZE * 35, 10), b_y.to('cuda').view(BATCH_SIZE * 35))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
                count = 0
                output = rnn(b_x.to('cuda'))  # (samples, time_step, input_size)
                # print('h_n', h_n)
                # print('h_c', h_c)
                pred_y = torch.max(output, 2)[1].data.cpu().numpy()
                # print(pred_y)
                # print(b_y)
                for i in range(BATCH_SIZE):
                    for j in range(35):
                        if pred_y[i][j] == b_y[i][j].data.cpu().numpy():
                            if j == 34:
                                count += 1
                                break
                        else:
                            count += 0
                            break
                accuracy = count / BATCH_SIZE * 100
                print(
                    f'Epoch:{epoch} | step:{step} | train loss:{loss.data.cpu().numpy()} | train accuracy: {accuracy}%')
            if loss.data.cpu().numpy() < 0.0000005:
                flag = 1
                print(loss.data.cpu().numpy())
                break
        if flag:
            break

    def new_dict():
        with open('./cache/slot/tag2idx.json', 'r', encoding='UTF-8') as file:
            output_sample = json.load(file)
            new_dict = {v: k for k, v in output_sample.items()}
        return new_dict

    def word_correction(file, prediction):
        adjust = prediction
        word = []
        count = 0
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i in range(len(data)):
                tokens = data[i]["tokens"]
                word.append(tokens)

            for j in range(len(word)):
                if len(prediction[j]) <= len(word[j]):
                    for long in range(len(word[j]) - len(prediction[j])):
                        adjust[j].append('O')
                if len(prediction[j]) > len(word[j]):
                    temp = adjust[j]
                    adjust[j] = temp[0:-(len(prediction[j]) - len(word[j]))]
        return adjust

    # torch.save(rnn.state_dict(),'slot_tags_try.pkl')

    #########################test data prediction#########################
    A = []
    bigA = []
    final_prediction = rnn(test_input_array().to('cuda'))
    # print(final_prediction.data.cpu().numpy().shape)

    for i in tqdm(range(len(final_prediction))):
        for j in range(len(final_prediction[i])):
            max_num = np.argmax(final_prediction[i][j].data.cpu().numpy())
            A.append(max_num)
        bigA.append(A)
        A = []
    print(np.array(bigA).shape)

    answer = []
    answer_adjust = []
    for i in range(len(bigA)):
        for j in range(len(bigA[i])):
            if bigA[i][j] in new_dict():
                if bigA[i][j] != 9:
                    answer.append(new_dict()[bigA[i][j]])
                    continue
            else:
                print('error')
                pass
        answer_adjust.append(answer)
        answer = []
    answer_adjust = word_correction('data/slot/test.json', answer_adjust)


    def id_list():
        id_list = []
        for i in range(len(answer_adjust)):
            id_list.append(f'test-{i}')
        return id_list

    submit_answer = []
    for line in answer_adjust:
        str = " ".join(line)
        submit_answer.append(str)

    dict = {"id": id_list(), 'tags': submit_answer}
    dataframe = pd.DataFrame(dict, columns=['id', 'tags'])
    dataframe.to_csv(args.csv, index=False, encoding='utf-8')


    ########################eval data prediction###########################
    B = []
    bigB = []
    counting = 0
    count_Token = 0
    count_j = 0
    eval_prediction = rnn(eval.eval_input_array().to('cuda'))
    eval_answer = eval.eval_answer()

    for i in tqdm(range(len(eval_prediction))):
        for j in range(len(eval_prediction[i])):
            max_num_eval = np.argmax(eval_prediction[i][j].data.cpu().numpy())
            B.append(max_num_eval)
        bigB.append(B)
        B = []
    print(np.array(bigB).shape)
    # print(bigB)
    answer = []
    answer_list = []
    for i in range(len(bigB)):
        for j in range(len(bigB[i])):
            if bigB[i][j] in new_dict():
                if bigB[i][j] != 9:
                    answer.append(new_dict()[bigB[i][j]])
                    continue
            else:
                print('error')
                pass
        answer_list.append(answer)
        answer = []

    answer_list = word_correction('data/slot/eval.json', answer_list)
    for i in range(len(eval_answer)):
        for j in range(len(eval_answer[i])):
            if answer_list[i][j] == eval_answer[i][j]:
                if j == len(eval_answer[i]) - 1:
                    counting += 1
                    break
            else:
                counting += 0
                break
    accuracy = counting / len(eval_answer) * 100
    print(f'eval accuracy after adjust:  {accuracy}%')
    print(f'joint Accuracy = { counting } / {len(eval_answer) }')

    for i in range(len(eval_answer)):
        for j in range(len(eval_answer[i])):
            count_j +=1
            if answer_list[i][j] == eval_answer[i][j]:
                count_Token += 1
                if j == len(eval_answer[i]) - 1:
                    break
            else:
                count_Token += 0
                continue
    print(f'Token Accuracy = { count_Token } / {count_j}')

    y_true = answer_list
    y_pred = eval.eval_answer()
    print(f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/sampleSubmission.csv",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
