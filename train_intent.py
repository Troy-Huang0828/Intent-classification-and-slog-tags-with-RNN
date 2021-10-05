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

TRAIN = "./data/intent/train.json"
GLOVE = "glove.840B.300d.txt"
DEV = "eval"
SPLITS = [TRAIN, DEV]
limit = 25
#######訓練資料######
EPOCH = 40               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 16
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 300        # rnn input size / image width
LR = 0.001

def main(args):
    def zero_list():
        array = np.zeros((300), dtype=np.float)
        return array.tolist()


    def sentence_word(file):
        storage = []
        sentence = []
        single_expened = []
        expened = []
        with open(file, 'r') as obj:
            data = json.load(obj)
            for i in range(len(data)):
                i = data[i]['text']
                storage.append(i)
            for k in storage:
                trans = k.split()
                sentence.append(trans)

            for i in tqdm(range(len(sentence))):
                for j in range(limit):
                    try:
                        single_expened.append(sentence[i][j])
                    except:
                        single_expened.append('<none>')
                expened.append(single_expened)
                single_expened = []
        return expened


    #####建立input train data####
    with open('dictionary.json', 'r', encoding='UTF-8') as file:
        new_dictionary = json.load(file)
        sentence_list = sentence_word('data/intent/train.json')

        for i in tqdm(range(len(sentence_list))):
            for j in range(limit):
                if sentence_list[i][j] in new_dictionary.keys():
                    sentence_list[i][j] = np.array(new_dictionary[sentence_list[i][j]])
                else:
                    sentence_list[i][j] = np.array(zero_list())
        sentence_array = np.array(sentence_list)
        train_input_data_tensor = torch.from_numpy(sentence_array).type(torch.FloatTensor)
        print(sentence_array.shape)

    #####建立output train label####
    with open('./cache/intent/intent2idx.json', 'r', encoding='UTF-8') as file:
        output_sample = json.load(file)
        with open('./data/intent/train.json', 'r', encoding='UTF-8') as fp:
            input_data = []
            output = []
            data = json.load(fp)
            for i in range(len(data)):
                i = data[i]['intent']
                input_data.append(i)
            for a in range(len(input_data)):
                if input_data[a] in output_sample.keys():
                    output.append(output_sample[input_data[a]])
            output_arrary = np.array(output)
        print(output_arrary.shape)
        train_output_tensor = torch.from_numpy(output_arrary).type(torch.LongTensor)

    #####################建立eval input data###################
    with open('dictionary.json', 'r', encoding='UTF-8') as file:
        new_dictionary = json.load(file)
        sentence_list = sentence_word('data/intent/eval.json')

        for i in tqdm(range(len(sentence_list))):
            for j in range(limit):
                if sentence_list[i][j] in new_dictionary.keys():
                    sentence_list[i][j] = np.array(new_dictionary[sentence_list[i][j]])
                else:
                    sentence_list[i][j] = np.array(zero_list())
        sentence_array = np.array(sentence_list)
        eval_input_data_tensor = torch.from_numpy(sentence_array).type(torch.FloatTensor)
        print(sentence_array.shape)

    #############建立eval label###################
    with open('./data/intent/eval.json', 'r', encoding='UTF-8') as fp:
        eval_data = []
        data = json.load(fp)
        for i in range(len(data)):
            i = data[i]['intent']
            eval_data.append(i)

    #################test data##################################
    def test_list():
        with open(args.test, 'r', encoding='UTF-8') as test_file:
            test_sentense = []
            input_test = []
            single_test_sentense = []
            final = []
            file = json.load(test_file)
            for i in range(len(file)):
                trans = file[i]['text']
                input_test.append(trans)

            for j in input_test:
                trans = j.split()
                test_sentense.append(trans)

            for a in range(len(test_sentense)):
                for b in range(limit):
                    try:
                        single_test_sentense.append(test_sentense[a][b])
                    except:
                        single_test_sentense.append('<none>')
                final.append(single_test_sentense)
                single_test_sentense = []

        return final

    print(np.array(test_list()).shape)

    ########################TEST_DATA###############################

    with open('dictionary.json', 'r', encoding='UTF-8') as file:
        new_dictionary = json.load(file)
        sentence_list = test_list()

        for i in tqdm(range(len(test_list()))):
            for j in range(limit):
                if sentence_list[i][j] in new_dictionary.keys():
                    sentence_list[i][j] = np.array(new_dictionary[sentence_list[i][j]])

                else:
                    sentence_list[i][j] = np.array(zero_list())
        sentence_array = np.array(sentence_list)
        test_data_tensor = torch.from_numpy(sentence_array).type(torch.FloatTensor)
        print(sentence_array.shape)

    ###########################################開始訓練####################################################
    torch_dataset = Data.TensorDataset(train_input_data_tensor, train_output_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()

            self.rnn = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=850,
                num_layers=1,
                batch_first=True,
            )

            self.out = nn.Linear(850, 150)

        def forward(self, x):
            r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch, time_step, input_size)
            out = self.out(r_out[:, -1, :])  # (batch, time_step, input_size)
            return out

    rnn = RNN().cuda()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    flag = 0
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(-1, limit, 300)

            output = rnn(b_x.to('cuda'))
            # print(output.data.cpu().numpy().shape)
            # print(b_y.numpy().shape)
            loss = loss_func(output, b_y.to('cuda'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                count = 0
                output = rnn(b_x.to('cuda'))  # (samples, time_step, input_size)
                # print('h_n', h_n)
                # print('h_c', h_c)
                pred_y = torch.max(output, 1)[1].data.cpu().numpy()
                for i in range(BATCH_SIZE):
                    if pred_y[i] == b_y[i].data.cpu().numpy():
                        count += 1
                accuracy = count / BATCH_SIZE * 100
                print(
                    f'Epoch:{epoch} | step:{step} | train loss:{loss.data.cpu().numpy()} | train accuracy: {accuracy}%')
            if loss.data.cpu().numpy() < 0.0000005:
                flag = 1
                print(loss.data.cpu().numpy())
                break
        if flag:
            break

    ###################把訓練好的模型儲存起來############################

    # torch.save(rnn.state_dict(), 'train_cls.pkl')

    #####################資料寫入csv檔##########################

    answer = []
    final_prediction = rnn(test_data_tensor.to('cuda'))

    max_num = torch.max(final_prediction, 1)[1].data.cpu().numpy()
    print(max_num)
    max_num = np.array(max_num.tolist())
    with open('./cache/intent/intent2idx.json', 'r', encoding='UTF-8') as file:
        output_sample = json.load(file)
        new_dict = {v: k for k, v in output_sample.items()}
        for i in max_num:
            if i in new_dict:
                answer.append(new_dict[i])

    def id_list():
        id_list = []
        for i in range(len(answer)):
            id_list.append(f'test-{i}')
        return id_list

    dict = {"id": id_list(), 'intent': answer}
    dataframe = pd.DataFrame(dict, columns=['id', 'intent'])
    dataframe.to_csv(args.csv, index=False, sep=',')
    #####################evaluate my model###################
    eval_answer = []
    eval_count = 0
    final_prediction = rnn(eval_input_data_tensor.to('cuda'))
    max_num = torch.max(final_prediction, 1)[1].data.cpu().numpy()
    max_num = np.array(max_num.tolist())
    with open('./cache/intent/intent2idx.json', 'r', encoding='UTF-8') as file:
        output_sample = json.load(file)
        new_dict = {v: k for k, v in output_sample.items()}
        for i in max_num:
            if i in new_dict:
                eval_answer.append(new_dict[i])

    for i in tqdm(range(len(eval_answer))):
        if eval_answer[i] == eval_data[i]:
            eval_count+=1
            continue
        else:
            eval_count+=0
            continue
    eval_accuracy = (eval_count/len(eval_data))*100
    print(f'accuracy = {eval_accuracy}%')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/sampleSubmission.csv",
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

