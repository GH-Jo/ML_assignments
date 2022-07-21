# -*- coding: utf-8 -*-

import torch
import os

class InputExample(object):
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

class InputFeatures(object):
    def __init__(self, input_idx, input_mask, segment_idx, label_idx):
        self.input_idx = input_idx
        self.input_mask = input_mask
        self.segment_idx = segment_idx
        self.label_idx = label_idx

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, pad_token_label_id):
    label2idx = {label: i for i, label in enumerate(label_list)}

    features = []
    for example in examples:
        input_idx = None
        input_mask = None
        segment_idx = None
        label_idx = None

        # ========== [ 여기부터 코딩하세요. ] ==========
        input_tokens = []
        sent_labels = []

        flag = 0
        for idx, word in enumerate(example.words):
            
            ## word_input_idx -> V
            ## word_input_mask -> X
            ## word_segment_idx -> X
            ## word_label_idx -> 

            # 1. make token and label (checked!)
            token = tokenizer.tokenize(word)
            # 1-1. [CLS]
            if idx==0:
              token.insert(0, '[CLS]')
            label = [example.labels[idx]]
            label.extend(['X']*(len(token)-1))

            # 2. Append
            input_tokens.extend(token)
            sent_labels.extend(label)

            #word_label_idx = [label_idx[lab] for lab in label]
            #label_idx.extend(word_label_idx)
            #print(word_label_idx)

            flag += 1
            if flag == 4:
              pass
                               
        # 0. segment id
        segment_idx = [0]*max_seq_length
        #print(len(input_tokens))
        # 1. [SEP]
        input_tokens.append('[SEP]')
        sent_labels.append('X')
        label2idx['X']=-100
        

        # 2. Input_mask
        input_mask = [1]*len(input_tokens)

        
        # 3. Convert
        input_idx = tokenizer.convert_tokens_to_ids(input_tokens)
        
        # 3. [PAD] 
        pad_length = max_seq_length-len(input_tokens)
        #print('pad_length:', pad_length)        
        input_idx.extend([0] * pad_length)
        input_mask.extend([0] * pad_length)
        
        # 4. convert
        label_idx = [label2idx[label] for label in sent_labels]
        
        # 5. Pad label_idx
        label_idx.extend([pad_token_label_id]*pad_length)
        #print(len(input_idx))
        #print(len(input_mask))
        #print(len(segment_idx))
        #print(len(label_idx))
        #print(input_idx)
        #print(input_mask)
        #print(segment_idx)
        #print(label_idx)
        
        

        """
        tokenizer를 사용하여 input_idx, input_mask, segment_idx, label_idx 구현
        tokenizer.tokenize(word) : 하나의 입력 형태소(word)를 여러 개의 sub-token으로 분할(분할되지 않을 수 있음)
        tokenizer.convert_tokens_to_ids(tokens) : input의 모든 token값을 index로 변환
        자세한 내용은 PDF 참조
        """


        # ========== [ 여기까지 코딩하세요. ] ==========   

        features.append(
            InputFeatures(
                input_idx=input_idx,
                input_mask=input_mask,
                segment_idx=segment_idx,
                label_idx=label_idx
                )
            )

    return features

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line == "" or line == "\n":
                if words:
                    examples.append(InputExample(words=words, labels=labels))
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "predict"
                    labels.append("O")
        if words:
            examples.append(InputExample(words=words, labels=labels))
    return examples

def get_labels():
    return ["O", "B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG", "B-DT", "I-DT", "B-TI", "I-TI"]

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    examples = read_examples_from_file(args.data_dir, mode)
    features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, pad_token_label_id)

    input_idx_tensor = None
    input_mask_tensor = None
    segment_idx_tensor = None
    label_idx_tensor = None

    dataset = None
    
    # ========== [ 여기부터 코딩하세요. ] ==========
    input_idx_list = []
    input_mask_list = []
    segment_idx_list = []
    label_idx_list = []
    
    for i in features:
      input_idx_list.append(i.input_idx)
      input_mask_list.append(i.input_mask)
      segment_idx_list.append(i.segment_idx)
      label_idx_list.append(i.label_idx)

    input_idx_tensor = torch.tensor(input_idx_list)
    input_mask_tensor = torch.tensor(input_mask_list)
    segment_idx_tensor = torch.tensor(segment_idx_list)
    label_idx_tensor = torch.tensor(label_idx_list)
    
    #print(input_idx_tensor.shape)
    #print(input_mask_tensor.shape)
    #print(segment_idx_tensor.shape)
    #print(label_idx_tensor.shape)

    #fw = open('./2019710672_조근혜_NER_BERT_util.txt', 'a', encoding='UTF-8')
    #if 'train' in  mode.lower():
    #  fw.write("NER_Train_Data\n")
    #else :
    #  fw.write('NER_Test_Data\n')
    #fw.write('input_idx_tensor size : {}\n'.format(input_idx_tensor.shape))
    #fw.write('input_mask_tensor size : {}\n'.format(input_mask_tensor.shape))
    #fw.write('segment_idx_tensor size : {}\n'.format(segment_idx_tensor.shape))
    #fw.write('label_idx_tensor size : {}\n'.format(label_idx_tensor.shape))
    #fw.write('\n')
    #fw.close()


    dataset = torch.utils.data.TensorDataset(input_idx_tensor,
                                             input_mask_tensor,
                                             segment_idx_tensor,
                                             label_idx_tensor)
    
    """
    1. features의 input_idx, input_mask, segment_idx, label_idx를 각각 tensor자료형으로 변환 후
       input_idx_tensor, input_mask_tensor, segment_idx_tensor, label_idx_tensor에 저장
    2. Tensor 자료형에 ".size()"함수를 사용하여
       input_idx_tensor, input_mask_tensor, segment_idx_tensor, label_idx_tensor의 tensor size를 text에 출력
    3. torch.utils,data.TensorDataset함수를 사용하여 dataset 구축
    """
    


    # ========== [ 여기까지 코딩하세요. ] ==========   

    return dataset