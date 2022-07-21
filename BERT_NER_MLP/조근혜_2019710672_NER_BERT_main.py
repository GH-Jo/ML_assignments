# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from bert.optimization import AdamW, get_linear_schedule_with_warmup
from bert.configuration import BertConfig
from bert.tokenization import BertTokenizer
from model import BERT_MLP

# ========= [ 학번, 이름을 수정하세요. ] ==========
from 조근혜_2019710672_NER_BERT_util import get_labels, load_and_cache_examples
# ========= [ 학번, 이름을 수정하세요. ] ==========

logger = logging.getLogger(__name__)

def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_step_proportion), num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    set_seed()
    model.train(True)
    for now_epoch in range(args.num_train_epochs):

        for step, batch in enumerate(train_dataloader):
            
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}

            loss, output = model(**inputs)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        logger.info("Epoch {} done. loss : {}".format(now_epoch + 1, logging_loss))
        tr_loss += logging_loss
        logging_loss = 0.0

    model.train(False)

    return global_step, tr_loss / global_step

def evaluate(args, eval_dataset, model, labels, pad_token_label_id):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}

            tmp_eval_loss, output = model(**inputs)

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        if preds is None:
            preds = output.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, output.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    precision = None
    recall = None
    f1 = None
    # ========== [ 9주차 과제를 참고하여 여기를 구현하세요. ] ==========
    # Sequential Labeling Evaluation
    # pred_list : 모델의 예측값 / out_label_list : 실제 정답
    # text파일에 file I/O를 이용하여 micro averaging (precision/recall/f1-score) 출력 
    def spanizer(conversation):
      span=[]
      entity_flag = False
      for speech in conversation:
        spch_span=[]
        for i,word in enumerate(speech):
          if word.startswith('B'):
            entity_flag = True
            word_split = word.split('-')
            spch_span.append([word_split[1], i, i])
          elif (entity_flag==True) and (word.startswith('I')):
            spch_span[-1][-1] = i
          elif (entity_flag==True) and (word=='O'):
            entity_flag = False  
        span.append(spch_span)
      return span

    pred_span = spanizer(preds_list)
    label_span = spanizer(out_label_list)
    def make_confusion(entity_type):
      confusion=[0,0,0,0]  # TP, FP, FN, TN
      for j in range(len(pred_span)):
        speech_pred = pred_span[j]
        speech_label = label_span[j]
        TP, TN, FP, FN = 0, 0, 0, 0
        # pred-based: TP, FP
        for p_word in speech_pred: 
          if p_word[0] == entity_type:
            if p_word in speech_label:
              TP += 1
            else:
              FP += 1
        # label-based: FN, TN
        for l_word in speech_label:
          if l_word[0] == entity_type:
            if l_word not in speech_pred:
              FN += 1
          else:
            TN += 1
        confusion[0] += TP
        confusion[1] += FP
        confusion[2] += FN
        confusion[3] += TN
      return confusion

    PS_confusion = make_confusion('PS')
    LC_confusion = make_confusion('LC')
    OG_confusion = make_confusion('OG')
    DT_confusion = make_confusion('DT')
    TI_confusion = make_confusion('TI')

    TOTAL_confusion=[0,0,0,0]
    for matrix in [PS_confusion, LC_confusion, OG_confusion, DT_confusion, TI_confusion]:
      for j in range(4):
        TOTAL_confusion[j] += matrix[j]


    # text파일에 file I/O를 이용하여 micro averaging (precision/recall/f1-score) 출력 
    TP, FP, FN, TN = TOTAL_confusion 
    #print(TP, FP, FN, TN)
    precision = 100*TP/(TP+FP)
    recall = 100*TP/(TP+FN)
    f1 = 2*(precision*recall) / (precision+recall)

    fw = open('조근혜_2019710672_NER_BERT.txt', 'w')
    print('Micro averaging precision : {:.4f}%\n'.format(precision))
    fw.write('Micro averaging precision : {:.4f}%\n\n'.format(precision))
    print('Micro averaging recall : {:.4f}%\n'.format(recall))
    fw.write('Micro averaging recall : {:.4f}%\n\n'.format(recall))
    print('Micro averaging f1-score : {:.4f}%'.format(f1))
    fw.write('Micro averaging f1-score : {:.4f}%\n'.format(f1))
    fw.close()



    # ========== [ 여기까지 구현하세요. ] ==========
    results = {"precision": precision, "recall": recall, "f1": f1}

    logger.info("***** Eval results *****")
    logger.info("Micro averaging precision : {}%;\tMicro averaging recall : {}%;\tMicro averaging f1-score : {}%"
        .format(results["precision"], results["recall"], results["f1"]))

    return results, preds_list

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='./',
        type=str,
        help="The input data dir. Should contain the training files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.warning("======[Device]: %s ======", args.device)

    args.max_seq_length = 512
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.weight_decay = 0.0
    args.warmup_step_proportion = 0.1
    args.eval_batch_size = 4
    args.gradient_accumulation_steps = 1

    # ========== [ Hyperparameters 실험 ] ==========
    # Batch size for training. (int)
    args.train_batch_size = 5
    # The initial learning rate. (float)
    args.learning_rate = 0.005
    # Total number of training epochs to perform. (int)
    args.num_train_epochs = 20
    # ========== [ Hyperparameters 실험 ] ==========

    logger.info("Training/evaluation parameters %s", args)
    set_seed()

    labels = get_labels()

    num_labels = len(labels)
    
    pad_token_label_id = CrossEntropyLoss().ignore_index

    config = BertConfig.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)}
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased",
        do_lower_case=False,
        cache_dir=None
    )
    model = BERT_MLP.from_pretrained(
        "bert-base-multilingual-cased",
        from_tf=False,
        config=config,
        cache_dir=None
    )
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="NER_train")
    test_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="NER_test")

    # Train
    global_step, tr_loss = train(args, train_dataset, model)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results, _ = evaluate(args, test_dataset, model, labels, pad_token_label_id)

if __name__ == "__main__":
    main()
