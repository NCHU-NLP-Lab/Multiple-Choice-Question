import torch
import torch.nn.functional as F
from tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from modeling import BertForGenerativeSeq
import collections
import logging
import os
import argparse
from rouge import Rouge

class Data(object):

    def __init__(self,
                 doc_tokens,
                 answers_text,
                 answer_start):
        self.doc_tokens = doc_tokens
        self.answers_text = answers_text
        self.answer_start = answer_start


class InputFeatures(object):

    def __init__(self,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_pos):
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_pos = label_pos       


class BERT_HLSQG(object):
    def __init__(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        modelpath = 'model/SQG_hl_id3_bert-base-chinese'
        training_modelpath = 'model/CH_HLSQG_model/pytorch_model_3.bin'
        device = torch.device("cuda")
        tokenizer = BertTokenizer.from_pretrained(modelpath)
        model_state_dict = torch.load(training_modelpath)
        
        model = BertForGenerativeSeq.from_pretrained(modelpath, state_dict=model_state_dict)
        model.eval()
        model.to(device)

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.BS_size = 5
        self.max_seq_length = 512
        self.doc_stride = 476
        self.max_query_length = 42

    def predict(self, context, answer, answer_start, BS = 5):
        rouge = Rouge()
        data = self.read_data(context = context, answer = answer, answer_start = answer_start)
        
        features =  self.convert_data_to_features(
                    data=data,
                    max_seq_length = self.max_seq_length,
                    doc_stride = self.doc_stride,
                    max_query_length = self.max_query_length)

        gen_QG = []

        for BS_size in range(BS, BS+1):
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            input_num = features[0].label_pos - 1

            tmp_QG = []
            output_rank = self.BS(input_ids[0], segment_ids[0], input_mask[0], features[0].label_pos-1, BS_size)

            while (len(tmp_QG) < BS_size):
                tmp_rank = []
                for i in range(BS_size - len(tmp_QG)):  
                    input_num = features[0].label_pos - 1
                    for token_id in output_rank[i][1]:
                        input_ids[0][input_num] = token_id
                        input_num += 1

                    input_ids[0][input_num] = 103 #[MASK]
                    input_mask[0][input_num] = 1
                    segment_ids[0][input_num] = 2

                    tmp_rank += self.BS(input_ids[0], segment_ids[0], input_mask[0], input_num, BS_size, output_rank[i])

                tmp_rank = sorted(tmp_rank, key=lambda x:x[0], reverse=True)

                output_rank = tmp_rank[:BS_size - len(tmp_QG)]
                
                for ele in output_rank[:BS_size - len(tmp_QG)]:
                    if '[SEP]' in ele[2]:
                        
                        tmp_QG.append((ele[0]/ele[4],ele))
                        output_rank.remove(ele)
                if input_num + 1 >= self.max_seq_length:
                    break 

            gen_QG += tmp_QG #單一

            # Qs = [ele[1][2] for ele in gen_QG]    #全部比較
            
            # for Q in sorted(tmp_QG, key=lambda x:x[0], reverse=True):
            #     if Q[1][2] not in Qs:
            #         gen_QG.append(Q)
            #         break

            # gen_QG.append(sorted(tmp_QG, key=lambda x:x[0], reverse=True)[0])  #全部取第一

        result = []
        # print(sorted(gen_QG, key=lambda x:x[0], reverse=True))

        for ele in sorted(gen_QG, key=lambda x:x[0], reverse=True):

            # result.append(ele[1][2].replace('[SEP]','').replace('？','?').replace(' ',''))
            
            flag = 0
            Q = ele[1][2].replace('[SEP]','').replace('？','?')

            if len(result) == 0:
                result.append(Q)
            else:
                for al_Q in result:
                    rouge_scores = rouge.get_scores(Q, al_Q)[0]
                    # print(rouge_scores['rouge-1']['r'])
                    if rouge_scores['rouge-l']['r'] == 1:
                        flag = 1
                        break
                if flag == 0:
                    result.append(Q)
            
            # result.append(Q)

        result = [ele.replace(' ','') for ele in result]
        if len([ele for ele in result if answer not in ele.replace(' ','')]) == 0:
            return result
        else:
            result = [ele for ele in result if answer not in ele.replace(' ','')]
            return result

    def BS(self, input_ids, segment_ids, input_mask, input_num, BS_size, per_data=None, min_query_length = 3, repeat_size = 1):

        res = []
        predictions = self.model(input_ids.unsqueeze(0), segment_ids.unsqueeze(0), input_mask.unsqueeze(0))
        predictions_SM = F.log_softmax(predictions[0][input_num],0) 
     
        while(len(res) < BS_size):
            
            predicted_index = torch.argmax(predictions_SM).item()
            sorce = predictions_SM[predicted_index].item()
            
            predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])
            predicted_text = predicted_token[0]
            order = str(len(res))
            
            predictions_SM[predicted_index] = -1000000000
            
            
            if (per_data == None or per_data[4] < min_query_length) and predicted_text == '[SEP]':   
                # print('min error')
                continue

            # if per_data != None  and predicted_index in per_data[1]:
            # if per_data != None  and predicted_index in per_data[1][-repeat_size:]:
            #     print('repeat error')
            #     continue

            if per_data == None:
                res.append((sorce, [predicted_index], predicted_text, order, 1))
            else:
                predicted_index_list = per_data[1] + [predicted_index]
                sorce = per_data[0] + sorce 

                if '##' in predicted_text:
                    predicted_text = per_data[2] + predicted_text.replace('##','')
                else:
                    predicted_text = per_data[2] + ' ' + predicted_text

                order = per_data[3] + order
                step = per_data[4] + 1

                res.append((sorce, predicted_index_list, predicted_text, order, step))
            
        return res        


    def read_data(self, context, answer, answer_start):

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        datas = []
        answer_text = answer
        answer_start = answer_start

        answer_len = len(answer_text)                                
        
        doc_tokens = []
        prev_is_whitespace = True
        for i, c in enumerate(context):
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                #chinese
                if len(answer_text) == 1 and i == answer_start:
                    c = "[HL]" + c + "[HL]"
                elif answer_text[0] == c and i == answer_start:
                    c = "[HL]" + c
                elif answer_text[-1] == c and i == answer_start + answer_len - 1:
                    c = c + "[HL]"

                #eng
                # if len(answer_text) == 1 and i == answer_start:
                #     doc_tokens.append("[HL]")
                #     doc_tokens.append(c)
                #     doc_tokens.append("[HL]")
                #     prev_is_whitespace = True
                #     continue
                # elif answer_text[0] == c and i == answer_start:
                #     doc_tokens.append("[HL]")
                #     doc_tokens.append(c)
                #     prev_is_whitespace = False
                #     continue
                # elif answer_text[-1] == c and i == answer_start + answer_len - 1:
                #     if prev_is_whitespace:
                #         doc_tokens.append(c)
                #     else:
                #         doc_tokens[-1] += c
                                    
                #     doc_tokens.append("[HL]")
                #     prev_is_whitespace = False
                #     continue

                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
                
        data = Data(
            doc_tokens=doc_tokens,
            answers_text=answer_text,
            answer_start=answer_start)
        datas.append(data)
        
        return datas


    def convert_data_to_features(self, data, max_seq_length,
                                     doc_stride, max_query_length):
        
        features = []

        answer_tokens = self.tokenizer.tokenize(data[0].answers_text)

        all_doc_tokens = []                 
        tok_to_orig_index = []              
        orig_to_tok_index = []              
        
        for (i, token) in enumerate(data[0].doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)


        # The -3 accounts for [CLS], [SEP] and [SEP] 
        max_tokens_for_doc = max_seq_length - max_query_length - 3 

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            context_token_text = ''
            answer_token_text = ''

            tokens.append("[CLS]")
            segment_ids.append(0)
            
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                context_token_text += all_doc_tokens[split_token_index] + ' '
                segment_ids.append(0)
            
            tokens.append("[SEP]")
            segment_ids.append(0)
         
            tokens.append("[MASK]")
            segment_ids.append(2)            




            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            check_symbol = 0
            for token_index, token_id in enumerate(input_ids):
                if token_id == 99:
                    check_symbol += 1
                    segment_ids[token_index] = 1
                    continue

                elif check_symbol == 1:
                    segment_ids[token_index] = 1
            
            if len(doc_spans) > 1 and check_symbol != 2:
                print("symbol error")
                if doc_span_index == len(doc_spans) - 1:
                    if check_symbol == 0:
                        print(data[0].doc_tokens)
                        print(data[0].answers_text)
                        print(check_symbol)
                        print(input_ids)
                        print('HL error')
                        exit()
                    else:     
                        insert_num = max_tokens_for_doc - len(input_ids) + 3 #[CLS] [SEP] [MASK]

                        for num, pre_input_id in enumerate(pre_input_ids[-insert_num - 2:-2]):
                            input_ids.insert(num + 1,pre_input_id)

                        segment_ids = []
                        segment_ids = [0] * len(input_ids)

                        check_symbol = 0
                        for token_index, token_id in enumerate(input_ids):
                            if token_id == 99:
                                check_symbol += 1
                                segment_ids[token_index] = 1
                                continue

                            elif check_symbol == 1:
                                segment_ids[token_index] = 1

                        segment_ids[-1] = 2
                else:
                    pre_input_ids = input_ids.copy()
                    continue

            input_mask = [1] * len(input_ids)
            label_pos = len(input_ids)
            
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)


            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_pos = label_pos
                    ))
            break
        return features

    def _check_is_max_context(self, doc_spans, cur_span_index, position):

        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index