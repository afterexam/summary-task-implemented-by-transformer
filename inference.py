from transformer import Transformer
from pathlib import Path
from my_tokenizer import get_or_build_tokenizer
import torch
from config import *
from my_dataset import SummaryDataset,causal_mask,ds_train
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
elif (device == 'mps'):
    print(f"Device name: <mps>")
else:
    print("NOTE: If you have a GPU, consider using it for training.")

device = torch.device(device)

tokenizer = get_or_build_tokenizer()


transformer = Transformer(tokenizer.get_vocab_size(),config['src_seq_len'],config['tgt_seq_len'],
                          config['d_model'],config['h'],config['N'],dropout=0.1
                          ).to(device)

model_path = get_best_file_path()

if model_path:
    state = torch.load(model_path)
    transformer.load_state_dict(state['model_state_dict'])
else:
    print('find no model , exit')
    exit(-1)


def create_dec_mask(dec_len):
    padding_mask = torch.ones((1,1,dec_len),dtype=torch.bool).to(device)
    not_look_ahead_mask = causal_mask(dec_len).unsqueeze(0).to(device)
    return padding_mask & not_look_ahead_mask
def greedy_decoding(input: str, max_length: int = config['tgt_seq_len']):
    transformer.eval()
    with torch.no_grad():
        # 1. 数据预处理
        data = [{
            "input": input,
            "output": ''
        }]
        data = Dataset.from_list(data)
        data = SummaryDataset(data, tokenizer)[0]
        encoder_input = data['encoder_input'].unsqueeze(0).to(torch.long).to(device)
        encoder_mask = data['encoder_mask'].unsqueeze(0).to(torch.long).to(device)
        # decoder_mask = data['decoder_mask'].unsqueeze(0).to(torch.long).to(device)

        # 2. 编码器生成编码输出
        encoder_output = transformer.encode(encoder_input, encoder_mask)

        # 3. 解码初始化
        decoder_input = torch.tensor(tokenizer.token_to_id('[SOS]')).unsqueeze(0).unsqueeze(0).to(torch.long).to(device)
        generated_tokens = []
        for i in range(1,max_length):
            # 解码当前序列
            
            decoder_mask = create_dec_mask(decoder_input.shape[-1])
            decoder_output = transformer.decode(decoder_input, encoder_output,encoder_mask,decoder_mask)
            proj = transformer.project(decoder_output)
            _, next_token_id = torch.max(proj[:, -1, :], dim=-1)  # 获取当前序列最后一个词的预测

            # 保存生成的词
            generated_tokens.append(next_token_id.item())

            # 如果生成了结束标志，停止解码
            if next_token_id.item() == tokenizer.token_to_id('[EOS]'):
                break

            # 动态更新 decoder_input
            decoder_input =torch.concat([decoder_input,next_token_id.unsqueeze(0)],dim=-1)

        # 将生成的 token 转回字符串
        decoded_text = tokenizer.decode(generated_tokens)
        return decoded_text.replace(' ','')


def beam_search_decoding(input: str,
                         beam_width: int = 5,
                         max_length: int = config['tgt_seq_len'],
                         length_penalty: float = 0.6):
    """
    Beam Search 解码实现
    参数：
        beam_width: 束宽，控制候选序列数量
        length_penalty: 长度惩罚系数（0.0-1.0，值越小惩罚越强）
    """
    transformer.eval()
    with torch.no_grad():
        # 数据预处理
        data = [{"input": input, "output": ''}]
        data = Dataset.from_list(data)
        data = SummaryDataset(data, tokenizer)[0]

        # 编码器处理
        encoder_input = data['encoder_input'].unsqueeze(0).to(device)
        encoder_mask = data['encoder_mask'].unsqueeze(0).to(device)
        encoder_output = transformer.encode(encoder_input, encoder_mask)

        # 扩展编码器输出以匹配束宽
        encoder_output = encoder_output.repeat(beam_width, 1, 1)  # (beam, seq, dim)
        encoder_mask = encoder_mask.repeat(beam_width, 1, 1, 1)  # (beam, 1, 1, seq)

        # 初始化束搜索
        start_token = tokenizer.token_to_id('[SOS]')
        initial_sequence = {
            'tokens': [start_token],
            'score': 0.0,
            'length': 1,
            'finished': False
        }

        # 当前候选束（按分数排序）
        current_beam = [initial_sequence]
        completed_sequences = []

        for step in range(max_length):
            # 收集所有需要处理的序列
            candidates = []
            for seq in current_beam:
                if seq['finished']:
                    completed_sequences.append(seq)
                    continue

                # 准备解码器输入
                decoder_input = torch.tensor(seq['tokens']).unsqueeze(0).to(device)

                # 生成解码器掩码
                decoder_mask = create_dec_mask(decoder_input.shape[-1]).to(device)

                # 解码步骤
                decoder_output = transformer.decode(
                    decoder_input,
                    encoder_output[:1],  # 使用第一个束的编码输出（已扩展）
                    encoder_mask[:1],
                    decoder_mask
                )

                # 获取下一个token概率
                proj = transformer.project(decoder_output[:, -1, :])  # (1, vocab)
                log_probs = torch.log_softmax(proj, dim=-1).squeeze(0)  # (vocab)

                # 保留top k候选
                topk_probs, topk_tokens = torch.topk(log_probs, beam_width)

                # 生成新候选
                for i in range(beam_width):
                    token = topk_tokens[i].item()
                    score = seq['score'] + topk_probs[i].item()

                    new_seq = {
                        'tokens': seq['tokens'] + [token],
                        'score': score,
                        'length': seq['length'] + 1,
                        'finished': (token == tokenizer.token_to_id('[EOS]'))
                    }
                    candidates.append(new_seq)

            # 选择top k候选
            candidates = sorted(
                candidates,
                key=lambda x: x['score'] / (x['length'] ** length_penalty),  # 长度标准化
                reverse=True
            )[:beam_width]

            current_beam = candidates

            # 提前终止条件
            if all(seq['finished'] for seq in current_beam):
                break

        # 合并已完成和未完成的序列
        final_candidates = completed_sequences + current_beam
        final_candidates = sorted(
            final_candidates,
            key=lambda x: x['score'] / (x['length'] ** length_penalty),
            reverse=True
        )

        # 选择最佳序列
        best_sequence = final_candidates[0]['tokens']

        # 移除EOS之后的token
        if tokenizer.token_to_id('[EOS]') in best_sequence:
            eos_index = best_sequence.index(tokenizer.token_to_id('[EOS]'))
            best_sequence = best_sequence[:eos_index]

        # 解码为文本
        decoded_text = tokenizer.decode(best_sequence[1:])  # 跳过[SOS]
        return decoded_text


for i in range(4):

    input = ds_train[i]['input']
    print(input)
    pred,target = beam_search_decoding(input),ds_train[i]['output']
    print('-'*100)
    print('pred:',pred)
    print('target:',target)
    print('-' * 100)


