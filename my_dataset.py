import torch
import datasets
from tokenizers import Tokenizer
from my_tokenizer import get_or_build_tokenizer
from config import config
import jieba

from torch.utils.data import Dataset
def sentence_to_tensor(config,sentence,tokenizer:Tokenizer,type='encoder'):
    tokenizer = tokenizer
    SOS = torch.tensor([tokenizer.token_to_id('[SOS]')])
    EOS = torch.tensor([tokenizer.token_to_id('[EOS]')])
    PAD = torch.tensor([tokenizer.token_to_id('[PAD]')])
    src_len = config['src_seq_len']
    tgt_len = config['tgt_seq_len']

    inputs = torch.tensor(tokenizer.encode(sentence).ids)
    if type =='encoder':
        padding_num = src_len- len(inputs) - 2
        inputs = torch.concat(
            [
                SOS,inputs,EOS,torch.tensor([PAD] * padding_num)
            ]
        )
    elif type=='target':
        padding_num = tgt_len - len(inputs) - 1
        inputs = torch.concat(
            [
                inputs,EOS,torch.tensor([PAD] * padding_num)
            ]
        )
    elif type == 'decoder':
        padding_num = tgt_len - len(inputs) - 1

        inputs = torch.concat(
            [
                SOS, inputs , torch.tensor([PAD] * padding_num)
            ]
        )
    else:
        print('三选一错误')
        raise TypeError

    assert padding_num >= 0 , 'padding越界'
    # if inputs.dim() == 1:
    #     inputs = inputs.unsqueeze(0)
    # assert inputs.shape[1] == config['seq_len'],'长度不对'
    return inputs
class SummaryDataset(Dataset):
    def __init__(self,ds,tokenizer):
        super().__init__()
        prompt = '在本任务中，您将获得一段文本，您的任务是生成该文本的摘要。'
        ds_process = ds.map(
            lambda x:{
                "input":' '.join(jieba.cut(x["input"].replace(prompt,''))),
                "output":' '.join(jieba.cut(x["output"].replace(prompt,'')))
            }
        )
        self.ds = ds_process
        self.tokenizer = tokenizer


    def __getitem__(self, item):
        encoder_input = sentence_to_tensor(config, self.ds[item]['input'], self.tokenizer, type='encoder')
        target = sentence_to_tensor(config, self.ds[item]['output'], self.tokenizer, type='target')
        decoder_input = sentence_to_tensor(config, self.ds[item]['output'], self.tokenizer, type='decoder')
        ecoder_mask = (encoder_input!= self.tokenizer.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0)

        decoder_mask = (target!=self.tokenizer.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0) & causal_mask(config['tgt_seq_len'])
        dic = {
            "encoder_input":encoder_input ,
            'decoder_input':decoder_input,
            "target":target,
            'encoder_mask':ecoder_mask,
            'decoder_mask':decoder_mask
        }

        return self.ds[item] | dic

    def __len__(self):
        return len(self.ds)

    # def __iter__(self):
    #     # 返回迭代器对象本身
    #     self._index = 0  # 每次迭代时重置索引
    #     return self
    # def __next__(self):
    #     # 检查是否超出数据集长度
    #     if self._index >= len(self.ds):
    #         raise StopIteration
    #     # 返回当前索引的数据
    #     item = self.ds[self._index]['input']
    #     self._index += 1  # 索引前进
    #     return item



def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def visualize_masks(encoder_mask:torch.Tensor, decoder_mask:torch.Tensor):
    """可视化 mask 用于调试"""
    import matplotlib.pyplot as plt
    src_seq_len = config['src_seq_len']
    tgt_seq_len = config['tgt_seq_len']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(encoder_mask.view(src_seq_len).repeat(src_seq_len,1), cmap='binary')
    ax1.set_title('Encoder Mask')
    
    ax2.imshow(decoder_mask.view(tgt_seq_len,tgt_seq_len), cmap='binary')
    ax2.set_title('Decoder Mask')
    plt.show()


print('dataset载入...')
ds_train_raw = datasets.load_dataset(path='./data',split='train',data_files={"train":"train.parquet","test":"test.parquet"})
ds_test_raw = datasets.load_dataset(path='./data',split='test',data_files={"train":"train.parquet","test":"test.parquet"})
print('dataset载入成功...')

# 随机采样10%数据（保持类别分布）
ds_train_raw = ds_train_raw.train_test_split(
    test_size=0.9,seed=2025  # 保留10% , 每次取出来一样的数据
)['train']

tokenizer = get_or_build_tokenizer(config,ds_train_raw,type="train")
ds_train = SummaryDataset(ds_train_raw,tokenizer)

if __name__=='__main__':
    for idx in range(2):
        a = ds_train[idx]
        print(a['input'])
        visualize_masks(a['encoder_mask'],a['decoder_mask'])



# prompt = ' '.join(jieba.cut(prompt))
# a = model.encode(sentence_to_tensor(config,prompt,tokenizer))
# print(a.shape)
