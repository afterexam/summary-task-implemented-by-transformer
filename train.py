from transformer import Transformer
from torch.utils.data import DataLoader
from my_dataset import ds_train,tokenizer
from config import *
import torch
from pathlib import Path
from tqdm import tqdm

from torch import nn
from torch.nn.utils import clip_grad_norm_

device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
elif (device == 'mps'):
    print(f"Device name: <mps>")
device = torch.device(device)

dataloader_train = DataLoader(
    ds_train,           # 必须：数据集对象
    batch_size=config['batch_size'],     # 每个小批量数据的大小
    shuffle=True,      # 是否打乱数据
    num_workers=4,     # 使用多少线程加载数据
    drop_last=True     # 是否丢弃最后一个不完整批次
)

transformer = Transformer(tokenizer.get_vocab_size(),config['src_seq_len'],config['tgt_seq_len'],
                          config['d_model'],config['h'],config['N'],dropout=0.1
                          ).to(device)

Path(config['model_file']).mkdir(exist_ok=True,parents=True)
# 替换原有优化器
if "grad_clip" in config:
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01  # 添加权重衰减
    )
else:
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['lr'], eps=1e-9)

model_file = get_best_file_path()
print('model:',model_file)
if model_file:
    print(f'Preloading model {model_file}')
    state = torch.load(model_file)
    transformer.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
else:
    initial_epoch = 0
    global_step = 0
    print('No model to preload, starting from scratch')
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

for epoch in range(initial_epoch,config['num_epochs']):
    torch.cuda.empty_cache() # 这个不写会咋样???
    transformer.train()
    batch_iterator = tqdm(dataloader_train, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
        decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
        encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
        target = batch['target'].to(device)

        encoder_output = transformer.encode(encoder_input,encoder_mask).to(device)
        decoder_output = transformer.decode(decoder_input,encoder_output,encoder_mask,decoder_mask).to(device)
        proj_output = transformer.project(decoder_output).to(device)

        # print(proj_output)
        # print(target)
        loss = loss_fn(proj_output.view(-1,tokenizer.get_vocab_size()) , target.view(-1) )
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Backpropagation the loss
        loss.backward()
        if 'grad_clip' in config and config['grad_clip'] > 0:
            # 应用梯度裁剪（基于梯度范数）
            clip_grad_norm_(
                transformer.parameters(),
                max_norm=config['grad_clip'],
                norm_type=2  # 使用L2范数
            )

        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    if config['save_one_best']:
        save_path = get_best_file_path(save=True)
    else:
        save_path = get_weights_file_path(epoch)
        pass
    torch.save({
        'epoch': epoch,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    },save_path)
