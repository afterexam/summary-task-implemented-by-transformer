from pathlib import Path
config = {
# 训练相关
        "batch_size":32,
        "num_epochs":100,
        'lr':3e-4,
        "grad_clip": 1.0,
        # "warmup_steps": 8000,     # 学习率预热步数

# 结构相关
        'd_model':512,
        'N': 6,
        'h': 8,
        'src_seq_len':150,
        'tgt_seq_len':50,

# 文件相关
        'tokenizer_file':"./tokenizer/tokenizer_{0}.json",
        'model_file':"./model",
        'model_basename':"smodel_{0}.pt",
        'model_best' : "smodel_best.pt",
        "save_one_best":True,
    }

# def latest_weights_file_path():
#         model_file = config['model_file']
#         model_basename = config['model_basename'].format('*')
#         weights_files = list(Path(model_file).glob(model_basename))
#         if len(weights_files) == 0:
#                 return None
#         latest_file = max(weights_files, key=lambda f: f.stat().st_mtime)
#         return str(latest_file)
def get_weights_file_path(epoch):
        model_folder = config['model_file']
        model_filename = config['model_basename'].format(epoch)
        return str(Path('.') / model_folder / model_filename)

def get_best_file_path(save=False):
        model_folder = config['model_file']
        model_filename = config['model_best']
        model_path = Path(model_folder) / model_filename
        # 检查文件是否存在
        if save or model_path.exists():
                return str(model_path)  # 返回路径字符串
        return None  # 文件不存在时返回 None



if __name__ =="__main__":
        a = get_best_file_path()
        print(a)