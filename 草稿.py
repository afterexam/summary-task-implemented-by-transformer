from transformers import BartForConditionalGeneration, BartTokenizer

# 加载预训练模型和分词器
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# 输入文本
text = "Your long text goes here..."

# 分词
inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

# 生成摘要
summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=200, early_stopping=True)

# 解码生成的摘要
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
