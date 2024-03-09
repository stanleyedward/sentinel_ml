from pytorch_modules import inference_utils
from pytorch_modules import config
from pytorch_modules.model import DistilBERTClass
from transformers import DistilBertTokenizer
import torch

model = inference_utils.load_model(config.MODEL_DIR)
tokenizer = DistilBertTokenizer.from_pretrained(config.TOKENIZER_DIR, truncation=True, do_lower_case=True)

def forward(text, model, tokenizer):
    
    tokenized_text = inference_utils.clean_and_tokenize_text(text, tokenizer)
    pred = inference_utils.inference_step(tokenized_text, model)
    
    return pred

if __name__ == '__main__':
    text = "DIE LJAHS:DLKJASD*!(@&#!(P@:___)"
    pred = int(forward(text, model_dir='models/test_model.pt', tokenizer_dir='models/tokenizer/'))
    print(pred)