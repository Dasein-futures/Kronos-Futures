import os

# 使用 HuggingFace 镜像加速国内访问（已设置 HF_ENDPOINT 环境变量时不覆盖）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from .kronos import KronosTokenizer, Kronos, KronosPredictor

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


