from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
config=AutoConfig.from_pretrained('/checkpoint',trust_remote_code=True,local_files_only=True)
cls=get_class_from_dynamic_module(config.auto_map['AutoModel'],'/checkpoint',local_files_only=True)
assert cls.__name__=='MiBoTForActionGeneration'
print('PINNED_MODEL_CLASS_IMPORT_OK',cls.__name__,flush=True)
