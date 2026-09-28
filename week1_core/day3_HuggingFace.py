import torch
from transformers import AutoModel, AutoTokenizer

#get_device
def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

#get_model_tokenizer

def get_model_tokenizer(model_name,device):
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)

    model=model.to(device)
    model.eval()
    return model,tokenizer


#get_inputs

def get_inputs(tokenizer,text,device):
    inputs=tokenizer(
        text,padding=True,truncation=True,return_tensors='pt'
    )
    inputs={
        key:value.to(device)
        for key,value in inputs.items()
    }

    return inputs

#get_outputs
def get_outputs(model,inputs):
    with torch.no_grad():
        outputs=model(**inputs)
    return outputs

#main()
def main():
    model_name='bert-base-uncased'
    device=get_device()
    text=['I love AI ','I love Computer Vision','I love Machine learning and Deep learning']
    try:
        model,tokenizer=get_model_tokenizer(model_name=model_name,device=device)
    except Exception as e:
        print(f"Error : {e}")
        return

    try:
        inputs=get_inputs(tokenizer=tokenizer,text=text,device=device)
    except Exception as e:
        print(f"Error : {e}")
        return

    try:
        outputs=get_outputs(model=model,inputs=inputs)
    except Exception as e :
        print(f"Error : {e}")
        return

    last_hidden_state=outputs.last_hidden_state

    print(f"input_ids.shape : {inputs['input_ids'].shape}")
    print(f"attention_mask.shape:{inputs['attention_mask'].shape}")

    print(f"last_hidden_state.shape is :{last_hidden_state.shape}")
if __name__=="__main__":
  main()