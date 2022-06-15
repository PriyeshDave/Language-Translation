from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json

app = FastAPI()
model = joblib.load('./MBart50/Models/model.pkl')
tokenizer = joblib.load('./MBart50/Models/tokenizer.pkl')

class language_translator(BaseModel):
  sentence : str


@app.get('/')
def entry():
  return {'Hello World'}

@app.post('/translate')
def translate(input_parameters : language_translator):
  sent = input_parameters.json()
  sent_dict = json.loads(sent)
  sentence = sent_dict['sentence']

  tokenizer.src_lang = "en_XX"
  encoded_english_text = tokenizer(sentence, return_tensors="pt")
  generated_tokens = model.generate(**encoded_english_text, forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"])
  translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  return {
    "translation" : translation,
    "success": True
  }


