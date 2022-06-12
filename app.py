from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import streamlit as st
import joblib

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def download_model():
    model = joblib.load('./MBart50/Models/model.pkl')
    tokenizer = joblib.load('./MBart50/Models/tokenizer.pkl')
    return model, tokenizer

st.title('English to Hindi Translator')
text = st.text_area("Enter Text:", value='', height=None, max_chars=None, key=None)
model, tokenizer = download_model()

if st.button('Translate to Hindi'):
    if text == '':
        st.write('Please enter English text for translation') 
    else: 
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer.src_lang = "en_XX"
        encoded_english_text = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_english_text, forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"])
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        st.write('', str(out).strip('][\''))
else: pass