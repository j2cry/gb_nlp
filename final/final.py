import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from telegram.ext import Updater, MessageHandler
from telegram.ext.filters import Filters


# load models
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
HISTORY_LENGTH = 5


def reply(update, context):
    """  """
    FIRST, SECOND = '@@ПЕРВЫЙ@@', '@@ВТОРОЙ@@'
    
    conversation_context = context.user_data.get('conversation_context', [])
    
    if not conversation_context:
        req = f'{FIRST}{update.message.text}{SECOND}'
    else:
        req = ''.join([f'{FIRST}{r}{SECOND}{a}' for r, a in conversation_context]) + FIRST + update.message.text + SECOND


    # generate answer
    inputs = tokenizer(req, return_tensors='pt')
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=3,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40
    )
    response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids][0]
    answer = re.sub('@@.*@@', '', response[len(req):])
    # send reply
    update.message.reply_text(answer)
    # update context
    print(f'Q: {update.message.text}')
    print(f'A: {answer}')
    print(*conversation_context, sep='\n')
    conversation_context.append((update.message.text, answer))
    context.user_data['conversation_context'] = conversation_context[-HISTORY_LENGTH:]


if __name__ == '__main__':
    # init bot updater
    updater = Updater(token='...')
    dispatcher = updater.dispatcher
    # init handlers
    dispatcher.add_handler(MessageHandler(Filters.text, reply))
    # run bot
    updater.start_polling()
    updater.idle()
