🔄 🔄 Трансформеры

📝 День 1: transformers_day01

День 1 — Архитектура трансформеров и токенизация

ЦЕЛЬ ДНЯ:
Понять как работают трансформеры и научиться превращать текст в токены.

ЗАДАЧА 1: Установка библиотек

1. Установите transformers:
pip install transformers torch

2. Создайте новый Python файл или Jupyter ноутбук

3. Импортируйте нужное:
from transformers import AutoTokenizer
import torch

ЗАДАЧА 2: Загрузка токенизатора

1. Выберите модель. Для английского:
model_name = "distilbert-base-uncased"

Для русского:
model_name = "distilbert-base-multilingual-cased"

2. Загрузите токенизатор:
tokenizer = AutoTokenizer.from_pretrained(model_name)

3. Посмотрите на параметры:
print(tokenizer.vocab_size)
print(tokenizer.model_max_length)

ЗАДАЧА 3: Токенизация текста

1. Возьмите пример текста:
text = "This movie was absolutely amazing!"

2. Токенизируйте его:
tokens = tokenizer(text)
print(tokens)

3. Посмотрите на токены:
input_ids = tokens['input_ids']
print(f'Количество токенов: {len(input_ids)}')

4. Декодируйте обратно:
decoded = tokenizer.decode(input_ids)
print(f'Декодировано: {decoded}')

ЗАДАЧА 4: Работа с батчами

1. Напишите функцию для токенизации списка текстов:
def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

2. Протестируйте на нескольких текстах:
texts = [
    "This movie was great!",
    "Terrible movie, waste of time."
]

tokens = tokenize_texts(texts)
print(f'Shape: {tokens["input_ids"].shape}')

3. Посмотрите на attention_mask:
print(f'Attention mask:\n{tokens["attention_mask"]}')

ЗАДАЧА 5: Специальные токены

1. Выведите специальные токены:
print(f'CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})')
print(f'SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})')
print(f'PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')

2. Посмотрите как выглядит токенизированный текст:
single = tokenizer(text, return_tensors="pt")
print(f'Input IDs: {single["input_ids"]}')
print(f'Decoded: {tokenizer.decode(single["input_ids"][0])}')

ЗАДАЧА 6: Функция explain_tokenization

Напишите функцию, которая показывает как текст разбился на токены:

def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f'Исходный текст: {text}')
    print(f'Токены: {tokens}')
    print(f'IDs: {ids}')
    print(f'Количество: {len(tokens)}')

# Протестируйте:
explain_tokenization("Transformers are amazing!", tokenizer)

ЧЕКПОИНТ:
К концу дня вы должны иметь:
• Загруженный токенизатор
• Понимание как текст превращается в токены
• Функцию для токенизации батчей текстов
• Функцию explain_tokenization

Этот код понадобится вам в День 2 для работы с моделью.

---
Удачи с реализацией! 🚀