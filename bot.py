import os
import json
import random
import io
import csv
import logging
import asyncio
import requests
from google import genai
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Переменные окружения
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Инициализация клиента Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# Источники данных
WORDS_URLS = [
    "https://www.dropbox.com/scl/fi/1tp1lxyy5eq00n7z2wg6h/minna.csv?rlkey=oh8nkj341lre3comk7awhwdho&dl=1",
    "https://www.dropbox.com/scl/fi/r2idrj4m3qrapx7j04q1m/kanji_2.csv?rlkey=ivix5uiotyon9jur1a7yc5bfk&dl=1",
    "https://www.dropbox.com/scl/fi/yjnlyu02y922hi15tckti/kanji.csv?rlkey=dqp2ddkel0klgaqy0y4z9zmvi&dl=1",
    "https://www.dropbox.com/scl/fi/os5lxmhrm96x1f3k2557q/add.csv?rlkey=srdaf5h8lc7klrcf9nsv703iu&dl=1",
]
GRAMMAR_URL = "https://www.dropbox.com/scl/fi/mq66pulqe64jtk3lrnf7d/grammar.csv?rlkey=u8qocxrn3027clihkygiazcmn&dl=1"

# Файлы хранения
ERRORS_FILE = "errors.json"
TASKS_FILE = "active_tasks.json"


# ==============================================================================
# Вспомогательные функции для работы с JSON и файлами
# ==============================================================================

def load_json(filepath: str, default: dict) -> dict:
    """Загружает JSON из файла, возвращает default при ошибке или отсутствии."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Не удалось загрузить {filepath}: {e}. Используется значение по умолчанию.")
        return default


def save_json(filepath: str, data: dict) -> None:
    """Сохраняет данные в JSON файл."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения {filepath}: {e}")


def init_files() -> None:
    """Создает файлы по умолчанию при их отсутствии."""
    if not os.path.exists(ERRORS_FILE):
        save_json(ERRORS_FILE, {"words": {}, "grammar": {}})
    if not os.path.exists(TASKS_FILE):
        save_json(TASKS_FILE, {})


# ==============================================================================
# Вспомогательные функции для загрузки данных
# ==============================================================================

def fetch_words() -> list:
    logging.info("Скачивает и объединяет все CSV со словами, удаляя дубликаты по Japanese")
    """Скачивает и объединяет все CSV со словами, удаляя дубликаты по Japanese."""
    all_words = {}
    for url in WORDS_URLS:
        try:
            logging.info("Качаем " + url)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            logging.info("Успешный запрос по " + url)
            f = io.StringIO(resp.text)
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                logging.info("Парсим строку " + row)
                jp = row.get('Japanese', '').strip()
                if jp and jp not in all_words:
                    try:
                        interval = float(row.get('Interval', 0) or 0)
                        ease = float(row.get('Ease', 100) or 100)
                    except (ValueError, TypeError):
                        interval = 0
                        ease = 100
                    all_words[jp] = {
                        'Japanese': jp,
                        'Russian': row.get('Russian', ''),
                        'Interval': interval,
                        'Ease': ease
                    }
        except Exception as e:
            logger.error(f"Ошибка скачивания слов по URL {url}: {e}")
    return list(all_words.values())


def fetch_grammar() -> list:
    logging.info("Скачивает CSV с грамматикой")
    """Скачивает CSV с грамматикой."""
    try:
        resp = requests.get(GRAMMAR_URL, timeout=10)
        resp.raise_for_status()
        logging.info("Успешно скачали " + GRAMMAR_URL)
        f = io.StringIO(resp.text)
        reader = csv.DictReader(f, delimiter=';')
        grammar_list = []
        for row in reader:
            logging.info("Парсим строчку " + row)
            front = row.get('Front', '').strip()
            if front:
                grammar_list.append({
                    'Front': front,
                    'Back': row.get('Back', '')
                })
        return grammar_list
    except Exception as e:
        logger.error(f"Ошибка скачивания грамматики: {e}")
        return []


def weighted_sample_without_replacement(population: list, weights: list, k: int) -> list:
    """Выбирает k элементов с учетом весов, удаляя выбранные из пула."""
    result = []
    pop = list(population)
    w = list(weights)
    for _ in range(k):
        if not pop:
            break
        chosen_idx = random.choices(range(len(pop)), weights=w, k=1)[0]
        result.append(pop[chosen_idx])
        pop.pop(chosen_idx)
        w.pop(chosen_idx)
    return result


# ==============================================================================
# Вспомогательные функции для работы с Gemini
# ==============================================================================

async def call_gemini_json(prompt: str) -> dict:
    """Делает запрос к Gemini и возвращает распарсенный JSON."""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_NAME,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        text = response.text.strip()
        # Очистка от markdown-оберток, если Gemini их добавил
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
    except Exception as e:
        logger.error(f"Ошибка при запросе к Gemini: {e}")
        return None


# ==============================================================================
# Обработчики команд Telegram
# ==============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    await update.message.reply_text(
        "Привет! Я бот для практики японского языка.\n\n"
        "Используйте /task для получения задания."
    )


async def reset_errors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /reset_errors."""
    try:
        os.remove(ERRORS_FILE)
        await update.message.reply_text("Файл ошибок успешно удален.")
    except FileNotFoundError:
        await update.message.reply_text("Файл ошибок не найден, удалять нечего.")
    except Exception as e:
        logger.error(f"Ошибка при удалении файла ошибок: {e}")
        await update.message.reply_text("Произошла ошибка при удалении файла ошибок.")


async def task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /task."""
    await update.message.reply_text("Генерирую задание, пожалуйста, подождите...")

    # Шаг 1: Скачивание данных
    words = fetch_words()
    grammar = fetch_grammar()

    if not words or not grammar:
        await update.message.reply_text("Не удалось загрузить данные для задания. Попробуйте позже.")
        return

    # Шаг 4: Загрузка ошибок для расчета весов
    errors_data = load_json(ERRORS_FILE, {"words": {}, "grammar": {}})

    # Шаг 3 и 4: Расчет весов для слов
    word_weights = []
    for w in words:
        jp = w['Japanese']
        try:
            interval = float(w.get('Interval', 0) or 0)
            ease = float(w.get('Ease', 100) or 100)
            base_weight = max((1 / max(interval, 1)) * (1 / (ease / 100)), 0.05)
        except (ValueError, TypeError, ZeroDivisionError):
            base_weight = 0.5
        
        errors_count = errors_data.get("words", {}).get(jp, {}).get("count", 0)
        weight = base_weight * (1 + 0.8 * errors_count)
        word_weights.append(weight)

    # Шаг 5: Расчет весов для грамматики
    grammar_weights = []
    for g in grammar:
        front = g['Front']
        errors_count = errors_data.get("grammar", {}).get(front, {}).get("count", 0)
        weight = 1 + 0.8 * errors_count
        grammar_weights.append(weight)

    # Шаг 6: Выбор 5 разных слов и 5 разных грамматик
    selected_words = weighted_sample_without_replacement(words, word_weights, 5)
    selected_grammar = weighted_sample_without_replacement(grammar, grammar_weights, 5)

    if len(selected_words) < 5 or len(selected_grammar) < 5:
        await update.message.reply_text("Недостаточно данных для формирования полного задания.")
        return

    pairs = list(zip(selected_words, selected_grammar))

    # Шаг 7: Формирование промпта и запрос к Gemini
    prompt = f"""Сгенерируй 5 русских предложений для задания по японскому языку.
Для каждой из 5 пар (Слово, Грамматика) создай одно русское предложение.
Требования:
- До 25 слов.
- Естественное и осмысленное.
- Обязательно используй значение японского слова (указано в скобках).
- Обязательно требуй от пользователя использования указанной японской грамматики в его будущем японском переводе (указано в скобках).

Пары:
1. Слово: {pairs[0][0]['Japanese']} ({pairs[0][0]['Russian']}), Грамматика: {pairs[0][1]['Front']}
2. Слово: {pairs[1][0]['Japanese']} ({pairs[1][0]['Russian']}), Грамматика: {pairs[1][1]['Front']}
3. Слово: {pairs[2][0]['Japanese']} ({pairs[2][0]['Russian']}), Грамматика: {pairs[2][1]['Front']}
4. Слово: {pairs[3][0]['Japanese']} ({pairs[3][0]['Russian']}), Грамматика: {pairs[3][1]['Front']}
5. Слово: {pairs[4][0]['Japanese']} ({pairs[4][0]['Russian']}), Грамматика: {pairs[4][1]['Front']}

Верни строго JSON в формате:
{{
  "tasks": [
    {{"id": 1, "sentence": "текст предложения 1"}},
    {{"id": 2, "sentence": "текст предложения 2"}},
    {{"id": 3, "sentence": "текст предложения 3"}},
    {{"id": 4, "sentence": "текст предложения 4"}},
    {{"id": 5, "sentence": "текст предложения 5"}}
  ]
}}
"""
    
    gemini_result = await call_gemini_json(prompt)
    if not gemini_result or "tasks" not in gemini_result:
        await update.message.reply_text("Ошибка при генерации задания. Попробуйте позже.")
        return

    tasks_list = gemini_result["tasks"]
    
    # Шаг 8: Отправка пользователю
    response_text = "\n".join([f"{t['id']}. {t['sentence']}" for t in tasks_list])
    await update.message.reply_text(f"Переведите следующие предложения на японский язык:\n\n{response_text}")

    # Шаг 9: Сохранение задания
    active_tasks = load_json(TASKS_FILE, {})
    user_id = str(update.effective_user.id)
    active_tasks[user_id] = {
        "sentences": [t["sentence"] for t in tasks_list],
        "words": [w["Japanese"] for w, g in pairs],
        "grammar": [g["Front"] for w, g in pairs]
    }
    save_json(TASKS_FILE, active_tasks)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений для проверки ответа."""
    user_id = str(update.effective_user.id)
    active_tasks = load_json(TASKS_FILE, {})

    # Шаг 1: Поиск задания
    if user_id not in active_tasks:
        await update.message.reply_text("Сначала используйте /task.")
        return

    task = active_tasks[user_id]
    user_answer = update.message.text

    # Шаг 2: Формирование промпта для проверки
    prompt = f"""Проверь ответы пользователя на задание по японскому языку.

Ожидаемые русские предложения для перевода:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(task['sentences']))}

Ожидаемые японские слова, которые должны быть использованы в переводе:
{', '.join(task['words'])}

Ожидаемая японская грамматика, которая должна быть использована в переводе:
{', '.join(task['grammar'])}

Ответ пользователя:
{user_answer}

Проанализируй, использовал ли пользователь правильные японские слова и грамматику в своем переводе.
Верни строго JSON в формате:
{{
  "feedback": "подробный разбор на русском языке",
  "wrong_words": ["слово1"],
  "wrong_grammar": ["грамматика1"]
}}
Если ошибок нет, массивы wrong_words и wrong_grammar должны быть пустыми [].
"""

    # Шаг 3: Запрос к Gemini
    result = await call_gemini_json(prompt)
    if not result:
        await update.message.reply_text("Ошибка при проверке ответа. Попробуйте позже.")
        return

    feedback = result.get("feedback", "Нет обратной связи.")
    wrong_words = result.get("wrong_words", [])
    wrong_grammar = result.get("wrong_grammar", [])

    # Шаг 4 и 5: Обновление errors.json
    errors_data = load_json(ERRORS_FILE, {"words": {}, "grammar": {}})

    for w in task["words"]:
        if w not in errors_data["words"]:
            errors_data["words"][w] = {"count": 0, "success_streak": 0}
        
        if w in wrong_words:
            errors_data["words"][w]["count"] += 1
            errors_data["words"][w]["success_streak"] = 0
        else:
            errors_data["words"][w]["success_streak"] += 1
            if errors_data["words"][w]["success_streak"] >= 3:
                errors_data["words"][w]["count"] = 0
                errors_data["words"][w]["success_streak"] = 0

    for g in task["grammar"]:
        if g not in errors_data["grammar"]:
            errors_data["grammar"][g] = {"count": 0, "success_streak": 0}
        
        if g in wrong_grammar:
            errors_data["grammar"][g]["count"] += 1
            errors_data["grammar"][g]["success_streak"] = 0
        else:
            errors_data["grammar"][g]["success_streak"] += 1
            if errors_data["grammar"][g]["success_streak"] >= 3:
                errors_data["grammar"][g]["count"] = 0
                errors_data["grammar"][g]["success_streak"] = 0

    save_json(ERRORS_FILE, errors_data)

    # Шаг 6: Удаление активного задания
    del active_tasks[user_id]
    save_json(TASKS_FILE, active_tasks)

    # Шаг 7: Отправка feedback пользователю
    await update.message.reply_text(feedback)


# ==============================================================================
# Основная функция запуска
# ==============================================================================

def main() -> None:
    """Запуск бота."""
    init_files()

    # Стандартное прямое подключение к Telegram API
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )
    logger.info("Бот запущен без прокси (напрямую).")

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset_errors", reset_errors))
    app.add_handler(CommandHandler("task", task))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск polling
    logger.info("Запуск polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
