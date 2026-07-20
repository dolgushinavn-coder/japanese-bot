
import os
import io
import csv
import json
import random
import requests
import google.generativeai as genai

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

WORDS_URLS = [
    "https://www.dropbox.com/scl/fi/1tp1lxyy5eq00n7z2wg6h/minna.csv?rlkey=oh8nkj341lre3comk7awhwdho&dl=1",
    "https://www.dropbox.com/scl/fi/r2idrj4m3qrapx7j04q1m/kanji_2.csv?rlkey=ivix5uiotyon9jur1a7yc5bfk&dl=1",
    "https://www.dropbox.com/scl/fi/yjnlyu02y922hi15tckti/kanji.csv?rlkey=dqp2ddkel0klgaqy0y4z9zmvi&dl=1",
    "https://www.dropbox.com/scl/fi/os5lxmhrm96x1f3k2557q/add.csv?rlkey=srdaf5h8lc7klrcf9nsv703iu&dl=1",
]

GRAMMAR_URL = "https://www.dropbox.com/scl/fi/mq66pulqe64jtk3lrnf7d/grammar.csv?rlkey=u8qocxrn3027clihkygiazcmn&dl=1"

ERRORS_FILE = "errors.json"
TASKS_FILE = "active_tasks.json"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_errors():
    return load_json(ERRORS_FILE, {"words": {}, "grammar": {}})


def load_tasks():
    return load_json(TASKS_FILE, {})


def save_tasks(tasks):
    save_json(TASKS_FILE, tasks)


def download_csv(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text), delimiter=";"))


def get_words():
    all_words = []
    seen = set()

    for url in WORDS_URLS:
        rows = download_csv(url)

        for row in rows:
            jp = row.get("Japanese", "").strip()
            if not jp or jp in seen:
                continue

            seen.add(jp)

            try:
                interval = float(row.get("Interval", 1))
                ease = float(row.get("Ease", 100))
                base_weight = max(
                    (1 / max(interval, 1)) * (1 / (ease / 100)),
                    0.05,
                )
            except Exception:
                base_weight = 0.5

            all_words.append(
                {
                    "Japanese": jp,
                    "Russian": row.get("Russian", "").strip(),
                    "base_weight": base_weight,
                }
            )

    return all_words


def get_grammar():
    return download_csv(GRAMMAR_URL)


def choose_pairs(words, grammar, errors):
    word_pool = []
    grammar_pool = []

    for w in words:
        cnt = errors["words"].get(w["Japanese"], {}).get("count", 0)
        weight = w["base_weight"] * (1 + 0.8 * cnt)
        word_pool.append((w, weight))

    for g in grammar:
        front = g["Front"]
        cnt = errors["grammar"].get(front, {}).get("count", 0)
        weight = 1 + 0.8 * cnt
        grammar_pool.append((g, weight))

    selected_words = []
    available_words = word_pool.copy()

    for _ in range(min(5, len(available_words))):
        item = random.choices(
            [x[0] for x in available_words],
            weights=[x[1] for x in available_words],
            k=1,
        )[0]
        selected_words.append(item)
        available_words = [x for x in available_words if x[0] != item]

    selected_grammar = []
    available_grammar = grammar_pool.copy()

    for _ in range(min(5, len(available_grammar))):
        item = random.choices(
            [x[0] for x in available_grammar],
            weights=[x[1] for x in available_grammar],
            k=1,
        )[0]
        selected_grammar.append(item)
        available_grammar = [x for x in available_grammar if x[0] != item]

    return list(zip(selected_words, selected_grammar))


def generate_sentences(pairs):
    payload = []
    for idx, (w, g) in enumerate(pairs, start=1):
        payload.append(
            {
                "id": idx,
                "word": w["Russian"],
                "grammar": g["Front"],
            }
        )

    prompt = f"""
Ты учитель японского языка.

Для каждого объекта создай одно естественное русское предложение длиной до 25 слов.

Обязательно используй значение слова и указанную грамматику.

Верни JSON:

{{"tasks":[{{"id":1,"sentence":"..."}}]}}

Данные:
{json.dumps(payload, ensure_ascii=False)}
"""

    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )

    return json.loads(response.text)["tasks"]


def check_translation(task_data, user_text):
    prompt = f"""
Ты проверяешь перевод японского языка.

Русские предложения:
{json.dumps(task_data["sentences"], ensure_ascii=False)}

Ожидаемые слова:
{json.dumps(task_data["words"], ensure_ascii=False)}

Ожидаемая грамматика:
{json.dumps(task_data["grammar"], ensure_ascii=False)}

Перевод пользователя:
{user_text}

Верни JSON:

{{
"feedback":"подробный разбор на русском",
"wrong_words":[],
"wrong_grammar":[]
}}
"""

    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )

    return json.loads(response.text)


def update_errors(task_data, result):
    errors = load_errors()

    wrong_words = set(result.get("wrong_words", []))
    wrong_grammar = set(result.get("wrong_grammar", []))

    for word in task_data["words"]:
        errors["words"].setdefault(
            word,
            {"count": 0, "success_streak": 0},
        )

        if word in wrong_words:
            errors["words"][word]["count"] += 1
            errors["words"][word]["success_streak"] = 0
        else:
            errors["words"][word]["success_streak"] += 1

            if errors["words"][word]["success_streak"] >= 3:
                errors["words"][word]["count"] = 0
                errors["words"][word]["success_streak"] = 0

    for grammar in task_data["grammar"]:
        errors["grammar"].setdefault(
            grammar,
            {"count": 0, "success_streak": 0},
        )

        if grammar in wrong_grammar:
            errors["grammar"][grammar]["count"] += 1
            errors["grammar"][grammar]["success_streak"] = 0
        else:
            errors["grammar"][grammar]["success_streak"] += 1

            if errors["grammar"][grammar]["success_streak"] >= 3:
                errors["grammar"][grammar]["count"] = 0
                errors["grammar"][grammar]["success_streak"] = 0

    save_json(ERRORS_FILE, errors)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет. Используй /task для получения задания."
    )


async def reset_errors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if os.path.exists(ERRORS_FILE):
        os.remove(ERRORS_FILE)

    await update.message.reply_text("Статистика ошибок сброшена.")


async def task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        words = get_words()
        grammar = get_grammar()
        errors = load_errors()

        pairs = choose_pairs(words, grammar, errors)
        generated = generate_sentences(pairs)

        sentences = [x["sentence"] for x in generated]
        words_expected = [x[0]["Japanese"] for x in pairs]
        grammar_expected = [x[1]["Front"] for x in pairs]

        text = "Переведи на японский:\n\n"
        for i, s in enumerate(sentences, start=1):
            text += f"{i}. {s}\n"

        tasks = load_tasks()
        tasks[str(update.effective_user.id)] = {
            "sentences": sentences,
            "words": words_expected,
            "grammar": grammar_expected,
        }
        save_tasks(tasks)

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(
            f"Ошибка загрузки данных: {e}"
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    tasks = load_tasks()

    if user_id not in tasks:
        return

    try:
        task_data = tasks[user_id]

        result = check_translation(
            task_data,
            update.message.text,
        )

        update_errors(task_data, result)

        del tasks[user_id]
        save_tasks(tasks)

        await update.message.reply_text(
            result.get("feedback", "Проверка завершена.")
        )

    except Exception as e:
        await update.message.reply_text(f"Ошибка проверки: {e}")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("task", task))
    app.add_handler(CommandHandler("reset_errors", reset_errors))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    app.run_polling()


if __name__ == "__main__":
    main()
