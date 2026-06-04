# Bronoskins Bot

Мультиагентный RAG-чат-бот для ВКонтакте сети сервисных центров **Bronoskins**.
Интегрирован с корпоративной базой знаний и API **YCLIENTS** для записи клиентов.

## Возможности

- **Умные продажи**: бот не просто отвечает — он ведёт диалог по маркетинговым скриптам, увеличивает средний чек и записывает на услугу
- **RAG-поиск**: ответы на основе базы знаний компании (инструкции, гарантия, цены, виды плёнок)
- **Интеграция с YCLIENTS**: просмотр свободных слотов и запись клиентов в реальном времени
- **Мультиагентная архитектура**: оркестратор координирует работу специализированных агентов (Sales, RAG, Critic, YCLIENTS)
- **Асинхронность**: все вызовы API (VK, YandexGPT, YCLIENTS) выполняются асинхронно

## Технологический стек

| Компонент | Технология |
|-----------|-----------|
| Язык | Python 3.10+ (asyncio) |
| VK API | vkbottle |
| LLM | YandexGPT |
| RAG | LangChain, ChromaDB / Qdrant |
| Эмбеддинги | intfloat/multilingual-e5-large |
| Конфигурация | pydantic-settings |
| Логирование | loguru |

## Быстрый старт

### 1. Клонирование и установка

```bash
cd bronoskins_bot
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Настройка

Скопируйте `.env` и заполните обязательные переменные:

```bash
cp .env.example .env
# Отредактируйте .env — укажите VK_TOKEN, YANDEXGPT_API_KEY, YANDEXGPT_FOLDER_ID
```

**Обязательные переменные:**
- `VK_TOKEN` — токен группы ВКонтакте
- `YANDEXGPT_API_KEY` — API-ключ Yandex Cloud
- `YANDEXGPT_FOLDER_ID` — ID каталога Yandex Cloud

**Опциональные (для записи через YCLIENTS):**
- `YCLIENTS_TOKEN` — partner_token из YCLIENTS
- `YCLIENTS_COMPANY_ID` — ID компании в YCLIENTS

### 3. Индексация базы знаний

Поместите документы в папку `data/` и запустите индексацию:

```bash
python -m src.rag.indexer --clear
```

При первом запуске модель эмбеддингов `intfloat/multilingual-e5-large` скачается автоматически (~2 ГБ).

### 4. Production: Qdrant (опционально)

Для production рекомендуется использовать Qdrant через Docker:

```bash
docker-compose up -d
```

И установите в `.env`:
```
VECTOR_STORE_BACKEND=qdrant
```

### 5. Запуск

```bash
python -m src.main
```

## Структура проекта

```
bronoskins_bot/
├── .env                  # Переменные окружения
├── docker-compose.yml    # Qdrant + Redis (опционально)
├── requirements.txt      # Зависимости
├── data/                 # Документы базы знаний
├── src/
│   ├── main.py           # Точка входа
│   ├── agents/           # Агенты мультиагентной системы
│   │   ├── orchestrator.py   # Оркестратор
│   │   ├── sales_agent.py    # Системный промпт и цены
│   │   ├── rag_agent.py      # Поиск в базе знаний
│   │   ├── critic_agent.py   # Оценка релевантности
│   │   ├── yclients_agent.py # Интеграция с YCLIENTS
│   │   └── tools.py          # Инструменты для CrewAI
│   ├── bot/              # VK-бот
│   │   ├── vk_client.py
│   │   ├── handlers.py
│   │   └── middleware.py
│   ├── llm/              # LLM-клиенты
│   │   ├── base.py
│   │   ├── yandexgpt.py
│   │   └── openrouter.py
│   ├── rag/              # RAG-модуль
│   │   ├── indexer.py
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   └── utils/            # Утилиты
│       ├── config.py
│       └── logger.py
```

## Архитектура агентов

```
Пользователь (ВК) → VK Bot (vkbottle) → Оркестратор
                                            ↓
                              Sales Agent (LLM + системный промпт)
                                 ↓              ↓
                          RAG Agent → Critic   YCLIENTS Agent
                                 ↓              ↓
                          Генератор ответа → VK Bot → Пользователь
```

## Лицензия

Проприетарное ПО. Все права принадлежат Bronoskins.
