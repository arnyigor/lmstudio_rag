#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль локального менеджера базы знаний (RAG).

Поддержка «путь по умолчанию» для:
    * Хранилища ChromaDB (`CHROMA_PATH`);
    * Каталога с документами, который вводит пользователь.

Используются переменные окружения:
    * CHROMA_DB_PATH – путь к каталогу ChromaDB
    * DEFAULT_DOCS_DIR – каталог по умолчанию для документов
"""

import hashlib
import os
from pathlib import Path
import sys
from typing import List, Dict, Any

import chromadb
import pypdf
from colorama import init, Fore, Style
from docx import Document
from openai import OpenAI
from tqdm import tqdm  # Прогресс бар

# Инициализация цветов консоли
init(autoreset=True)

# ------------------------------------------------------------------
# Конфигурация: пути по умолчанию + переменные окружения
# ------------------------------------------------------------------

DEFAULT_DB_PATH = Path(r"G:\AIModels\chroma_db\chroma_db")
CHROMA_PATH: Path = Path(os.getenv("CHROMA_DB_PATH", str(DEFAULT_DB_PATH)))

DEFAULT_DOCS_DIR = Path(r"G:\Android\ChromaDbDocuments")  # Можно заменить на любой каталог
DOCS_DIR: Path = Path(os.getenv("DEFAULT_DOCS_DIR", str(DEFAULT_DOCS_DIR)))

# Настройки подключения к LM Studio
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

# Настройки чанкинга (разбиения текста)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class VectorDBManager:
    """Управление коллекцией ChromaDB."""

    def __init__(self,
                 db_path: Path,
                 collection_name: str = "main_collection",
                 embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q8_0",
                 base_url: str = "http://localhost:1234/v1"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Клиент OpenAI → LM Studio
        self.embedding_client = OpenAI(base_url=base_url, api_key="lm-studio")

        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(
                f"{Fore.GREEN}[OK] Коллекция '{self.collection_name}' готова. "
                f"Всего документов: {self.collection.count()}"
            )
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Ошибка инициализации БД: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def get_embedding(self, text: str) -> List[float]:
        """Получаем эмбеддинг через LM Studio (возвращает list)."""
        try:
            resp = self.embedding_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
                # encoding_format="float"  – можно явно указать, но это по умолчанию
            )
            embedding = resp.data[0].embedding

            if isinstance(embedding, list):
                return embedding          # <‑ уже нужный формат
            raise TypeError(f"Unexpected embedding type: {type(embedding)}")

        except Exception as e:
            print(f"{Fore.RED}[ERROR] {e}")
            sys.exit(1)

    def delete_collection(self) -> None:
        """Удаляет коллекцию и создаёт пустую."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"{Fore.YELLOW}[WARN] Коллекция '{self.collection_name}' удалена.")
            # Пересоздаём пустую
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Не удалось удалить коллекцию: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Добавляет документы в базу пачками."""
        if not documents:
            return

        ids = [doc["id"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        documents_text = [doc["text"] for doc in documents]

        print(
            f"{Fore.CYAN}[PROCESS] Генерация эмбеддингов через LM Studio "
            f"({len(documents)} чанков)..."
        )
        embeddings: List[List[float]] = []

        # tqdm – прогресс‑бар
        for text in tqdm(documents_text, desc="Векторизация"):
            embeddings.append(self.get_embedding(text))

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"{Fore.GREEN}[OK] Успешно добавлено {len(documents)} фрагментов.")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Ошибка записи в БД: {e}")


class FileProcessor:
    """Чтение и разбиение файлов."""

    @staticmethod
    def read_file(file_path: Path) -> str:
        ext = file_path.suffix.lower()

        try:
            if ext in (".txt", ".md"):
                return file_path.read_text(encoding="utf-8")

            elif ext == ".pdf":
                text = ""
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        txt_page = page.extract_text()
                        if txt_page:  # иногда extract_text() может вернуть None
                            text += txt_page + "\n"
                return text

            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])

            else:
                print(f"{Fore.YELLOW}[SKIP] Формат {ext} не поддерживается: {file_path}")
                return ""

        except Exception as e:
            print(f"{Fore.RED}[ERROR] Не удалось прочитать файл {file_path}: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str, filename: str) -> List[Dict[str, Any]]:
        """Разбивает текст на куски (чанки)."""
        chunks = []
        if not text:
            return chunks

        total_len = len(text)
        start = 0

        while start < total_len:
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            # Создаём уникальный ID для чанка
            chunk_id = hashlib.md5((filename + str(start)).encode()).hexdigest()

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {"source": filename, "position": start},
                }
            )
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

# ------------------------------------------------------------------
# Меню и логика взаимодействия
# ------------------------------------------------------------------

def main_menu() -> str:
    """Выводит меню и возвращает выбранный пункт."""
    print(f"\n{Style.BRIGHT}=== Менеджер Локальной Базы Знаний (RAG) ===")
    print("1. Индексировать папку (Добавить файлы)")
    print("2. Удалить текущую коллекцию (Очистить базу)")
    print("3. Информация о базе")
    print("4. Выход")

    choice = input(f"{Fore.YELLOW}Выберите действие (1-4): ")
    return choice.strip()


def run() -> None:
    """Основная функция запуска."""
    # Убедимся, что каталог ChromaDB существует
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    # Имя коллекции можно задать в коде или через переменную окружения
    collection_name = os.getenv("CHROMA_COLLECTION", "my_knowledge_base")

    db = VectorDBManager(CHROMA_PATH, collection_name)
    processor = FileProcessor()

    while True:
        choice = main_menu()

        if choice == "1":
            # Путь к папке с документами: пользователь может оставить пустым – будет использован DOCS_DIR
            folder_input = input(
                f"{Fore.YELLOW}Введите путь к папке с файлами "
                f"(по умолчанию: {DOCS_DIR}): "
            ).strip()
            folder_path = Path(folder_input) if folder_input else DOCS_DIR

            if not folder_path.is_dir():
                print(f"{Fore.RED}Папка не найдена!")
                continue

            all_chunks: List[Dict[str, Any]] = []
            print(f"{Fore.CYAN}[SCAN] Сканирование файлов...")

            for file_name in os.listdir(folder_path):
                file_path = folder_path / file_name
                if not file_path.is_file():
                    continue  # пропускаем каталоги

                print(f"Обработка: {file_name}")
                text = processor.read_file(file_path)
                if text:
                    chunks = processor.chunk_text(text, file_name)
                    all_chunks.extend(chunks)

            if all_chunks:
                print(
                    f"{Fore.CYAN}Найдено {len(all_chunks)} фрагментов текста. "
                    "Начинаем загрузку..."
                )
                # Разбиваем на пакеты по 50 штук
                batch_size = 50
                for i in range(0, len(all_chunks), batch_size):
                    db.add_documents(all_chunks[i : i + batch_size])
                print(f"{Fore.GREEN}[DONE] Индексация завершена!")
            else:
                print(f"{Fore.YELLOW}Текстовых данных не найдено.")

        elif choice == "2":
            confirm = input(
                f"{Fore.RED}Вы уверены, что хотите УДАЛИТЬ ВСЕ данные из коллекции "
                f"'{collection_name}'? (yes/no): "
            ).strip().lower()
            if confirm == "yes":
                db.delete_collection()

        elif choice == "3":
            count = db.collection.count()
            print(f"{Fore.BLUE}Путь к БД: {CHROMA_PATH}")
            print(f"{Fore.BLUE}Коллекция: {collection_name}")
            print(f"{Fore.BLUE}Всего векторов: {count}")

            if count > 0:
                # Показать примеры источников
                metas = db.collection.get(limit=5)["metadatas"]
                print("Примеры источников:", metas)

        elif choice == "4":
            print("Выход.")
            break

        else:
            print(f"{Fore.RED}Неверная команда. Попробуйте снова.")


if __name__ == "__main__":
    run()
