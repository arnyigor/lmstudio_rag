#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Knowledge Base Manager v3.0
Архитектура: Python Native Client + LM Studio API + ChromaDB Persistent
Функции:
- Умный чанкинг (Recursive separation)
- Инкрементальное обновление (Hash check)
- Полное управление коллекциями (Switch/Clear/Drop)
- Пакетная обработка (Batch embedding)
"""

import hashlib
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Библиотеки
import chromadb
import pypdf
from colorama import init, Fore, Style
from docx import Document
from openai import OpenAI
from tqdm import tqdm

# Инициализация цветов консоли
init(autoreset=True)

# --- КОНФИГУРАЦИЯ ---
@dataclass
class Config:
    # Путь к базе данных (Persistent storage)
    # Должен совпадать с тем, что прописан в MCP settings
    CHROMA_PATH: Path = Path(os.getenv("CHROMA_DB_PATH", r"C:\chroma_db"))

    # Дефолтная папка с документами
    DEFAULT_DOCS_DIR: Path = Path(os.getenv("DEFAULT_DOCS_DIR", r"С:\Documents"))

    # Настройки LM Studio
    LM_STUDIO_URL: str = "http://localhost:1234/v1"
    API_KEY: str = "lm-studio" # Локально не проверяется, но нужен для клиента
    # Модель эмбеддинга (имя может быть любым, LM Studio использует загруженную, но лучше указывать явно)
    EMBEDDING_MODEL: str = "nomic-embed-text-v1.5"

    # Параметры нарезки текста
    CHUNK_SIZE: int = 1000      # Размер кусочка текста
    CHUNK_OVERLAP: int = 200    # Перекрытие для сохранения контекста
    BATCH_SIZE: int = 50        # Размер пачки для отправки в API (влияет на скорость)

config = Config()

# --- ЛОГИКА ОБРАБОТКИ ТЕКСТА ---

class TextProcessor:
    """Класс для чтения файлов и умной нарезки текста"""

    @staticmethod
    def read_file(file_path: Path) -> str:
        """Читает файл в зависимости от расширения с обработкой ошибок"""
        ext = file_path.suffix.lower()
        try:
            if ext in [".txt", ".md", ".py", ".json", ".xml", ".html"]:
                return file_path.read_text(encoding="utf-8", errors='replace')

            elif ext == ".pdf":
                text = ""
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                return text

            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])

            else:
                return "" # Пропуск неподдерживаемых форматов без ошибки

        except Exception as e:
            print(f"{Fore.RED}[READ ERROR] Файл {file_path.name}: {e}")
            return ""

    @staticmethod
    def recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Рекурсивный сплиттер. Пытается разбить текст по смысловым разделителям.
        Приоритет: \n\n (абзац) -> \n (строка) -> . (предложение) -> " " (слово)
        """
        if not text:
            return []

        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size

            if end >= text_len:
                chunk = text[start:].strip()
                if chunk: chunks.append(chunk)
                break

            # Ищем лучший разделитель в конце чанка (в зоне overlap)
            best_split = -1
            search_start = max(start, end - overlap)
            search_area = text[search_start:end]

            for sep in separators:
                if sep == "": # Если ничего не нашли, режем жестко
                    best_split = end
                    break

                idx = search_area.rfind(sep)
                if idx != -1:
                    # Корректируем позицию с учетом search_start и длины разделителя
                    best_split = search_start + idx + len(sep)
                    break

            # Добавляем чанк
            chunk = text[start:best_split].strip()
            if chunk:
                chunks.append(chunk)

            # Следующий старт сдвигаем назад на overlap, но не дальше точки разрыва
            # В рекурсивном методе overlap формируется естественным путем за счет логики,
            # но здесь мы используем упрощенный sliding window по разделителям.
            start = best_split

            # Защита от зацикливания (если разделитель не сдвинул нас вперед)
            if start <= (end - chunk_size):
                start = end - overlap

        return chunks

    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """MD5 хэш файла для проверки изменений"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception:
            return "error_hash"

# --- МЕНЕДЖЕР БАЗЫ ДАННЫХ ---

class VectorDBManager:
    def __init__(self):
        # Создаем папку БД, если нет
        config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        print(f"{Fore.CYAN}[INIT] База данных: {config.CHROMA_PATH}")

        # Инициализация клиентов
        try:
            self.client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
            self.openai_client = OpenAI(base_url=config.LM_STUDIO_URL, api_key=config.API_KEY)
        except Exception as e:
            print(f"{Fore.RED}[CRITICAL] Ошибка инициализации: {e}")
            sys.exit(1)

        # Дефолтная коллекция
        self.collection_name = "main_collection"
        self._refresh_collection_obj()

    def _refresh_collection_obj(self):
        """Обновляет объект коллекции (безопасно)"""
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def check_lm_studio(self) -> bool:
        """Пинг LM Studio"""
        try:
            self.openai_client.models.list()
            return True
        except Exception:
            print(f"{Fore.RED}[ERROR] Нет связи с LM Studio (localhost:1234).")
            print(f"{Fore.YELLOW}Подсказка: Запустите 'Local Server' в LM Studio и загрузите Embedding Model.")
            return False

    def set_collection(self, name: str):
        """Переключение рабочей коллекции"""
        self.collection_name = name
        self._refresh_collection_obj()
        print(f"{Fore.GREEN}[OK] Текущая коллекция: {name}")

    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]

    def truncate_collection(self):
        """Очистить данные, оставить коллекцию"""
        try:
            self.client.delete_collection(self.collection_name)
            self._refresh_collection_obj() # Создаст пустую заново
            print(f"{Fore.YELLOW}[WARN] Коллекция '{self.collection_name}' очищена.")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Ошибка очистки: {e}")

    def drop_collection(self):
        """Удалить коллекцию полностью"""
        target = self.collection_name
        if target == "main_collection":
            print(f"{Fore.RED}[ERROR] Системную коллекцию 'main_collection' нельзя удалить. Только очистить.")
            return

        try:
            self.client.delete_collection(target)
            print(f"{Fore.YELLOW}[WARN] Коллекция '{target}' удалена безвозвратно.")
            self.set_collection("main_collection") # Возврат к безопасной базе
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Не удалось удалить: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получение векторов пачкой (Batch Processing)"""
        # Убираем переносы строк, которые могут сбить модель
        cleaned_texts = [t.replace("\n", " ") for t in texts]
        try:
            resp = self.openai_client.embeddings.create(
                input=cleaned_texts,
                model=config.EMBEDDING_MODEL
            )
            # Сортируем по индексу, чтобы порядок векторов совпадал с текстом
            data = sorted(resp.data, key=lambda x: x.index)
            return [item.embedding for item in data]
        except Exception as e:
            print(f"{Fore.RED}[API ERROR] Ошибка LM Studio: {e}")
            raise e

    def file_needs_update(self, filename: str, current_hash: str) -> bool:
        """Проверка: нужно ли обновлять файл?"""
        try:
            # Берем 1 чанк этого файла, чтобы проверить хэш
            res = self.collection.get(
                where={"source": filename},
                include=["metadatas"],
                limit=1
            )
            if not res['metadatas']:
                return True # Файла нет в базе

            stored_hash = res['metadatas'][0].get("file_hash", "")
            return stored_hash != current_hash
        except Exception:
            return True

    def clean_file_chunks(self, filename: str):
        """Удаляет все старые чанки конкретного файла"""
        try:
            self.collection.delete(where={"source": filename})
        except Exception:
            pass

    def ingest_files(self, folder_path: Path):
        """Основной процесс индексации папки"""
        if not folder_path.exists():
            print(f"{Fore.RED}Папка не найдена: {folder_path}")
            return

        all_files = [f for f in folder_path.iterdir() if f.is_file()]
        print(f"{Fore.CYAN}Сканирование {len(all_files)} файлов...")

        batch_docs = []
        files_processed = 0

        for file_path in all_files:
            # 1. Проверка хэша
            f_hash = TextProcessor.get_file_hash(file_path)
            if not self.file_needs_update(file_path.name, f_hash):
                print(f"{Fore.GREEN}[SKIP] {file_path.name} (без изменений)")
                continue

            # 2. Если изменился - удаляем старое, читаем новое
            print(f"{Fore.YELLOW}[READ] {file_path.name}...")
            self.clean_file_chunks(file_path.name)

            text = TextProcessor.read_file(file_path)
            if not text: continue

            chunks = TextProcessor.recursive_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

            # 3. Готовим данные для БД
            for i, chunk_text in enumerate(chunks):
                # Уникальный ID: ИмяФайла_ИндексЧанка
                chunk_id = f"{file_path.name}_{i}"
                batch_docs.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": file_path.name,
                        "file_hash": f_hash,
                        "chunk_index": i
                    }
                })
            files_processed += 1

        # 4. Пакетная загрузка (Batch Upload)
        total_chunks = len(batch_docs)
        if total_chunks > 0:
            print(f"{Fore.CYAN}[EMBED] Генерация векторов для {total_chunks} фрагментов...")

            # Разбиваем на батчи по BATCH_SIZE (например, по 50 штук)
            for i in tqdm(range(0, total_chunks, config.BATCH_SIZE), desc="Загрузка в БД"):
                batch = batch_docs[i : i + config.BATCH_SIZE]

                ids = [d["id"] for d in batch]
                texts = [d["text"] for d in batch]
                metas = [d["metadata"] for d in batch]

                try:
                    # Генерируем векторы
                    embeddings = self.get_embeddings(texts)
                    # Пишем в Chroma
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metas
                    )
                except Exception:
                    print(f"{Fore.RED}[ERROR] Сбой на пакете {i}")

            print(f"{Fore.GREEN}[DONE] Успешно обновлено файлов: {files_processed}")
        else:
            print("Нет новых данных для индексации.")

# --- UI / ИНТЕРФЕЙС ---

def manage_collections_menu(db: VectorDBManager):
    while True:
        colls = db.list_collections()
        print(f"\n{Fore.BLUE}--- Управление коллекциями ---")
        for idx, name in enumerate(colls, 1):
            marker = "*" if name == db.collection_name else " "
            print(f"{marker} {idx}. {name}")

        print("\nДействия: (N)ew, (S)witch, (D)elete, (B)ack")
        cmd = input("Выбор: ").strip().lower()

        if cmd == 'b': break

        elif cmd == 'n':
            name = input("Имя новой коллекции: ").strip()
            if name: db.set_collection(name)

        elif cmd == 's':
            try:
                idx = int(input("Номер коллекции: ")) - 1
                if 0 <= idx < len(colls):
                    db.set_collection(colls[idx])
                else: print("Неверный номер.")
            except ValueError: pass

        elif cmd == 'd':
            confirm = input(f"{Fore.RED}Удалить '{db.collection_name}' НАСОВСЕМ? (y/n): ").lower()
            if confirm in ['y', 'yes']:
                db.drop_collection()

def main():
    db = VectorDBManager()

    if not db.check_lm_studio():
        return

    while True:
        print(f"\n{Style.BRIGHT}=== RAG MANAGER v3.0 [{Fore.GREEN}{db.collection_name}{Style.RESET_ALL}] ===")
        print("1. Индексировать папку (Smart Update)")
        print("2. Управление коллекциями (Создать/Выбрать/Удалить)")
        print("3. Очистить текущую коллекцию (Truncate)")
        print("4. Статистика и проверка")
        print("5. Выход")

        choice = input(f"{Fore.YELLOW}>>> ").strip()

        if choice == '1':
            path_input = input(f"Путь к файлам [{config.DEFAULT_DOCS_DIR}]: ").strip()
            target_path = Path(path_input) if path_input else config.DEFAULT_DOCS_DIR
            db.ingest_files(target_path)

        elif choice == '2':
            manage_collections_menu(db)

        elif choice == '3':
            confirm = input(f"{Fore.RED}Вы уверены, что хотите очистить данные в '{db.collection_name}'? (y/n): ").lower()
            if confirm in ['y', 'yes']:
                db.truncate_collection()

        elif choice == '4':
            cnt = db.collection.count()
            print(f"{Fore.CYAN}Всего векторов в '{db.collection_name}': {cnt}")
            if cnt > 0:
                print("Пример записи:", db.collection.get(limit=1)['metadatas'])

        elif choice == '5':
            print("Выход.")
            break
        else:
            print("Неверная команда.")

if __name__ == "__main__":
    main()