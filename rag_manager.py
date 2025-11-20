#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Knowledge Base Manager v3.1
Архитектура: Python Native Client + LM Studio API + ChromaDB Persistent
Функции:
- Умный чанкинг (Recursive separation)
- Инкрементальное обновление (Hash check)
- Управление коллекциями (Lifecycle)
- Пакетная обработка (Batching)
- [NEW] Выборочная индексация (Range selection)
"""

import hashlib
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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
    # Путь к базе данных
    CHROMA_PATH: Path = Path(os.getenv("CHROMA_DB_PATH", r"G:\AIModels\chroma_db\chroma_db"))

    # Дефолтная папка с документами
    DEFAULT_DOCS_DIR: Path = Path(os.getenv("DEFAULT_DOCS_DIR", r"G:\Android\ChromaDbDocuments"))

    # Настройки LM Studio
    LM_STUDIO_URL: str = "http://localhost:1234/v1"
    API_KEY: str = "lm-studio"
    EMBEDDING_MODEL: str = "nomic-embed-text-v1.5"

    # Параметры
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    BATCH_SIZE: int = 50

config = Config()

# --- ЛОГИКА ОБРАБОТКИ ТЕКСТА ---

class TextProcessor:
    @staticmethod
    def read_file(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        try:
            if ext in [".txt", ".md", ".py", ".json", ".xml", ".html", ".java", ".kt", ".cpp"]:
                return file_path.read_text(encoding="utf-8", errors='replace')

            elif ext == ".pdf":
                text = ""
                with open(file_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted + "\n"
                return text

            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])

            else:
                return ""
        except Exception as e:
            print(f"{Fore.RED}[READ ERROR] Файл {file_path.name}: {e}")
            return ""

    @staticmethod
    def recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
        if not text: return []
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

            best_split = -1
            search_start = max(start, end - overlap)
            search_area = text[search_start:end]

            for sep in separators:
                if sep == "":
                    best_split = end
                    break
                idx = search_area.rfind(sep)
                if idx != -1:
                    best_split = search_start + idx + len(sep)
                    break

            chunk = text[start:best_split].strip()
            if chunk: chunks.append(chunk)

            start = best_split
            if start <= (end - chunk_size): start = end - overlap

        return chunks

    @staticmethod
    def get_file_hash(file_path: Path) -> str:
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
        config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        print(f"{Fore.CYAN}[INIT] База данных: {config.CHROMA_PATH}")

        try:
            self.client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
            self.openai_client = OpenAI(base_url=config.LM_STUDIO_URL, api_key=config.API_KEY)
        except Exception as e:
            print(f"{Fore.RED}[CRITICAL] Ошибка инициализации: {e}")
            sys.exit(1)

        self.collection_name = "main_collection"
        self._refresh_collection_obj()

    def _refresh_collection_obj(self):
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def check_lm_studio(self) -> bool:
        try:
            self.openai_client.models.list()
            return True
        except Exception:
            print(f"{Fore.RED}[ERROR] Нет связи с LM Studio (localhost:1234).")
            return False

    def set_collection(self, name: str):
        self.collection_name = name
        self._refresh_collection_obj()
        print(f"{Fore.GREEN}[OK] Текущая коллекция: {name}")

    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]

    def truncate_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            self._refresh_collection_obj()
            print(f"{Fore.YELLOW}[WARN] Коллекция '{self.collection_name}' очищена.")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Ошибка очистки: {e}")

    def drop_collection(self):
        target = self.collection_name
        if target == "main_collection":
            print(f"{Fore.RED}[ERROR] Системную коллекцию нельзя удалить.")
            return
        try:
            self.client.delete_collection(target)
            print(f"{Fore.YELLOW}[WARN] Коллекция '{target}' удалена.")
            self.set_collection("main_collection")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Ошибка удаления: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        cleaned_texts = [t.replace("\n", " ") for t in texts]
        try:
            resp = self.openai_client.embeddings.create(
                input=cleaned_texts, model=config.EMBEDDING_MODEL
            )
            data = sorted(resp.data, key=lambda x: x.index)
            return [item.embedding for item in data]
        except Exception as e:
            print(f"{Fore.RED}[API ERROR] Ошибка LM Studio: {e}")
            raise e

    def file_needs_update(self, filename: str, current_hash: str) -> bool:
        try:
            res = self.collection.get(where={"source": filename}, include=["metadatas"], limit=1)
            if not res['metadatas']: return True
            return res['metadatas'][0].get("file_hash", "") != current_hash
        except Exception:
            return True

    def clean_file_chunks(self, filename: str):
        try:
            self.collection.delete(where={"source": filename})
        except Exception: pass

    # --- НОВАЯ ЛОГИКА ВЫБОРА ФАЙЛОВ ---

    def _select_files_interactive(self, all_files: List[Path]) -> List[Path]:
        """
        Интерактивное меню для выбора диапазона файлов.
        Возвращает отфильтрованный список.
        """
        total = len(all_files)
        print(f"\n{Fore.BLUE}Найдено файлов: {total}")

        # Показываем первые 5 и последние 5 для ориентира
        print(f"{Style.DIM}Примеры списка:")
        limit = 5
        for i in range(min(limit, total)):
            print(f"  [{i+1}] {all_files[i].name}")
        if total > limit * 2:
            print("  ...")
        if total > limit:
            for i in range(max(limit, total - limit), total):
                print(f"  [{i+1}] {all_files[i].name}")
        print(f"{Style.RESET_ALL}")

        print("Как индексировать?")
        print("  [Enter] или 'a' -> Все файлы")
        print("  '1-5'           -> Диапазон (с 1 по 5 включительно)")
        print("  '10+'           -> С 10-го до конца")
        print("  '5'             -> Только файл №5")

        choice = input(f"{Fore.YELLOW}Выбор > ").strip().lower()

        if choice == '' or choice == 'a' or choice == 'all':
            return all_files

        selected = []

        try:
            # Парсинг диапазона 1-5
            if '-' in choice:
                start_s, end_s = choice.split('-')
                start = int(start_s) if start_s else 1
                end = int(end_s) if end_s else total

                # Корректируем под индексы Python (0-based)
                start_idx = max(0, start - 1)
                end_idx = min(total, end) # slice не включает последний элемент, но логика 1-5 подразумевает включение

                selected = all_files[start_idx : end_idx]

            # Парсинг 10+
            elif choice.endswith('+'):
                start = int(choice[:-1])
                start_idx = max(0, start - 1)
                selected = all_files[start_idx:]

            # Одиночный файл
            else:
                idx = int(choice) - 1
                if 0 <= idx < total:
                    selected = [all_files[idx]]
                else:
                    print(f"{Fore.RED}Номер файла вне диапазона.")
                    return []

        except ValueError:
            print(f"{Fore.RED}Ошибка формата. Будут обработаны ВСЕ файлы.")
            return all_files

        print(f"{Fore.CYAN}Выбрано файлов для обработки: {len(selected)}")
        return selected

    def ingest_files(self, folder_path: Path):
        if not folder_path.exists():
            print(f"{Fore.RED}Папка не найдена: {folder_path}")
            return

        # 1. Сортируем файлы, чтобы порядок был всегда одинаковым (для выбора по номерам)
        all_files = sorted([f for f in folder_path.iterdir() if f.is_file()])

        if not all_files:
            print(f"{Fore.YELLOW}Папка пуста.")
            return

        # 2. Выбор пользователя
        target_files = self._select_files_interactive(all_files)

        if not target_files:
            return

        print(f"{Fore.CYAN}Запуск анализа {len(target_files)} файлов...")

        batch_docs = []
        files_processed = 0

        # 3. Обработка выбранных файлов
        for file_path in target_files:
            f_hash = TextProcessor.get_file_hash(file_path)

            if not self.file_needs_update(file_path.name, f_hash):
                print(f"{Fore.GREEN}[SKIP] {file_path.name}")
                continue

            print(f"{Fore.YELLOW}[READ] {file_path.name}...")
            self.clean_file_chunks(file_path.name)

            text = TextProcessor.read_file(file_path)
            if not text: continue

            chunks = TextProcessor.recursive_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

            for i, chunk_text in enumerate(chunks):
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

        # 4. Загрузка в БД
        total_chunks = len(batch_docs)
        if total_chunks > 0:
            print(f"{Fore.CYAN}[EMBED] Векторизация {total_chunks} фрагментов...")

            for i in tqdm(range(0, total_chunks, config.BATCH_SIZE), desc="Загрузка"):
                batch = batch_docs[i : i + config.BATCH_SIZE]
                try:
                    self.collection.add(
                        ids=[d["id"] for d in batch],
                        embeddings=self.get_embeddings([d["text"] for d in batch]),
                        documents=[d["text"] for d in batch],
                        metadatas=[d["metadata"] for d in batch]
                    )
                except Exception:
                    print(f"{Fore.RED}[ERROR] Сбой на пакете {i}")

            print(f"{Fore.GREEN}[DONE] Обновлено файлов: {files_processed}")
        else:
            print("Нет новых данных для записи (все файлы актуальны).")

# --- UI ---

def manage_collections_menu(db: VectorDBManager):
    while True:
        colls = db.list_collections()
        print(f"\n{Fore.BLUE}--- Коллекции ---")
        for idx, name in enumerate(colls, 1):
            marker = "*" if name == db.collection_name else " "
            print(f"{marker} {idx}. {name}")

        print("\n(N)ew, (S)witch, (D)elete, (B)ack")
        cmd = input("Выбор: ").strip().lower()

        if cmd == 'b': break
        elif cmd == 'n':
            name = input("Имя: ").strip()
            if name: db.set_collection(name)
        elif cmd == 's':
            try:
                idx = int(input("Номер: ")) - 1
                if 0 <= idx < len(colls): db.set_collection(colls[idx])
            except ValueError: pass
        elif cmd == 'd':
            if input("Удалить? (y/n): ") == 'y': db.drop_collection()

def main():
    db = VectorDBManager()
    if not db.check_lm_studio(): return

    while True:
        print(f"\n{Style.BRIGHT}=== RAG MANAGER v3.1 [{Fore.GREEN}{db.collection_name}{Style.RESET_ALL}] ===")
        print("1. Индексировать папку (Выбор файлов)")
        print("2. Управление коллекциями")
        print("3. Очистить текущую коллекцию")
        print("4. Статистика")
        print("5. Выход")

        choice = input(f"{Fore.YELLOW}>>> ").strip()

        if choice == '1':
            path_input = input(f"Путь [{config.DEFAULT_DOCS_DIR}]: ").strip()
            path = Path(path_input) if path_input else config.DEFAULT_DOCS_DIR
            db.ingest_files(path)
        elif choice == '2':
            manage_collections_menu(db)
        elif choice == '3':
            if input("Очистить данные? (y/n): ") == 'y': db.truncate_collection()
        elif choice == '4':
            print(f"Всего векторов: {db.collection.count()}")
        elif choice == '5':
            break

if __name__ == "__main__":
    main()