#!/usr/bin/env python3
"""
Автоматизация миграции данных Mattermost из Percona 5.7 (MySQL) в PostgreSQL 13.

Требования окружения: Python 3.13, Docker, Docker Compose, dimitri/pgloader:latest,
образы percona:5.7 и postgres:13. Устанавливать зависимости не требуется.

Скрипт использует только стандартную библиотеку Python.
"""

from __future__ import annotations

import datetime
import os
import shutil
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence
import threading


# =============================
# Конфигурация (измените при необходимости)
# =============================

# Рабочая директория проекта, где лежат docker-compose.yml, дампы и артефакты
WORK_DIR = Path("/home/stfu/mattermost-migrate")

# Путь к файлу docker-compose
DOCKER_COMPOSE_FILE = WORK_DIR / "docker-compose.yml"

# Путь к исходному дампу MySQL (Percona 5.7)
MYSQL_DUMP_PATH = WORK_DIR / "mattermost.sql"

# Директория для финального дампа PostgreSQL (без сжатия)
POSTGRES_FINAL_DUMP_DIR = WORK_DIR

# Расположение и способ запуска migration-assist
# Если у вас он доступен через toolbox как в гайде — оставьте по умолчанию.
# Иначе укажите просто бинарь, например: MIGRATION_ASSIST_CMD = ["migration-assist"]
MIGRATION_ASSIST_CMD: List[str] = [
    "toolbox",
    "run",
    "-c",
    "work-stuff",
    "/home/stfu/go/bin/migration-assist",
]

# Версия Mattermost для запуска миграций в PostgreSQL (используется migration-assist)
MATTERMOST_VERSION = "v9.3"

# Параметры БД MySQL (Percona 5.7)
MYSQL_ROOT_USER = "root"
MYSQL_ROOT_PASSWORD = "pass"
MYSQL_DB_NAME = "mattermost"

# Параметры БД PostgreSQL 13
POSTGRES_USER = "mmuser"
POSTGRES_PASSWORD = "mmuserpass"
POSTGRES_DB_NAME = "mattermost"

# Имена сервисов из docker-compose.yml
MYSQL_SERVICE_NAME = "percona"
POSTGRES_SERVICE_NAME = "postgres"

# Имена контейнеров (задаются в docker-compose.yml -> container_name)
PERCONA_CONTAINER_NAME = "percona_mattermost"
POSTGRES_CONTAINER_NAME = "postgres_mattermost"

# Сетевое имя в Docker Compose (автоматически составляется как <project>_<network>)
COMPOSE_NETWORK_NAME = "mattermost-migrate-network"

# Хосты внутри docker-сети для pgloader (используем имена сервисов)
MYSQL_HOST_IN_NETWORK = MYSQL_SERVICE_NAME
POSTGRES_HOST_IN_NETWORK = POSTGRES_SERVICE_NAME

# Параметры pgloader (оптимизированы для стабильности соединений)
PGLOADER_MEMORY_MB = 8192
PGLOADER_WORKERS = 2
PGLOADER_CONCURRENCY = 1
PGLOADER_ROWS_PER_RANGE = 2500
PGLOADER_PREFETCH_ROWS = 2500
PGLOADER_BATCH_ROWS = 500

# Пути артефактов
PGLOADER_LOAD_FILE = WORK_DIR / "pgloader.load"
PGLOADER_LOG_FILE = WORK_DIR / "pgloader.log"

# Таймауты ожидания готовности контейнеров (секунды)
MYSQL_READY_TIMEOUT_S = 600
POSTGRES_READY_TIMEOUT_S = 600


# =============================
# Утилиты исполнения команд
# =============================

def run_cmd(args: Sequence[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    """Запускает внешнюю команду и стримит вывод в консоль. Бросает исключение при ненулевом коде выхода."""
    print(f"$ {' '.join(shlex.quote(a) for a in args)}")
    subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_cmd_capture(
    args: Sequence[str], *, cwd: Path | None = None, env: dict | None = None
) -> str:
    """Запускает внешнюю команду и возвращает stdout как строку. Бросает исключение при ненулевом коде выхода."""
    print(f"$ {' '.join(shlex.quote(a) for a in args)}")
    out = subprocess.check_output(args, cwd=str(cwd) if cwd else None, env=env)
    return out.decode("utf-8", errors="replace")


_COMPOSE_BASE_CMD: List[str] | None = None


def _detect_compose_base_cmd() -> List[str]:
    """Определяет как вызывать Compose: предпочитает docker-compose, затем docker compose.

    Причина: в некоторых окружениях плагин `docker compose` недоступен или ломается на `-f`.
    """
    # Принудительный выбор через переменную окружения, если нужно
    prefer_legacy = os.environ.get("PREFER_DOCKER_COMPOSE", "").lower() in {"1", "true", "yes"}

    docker_compose_path = shutil.which("docker-compose")
    if prefer_legacy and docker_compose_path:
        return [docker_compose_path]

    # Попытка использовать плагин `docker compose`
    try:
        out = run_cmd_capture(["docker", "compose", "version"]).strip()
        if out:
            # Проверяем, что -f работает с указанным compose файлом
            try:
                run_cmd(["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "config", "-q"])
                return ["docker", "compose"]
            except Exception:
                pass
    except Exception:
        pass

    # Фолбэк на бинарь docker-compose
    if docker_compose_path:
        return [docker_compose_path]

    # Последний шанс — вернуть docker compose как есть
    return ["docker", "compose"]


def docker_compose_cmd(*args: str) -> List[str]:
    global _COMPOSE_BASE_CMD
    if _COMPOSE_BASE_CMD is None:
        _COMPOSE_BASE_CMD = _detect_compose_base_cmd()
        print(f"Выбран Compose: {' '.join(_COMPOSE_BASE_CMD)}")
    return [
        *_COMPOSE_BASE_CMD,
        "-f",
        str(DOCKER_COMPOSE_FILE),
        *args,
    ]

def docker_inspect_health(container_name: str) -> str:
    # Если healthcheck отсутствует, используем State.Status (running/...) для обратной совместимости
    format_arg = (
        "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}"
    )
    args = [
        "docker",
        "inspect",
        "-f",
        format_arg,
        container_name,
    ]
    try:
        status = run_cmd_capture(args).strip()
    except subprocess.CalledProcessError:
        return "unknown"
    return status


def wait_for_health(container_name: str, timeout_s: int) -> None:
    """Ожидает готовности контейнера.

    Логика:
    - если у контейнера есть healthcheck — ждём статус "healthy";
    - если healthcheck отсутствует — docker inspect вернёт общий State.Status,
      тогда считаем "running" достаточным признаком готовности.
    """
    print(f"Ожидание готовности контейнера {container_name} (healthy)...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = docker_inspect_health(container_name)
        print(f"  {container_name}: {status}")
        # Теперь требуем именно healthy статус, так как у нас есть healthcheck'и
        if status == "healthy":
            return
        if status in {"exited", "dead"}:
            raise RuntimeError(f"Контейнер {container_name} остановлен (status={status}).")
        time.sleep(5)
    raise TimeoutError(f"Контейнер {container_name} не стал healthy за {timeout_s} секунд.")


def wait_for_mysql_connection(timeout_s: int = 60) -> None:
    """Дополнительная проверка подключения к MySQL."""
    print("Дополнительная проверка подключения к MySQL...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            run_cmd_capture(
                docker_compose_cmd(
                    "exec",
                    "-T",
                    MYSQL_SERVICE_NAME,
                    "mysql",
                    "-u",
                    MYSQL_ROOT_USER,
                    f"-p{MYSQL_ROOT_PASSWORD}",
                    "-e",
                    "SELECT 1;",
                ),
                cwd=WORK_DIR,
            )
            print("  MySQL: подключение успешно!")
            return
        except subprocess.CalledProcessError:
            print("  MySQL: еще не готов к подключению...")
            time.sleep(3)
    raise TimeoutError(f"MySQL не ответил на подключение за {timeout_s} секунд")


def wait_for_postgres_connection(timeout_s: int = 60) -> None:
    """Дополнительная проверка подключения к PostgreSQL."""
    print("Дополнительная проверка подключения к PostgreSQL...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            run_cmd_capture(
                [
                    "docker",
                    "exec",
                    "-i",
                    POSTGRES_CONTAINER_NAME,
                    "psql",
                    "-U",
                    POSTGRES_USER,
                    "-d",
                    POSTGRES_DB_NAME,
                    "-c",
                    "SELECT 1;",
                ]
            )
            print("  PostgreSQL: подключение успешно!")
            return
        except subprocess.CalledProcessError:
            print("  PostgreSQL: еще не готов к подключению...")
            time.sleep(3)
    raise TimeoutError(f"PostgreSQL не ответил на подключение за {timeout_s} секунд")


def compose_up() -> None:
    run_cmd(docker_compose_cmd("up", "-d", MYSQL_SERVICE_NAME, POSTGRES_SERVICE_NAME), cwd=WORK_DIR)


def compose_down() -> None:
    """Останавливает и удаляет контейнеры/сеть проекта."""
    try:
        run_cmd(docker_compose_cmd("down", "--remove-orphans"), cwd=WORK_DIR)
    except Exception as exc:
        # Если проекта ещё не было — не считаем ошибкой
        print(f"Предупреждение: не удалось корректно выполнить compose down: {exc}")


# =============================
# Работа с Docker Volumes
# =============================

def _env_compose_project_name() -> str:
    """Возвращает COMPOSE_PROJECT_NAME из окружения, если задан, иначе пустую строку."""
    return os.environ.get("COMPOSE_PROJECT_NAME", "").strip()


def _candidate_project_names() -> List[str]:
    """Формирует приоритетный список возможных имён проекта compose для поиска ресурсов.

    Порядок:
    1) COMPOSE_PROJECT_NAME из окружения (если задан)
    2) Имя директории WORK_DIR (дефолтное имя проекта compose)
    """
    names: List[str] = []
    env_name = _env_compose_project_name()
    if env_name:
        names.append(env_name)
    base = docker_project_name()
    if base not in names:
        names.append(base)
    return names


def _expected_volume_names() -> List[str]:
    """Возвращает список возможных имён томов для текущего проекта.

    Учитываются варианты с префиксом проекта и «голые» имена из docker-compose.yml.
    """
    result: list[str] = []
    for project in _candidate_project_names():
        result.append(f"{project}_percona_data")
        result.append(f"{project}_pg_data")
    # На случай, если тома были созданы вручную без префикса
    result.append("percona_data")
    result.append("pg_data")
    # Дедупликация с сохранением порядка
    seen: set[str] = set()
    unique: list[str] = []
    for name in result:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def docker_volume_exists(volume_name: str) -> bool:
    """Проверяет существование тома docker по имени."""
    try:
        subprocess.check_output(["docker", "volume", "inspect", volume_name], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def detect_existing_project_volumes() -> List[str]:
    """Ищет существующие тома проекта (percona_data, pg_data) и возвращает список найденных имён."""
    existing: list[str] = []
    for name in _expected_volume_names():
        if docker_volume_exists(name):
            existing.append(name)
    return existing


def prompt_to_remove_volumes(volumes: Sequence[str]) -> bool:
    """Показывает пользователю список найденных томов и спрашивает, удалять ли их.

    Возвращает True, если пользователь подтвердил удаление.
    """
    print("\nОбнаружены существующие тома проекта (возможные остатки предыдущего запуска):")
    for v in volumes:
        print(f"  - {v}")
    print("Предупреждение: удаление томов безвозвратно сотрёт данные в них.")
    try:
        answer = input("Удалить эти тома перед началом? [y/N]: ").strip().lower()
    except EOFError:
        answer = "n"
    return answer in {"y", "yes", "д", "да"}


def remove_docker_volumes(volumes: Sequence[str]) -> None:
    """Удаляет указанные тома docker, игнорируя ошибки для отсутствующих."""
    to_remove = [v for v in volumes if docker_volume_exists(v)]
    if not to_remove:
        return
    print("Удаление томов:")
    for v in to_remove:
        print(f"  - {v}")
    try:
        run_cmd(["docker", "volume", "rm", "-f", *to_remove])
    except Exception as exc:
        print(f"Предупреждение: не удалось удалить тома ({exc}). Продолжаю работу.")


def mysql_create_db() -> None:
    sql = (
        f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB_NAME} "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    )
    run_cmd(
        docker_compose_cmd(
            "exec",
            "-T",
            MYSQL_SERVICE_NAME,
            "mysql",
            "-u",
            MYSQL_ROOT_USER,
            f"-p{MYSQL_ROOT_PASSWORD}",
            "-e",
            sql,
        ),
        cwd=WORK_DIR,
    )


def mysql_import_dump() -> None:
    if not MYSQL_DUMP_PATH.is_file():
        raise FileNotFoundError(f"Не найден дамп MySQL: {MYSQL_DUMP_PATH}")
    print(f"Импорт дампа в Percona 5.7: {MYSQL_DUMP_PATH}")
    # Аналог: cat dump.sql | docker compose exec -T db mysql -u root -ppass mattermost
    total_size = MYSQL_DUMP_PATH.stat().st_size
    sent_bytes = 0
    last_percent = -1

    def _print_progress(percent: int) -> None:
        bar_width = 30
        filled = max(0, min(bar_width, int(bar_width * percent / 100)))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\rИмпорт MySQL: [{bar}] {percent:3d}%", end="", flush=True)

    with open(MYSQL_DUMP_PATH, "rb") as f:
        proc = subprocess.Popen(
            docker_compose_cmd(
                "exec",
                "-T",
                MYSQL_SERVICE_NAME,
                "mysql",
                "-u",
                MYSQL_ROOT_USER,
                f"-p{MYSQL_ROOT_PASSWORD}",
                MYSQL_DB_NAME,
            ),
            cwd=str(WORK_DIR),
            stdin=subprocess.PIPE,
        )
        assert proc.stdin is not None
        chunk_size = 1024 * 1024
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            proc.stdin.write(chunk)
            proc.stdin.flush()
            sent_bytes += len(chunk)
            if total_size > 0:
                percent = int(sent_bytes * 100 / total_size)
                if percent != last_percent:
                    _print_progress(min(99, percent))
                    last_percent = percent
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            print()  # перенести строку перед ошибкой
            raise RuntimeError("Импорт дампа MySQL завершился с ошибкой")
        _print_progress(100)
        print()


def run_migration_assist_mysql() -> None:
    dsn = f"{MYSQL_ROOT_USER}:{MYSQL_ROOT_PASSWORD}@tcp(localhost:3306)/{MYSQL_DB_NAME}"
    args = [
        *MIGRATION_ASSIST_CMD,
        "mysql",
        dsn,
        "--fix-unicode",
        "--fix-varchar",
        "--fix-artifacts",
    ]
    run_cmd(args, cwd=WORK_DIR)


def run_migration_assist_postgres() -> None:
    pg_url = (
        f"postgres://{POSTGRES_USER}:{POSTGRES_PASSWORD}@127.0.0.1:5432/"
        f"{POSTGRES_DB_NAME}?sslmode=disable"
    )
    args = [
        *MIGRATION_ASSIST_CMD,
        "postgres",
        pg_url,
        "--run-migrations",
        "--mattermost-version",
        MATTERMOST_VERSION,
    ]
    run_cmd(args, cwd=WORK_DIR)


def postgres_apply_sql_file(sql_path: Path) -> None:
    """Применяет один .sql файл к PostgreSQL внутри контейнера.

    Выполняется через docker exec + psql, с остановкой при первой ошибке.
    """
    if not sql_path.is_file():
        raise FileNotFoundError(f"Не найден файл миграции: {sql_path}")

    print(f"Применение миграции PostgreSQL: {sql_path.name}")
    with open(sql_path, "rb") as f:
        proc = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                POSTGRES_CONTAINER_NAME,
                "psql",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                POSTGRES_USER,
                "-d",
                POSTGRES_DB_NAME,
                "-f",
                "-",
            ],
            stdin=subprocess.PIPE,
        )
        assert proc.stdin is not None
        shutil.copyfileobj(f, proc.stdin)
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"psql завершился с ошибкой на файле {sql_path.name}")


def apply_postgres_migrations_from_dir(migrations_dir: Path) -> None:
    """Применяет все *.up.sql миграции из каталога в алфавитном порядке.

    Требуется для фолбэка, когда migration-assist postgres не может запустить миграции
    (например, бага при проверке пустоты таблиц).
    """
    if not migrations_dir.is_dir():
        raise FileNotFoundError(f"Каталог с миграциями не найден: {migrations_dir}")

    files = sorted(migrations_dir.glob("*.up.sql"))
    if not files:
        raise FileNotFoundError(
            f"В каталоге {migrations_dir} не найдено файлов *.up.sql для применения"
        )

    for sql_file in files:
        postgres_apply_sql_file(sql_file)


def run_migration_assist_pgloader() -> None:
    pg_url = (
        f"postgres://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/"
        f"{POSTGRES_DB_NAME}?sslmode=disable"
    )
    mysql_dsn = f"{MYSQL_ROOT_USER}:{MYSQL_ROOT_PASSWORD}@tcp(localhost:3306)/{MYSQL_DB_NAME}"
    args = [
        *MIGRATION_ASSIST_CMD,
        "pgloader",
        f"--postgres={pg_url}",
        f"--mysql={mysql_dsn}",
        "--output",
        str(PGLOADER_LOAD_FILE),
    ]
    run_cmd(args, cwd=WORK_DIR)


def rewrite_pgloader_load(load_path: Path) -> None:
    """Правит pgloader.load: меняет коннекты на docker-сеть и добавляет тюнинг WITH."""
    if not load_path.is_file():
        raise FileNotFoundError(f"Не найден файл {load_path} для правки")

    text = load_path.read_text(encoding="utf-8", errors="replace")

    # FROM mysql://... -> хост внутри сети
    text = re.sub(
        r"FROM\s+mysql://[^@]+@[^/\s:]+(?::\d+)?/([\w-]+)",
        f"FROM mysql://{MYSQL_ROOT_USER}:{MYSQL_ROOT_PASSWORD}@{MYSQL_HOST_IN_NETWORK}:3306/\\1",
        text,
        flags=re.IGNORECASE,
    )

    # INTO pgsql://... -> хост внутри сети
    text = re.sub(
        r"INTO\s+pgsql://[^@]+@[^/\s:]+(?::\d+)?/([\w-]+)",
        f"INTO pgsql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST_IN_NETWORK}:5432/\\1",
        text,
        flags=re.IGNORECASE,
    )

    # Добавить/заменить WITH-блок для тюнинга
    tuning_block = (
        "WITH data only, truncate,\n"
        f"    workers = {PGLOADER_WORKERS}, concurrency = {PGLOADER_CONCURRENCY},\n"
        f"    multiple readers per thread, rows per range = {PGLOADER_ROWS_PER_RANGE},\n"
        f"    prefetch rows = {PGLOADER_PREFETCH_ROWS}, batch rows = {PGLOADER_BATCH_ROWS},\n"
        "    create no tables, create no indexes,\n"
        "    preserve index names\n\n"
        "SET PostgreSQL PARAMETERS\n"
        "    maintenance_work_mem to '2GB',\n"
        "    work_mem to '256MB'\n\n"
        "SET MySQL PARAMETERS\n"
        "    net_read_timeout  = '3600',\n"
        "    net_write_timeout = '3600',\n"
        "    interactive_timeout = '28800',\n"
        "    wait_timeout = '28800'\n\n"
    )

    if re.search(r"\bWITH\b", text, flags=re.IGNORECASE):
        # Грубо заменим первый WITH-блок
        text = re.sub(
            r"WITH[\s\S]*?(?=\n\S|$)",
            tuning_block,
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        # Вставим после строки INTO ...
        text = re.sub(
            r"(INTO\s+pgsql://[^\n]+\n)",
            r"\1" + tuning_block,
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    # Защитное удаление любых вхождений max_allowed_packet из файла
    # Удаляем строку, даже если после неё есть запятая
    text = re.sub(
        r"(?im)^\s*max_allowed_packet\s*=\s*['\"]?[^'\"\n]+['\"]?\s*,?\s*$\n?",
        "",
        text,
    )

    load_path.write_text(text, encoding="utf-8")
    print(f"pgloader.load обновлён под docker-сеть: {load_path}")

def _pgloader_log_has_errors(log_path: Path) -> tuple[bool, str]:
    """Простая проверка лога pgloader на ошибки. Возвращает (has_errors, short_message).

    Фильтрует ложные срабатывания по слову "errors" в сводной таблице отчёта, проверяя
    именно уровни логирования ERROR/FATAL/CRITICAL как отдельные слова.
    """
    severity_re = re.compile(r"\b(ERROR|FATAL|CRITICAL)\b", re.IGNORECASE)
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                # Пропускаем блоки отчёта и заголовки, где встречается слово "errors"
                if (
                    line.startswith("table name")
                    or line.startswith("------------------------------------")
                    or line.startswith("Total import time")
                    or line.startswith("before load")
                    or line.startswith("after load")
                    or line.startswith("LOG report summary reset")
                ):
                    continue
                # Если строка имеет привычный формат "<ts> <LEVEL> ...", проверим LEVEL явно
                # Иначе отфильтруем только отдельные слова ERROR/FATAL/CRITICAL
                parts = line.split()
                if len(parts) >= 2 and re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*", parts[0]):
                    level = parts[1].upper()
                    if level in {"ERROR", "FATAL", "CRITICAL"}:
                        return True, line
                elif severity_re.search(line):
                    return True, line
    except FileNotFoundError:
        return True, "pgloader.log не найден"
    return False, ""


def docker_project_name() -> str:
    # По умолчанию Compose использует имя директории как project name
    return WORK_DIR.name


def compose_network_full_name() -> str:
    return f"{docker_project_name()}_{COMPOSE_NETWORK_NAME}"


def run_pgloader() -> None:
    # docker run --rm --network <project>_mattermost-migrate-network -e SBCL_DYNAMIC_SPACE_SIZE=6144 \
    #   -v "$PWD":/work -w /work dimitri/pgloader:latest pgloader --verbose ./pgloader.load > pgloader.log
    network_name = compose_network_full_name()
    # В контейнер примонтирован WORK_DIR как /work, поэтому путь к файлу
    # конфигурации pgloader должен быть контейнерным, а не хостовым.
    container_load_path = f"/work/{PGLOADER_LOAD_FILE.name}"

    args = [
        "docker",
        "run",
        "--rm",
        "--network",
        network_name,
        "-e",
        f"SBCL_DYNAMIC_SPACE_SIZE={PGLOADER_MEMORY_MB}",
        "-v",
        f"{str(WORK_DIR)}:/work",
        "-w",
        "/work",
        "dimitri/pgloader:latest",
        "pgloader",
        "--verbose",
        container_load_path,
    ]
    print(f"Запуск pgloader (лог: {PGLOADER_LOG_FILE})...")

    def _print_progress(percent: int, current_bytes: int, total_bytes: int) -> None:
        bar_width = 30
        filled = max(0, min(bar_width, int(bar_width * percent / 100)))
        bar = "#" * filled + "-" * (bar_width - filled)
        def _fmt_bytes(n: int) -> str:
            units = ["B", "KiB", "MiB", "GiB", "TiB"]
            size = float(n)
            idx = 0
            while size >= 1024.0 and idx < len(units) - 1:
                size /= 1024.0
                idx += 1
            return f"{size:.1f} {units[idx]}"
        cur_h = _fmt_bytes(current_bytes)
        tot_h = _fmt_bytes(total_bytes) if total_bytes > 0 else "?"
        print(f"\rpgloader:    [{bar}] {percent:3d}%  ({cur_h}/{tot_h})", end="", flush=True)

    def _mysql_db_size_bytes() -> int:
        try:
            out = run_cmd_capture(
                docker_compose_cmd(
                    "exec",
                    "-T",
                    MYSQL_SERVICE_NAME,
                    "mysql",
                    "-N",
                    "-u",
                    MYSQL_ROOT_USER,
                    f"-p{MYSQL_ROOT_PASSWORD}",
                    "-e",
                    f"SELECT COALESCE(SUM(data_length + index_length),0) FROM information_schema.tables WHERE table_schema='{MYSQL_DB_NAME}';",
                ),
                cwd=WORK_DIR,
            )
            return int(out.strip().splitlines()[-1] or 0)
        except Exception:
            return 0

    def _postgres_db_size_bytes() -> int:
        try:
            out = run_cmd_capture(
                [
                    "docker",
                    "exec",
                    "-i",
                    POSTGRES_CONTAINER_NAME,
                    "psql",
                    "-U",
                    POSTGRES_USER,
                    "-d",
                    POSTGRES_DB_NAME,
                    "-At",
                    "-c",
                    "select pg_database_size(current_database());",
                ]
            )
            return int(out.strip().splitlines()[-1] or 0)
        except Exception:
            return 0

    total_bytes = _mysql_db_size_bytes()
    stop_event = threading.Event()

    def _monitor() -> None:
        last_percent = -1
        while not stop_event.is_set():
            cur = _postgres_db_size_bytes()
            if total_bytes > 0:
                percent = int((cur * 100) / total_bytes)
            else:
                percent = 0
            percent = max(0, min(99, percent))
            if percent != last_percent:
                _print_progress(percent, cur, total_bytes)
                last_percent = percent
            time.sleep(2)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    with open(PGLOADER_LOG_FILE, "wb") as logf:
        proc = subprocess.Popen(args, cwd=str(WORK_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        # Тийм выход pgloader в лог, пока монитор рисует прогресс
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            if not chunk:
                break
            logf.write(chunk)
        ret = proc.wait()

    stop_event.set()
    monitor_thread.join(timeout=5)

    # Печатаем 100% только при успешном завершении
    if ret == 0:
        _print_progress(100, total_bytes if total_bytes else 0, total_bytes)
        print()
    else:
        print()

    # Дополнительная проверка по логу (pgloader часто возвращает 0 даже при ошибке)
    has_errors, short_msg = _pgloader_log_has_errors(PGLOADER_LOG_FILE)
    if ret != 0 or has_errors:
        details = f"; {short_msg}" if short_msg else ""
        raise RuntimeError(
            f"pgloader завершился с ошибкой, см. лог: {PGLOADER_LOG_FILE}{details}"
        )


def postgres_set_search_path() -> None:
    alter_cmd = [
        "docker",
        "exec",
        "-i",
        POSTGRES_CONTAINER_NAME,
        "psql",
        "-U",
        POSTGRES_USER,
        "-d",
        POSTGRES_DB_NAME,
        "-c",
        "ALTER USER mmuser SET search_path = public;",
    ]
    run_cmd(alter_cmd)


def postgres_dump_final() -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    pg_version_label = "13"
    out_name = f"mattermost-pg{pg_version_label}-{timestamp}.sql"
    out_path = POSTGRES_FINAL_DUMP_DIR / out_name

    dump_cmd = [
        "docker",
        "exec",
        "-i",
        POSTGRES_CONTAINER_NAME,
        "pg_dump",
        "-U",
        POSTGRES_USER,
        "-d",
        POSTGRES_DB_NAME,
    ]
    print(f"Снятие финального дампа PostgreSQL: {out_path}")

    # Оценочный общий объём = текущий размер базы
    def _postgres_db_size_bytes() -> int:
        try:
            out = run_cmd_capture(
                [
                    "docker",
                    "exec",
                    "-i",
                    POSTGRES_CONTAINER_NAME,
                    "psql",
                    "-U",
                    POSTGRES_USER,
                    "-d",
                    POSTGRES_DB_NAME,
                    "-At",
                    "-c",
                    "select pg_database_size(current_database());",
                ]
            )
            return int(out.strip().splitlines()[-1] or 0)
        except Exception:
            return 0

    total_estimated = _postgres_db_size_bytes()

    def _print_progress(percent: int, written: int, total: int) -> None:
        bar_width = 30
        filled = max(0, min(bar_width, int(bar_width * percent / 100)))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\rpg_dump:     [{bar}] {percent:3d}%", end="", flush=True)

    with open(out_path, "wb") as outf:
        proc = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        written = 0
        last_percent = -1
        for chunk in iter(lambda: proc.stdout.read(1024 * 64), b""):
            if not chunk:
                break
            outf.write(chunk)
            written += len(chunk)
            if total_estimated > 0:
                percent = int((written * 100) / total_estimated)
                percent = min(99, percent)
                if percent != last_percent:
                    _print_progress(percent, written, total_estimated)
                    last_percent = percent
        ret = proc.wait()
        if ret != 0:
            print()
            raise RuntimeError("pg_dump завершился с ошибкой")
        _print_progress(100, written, total_estimated)
        print()
    return out_path


def main() -> None:
    print("=== Mattermost: миграция Percona 5.7 -> PostgreSQL 13 ===")
    print(f"Рабочая директория: {WORK_DIR}")
    print(f"Docker Compose файл: {DOCKER_COMPOSE_FILE}")

    # Ранний аудит существующих томов: предложить очистку перед стартом
    existing_vols = detect_existing_project_volumes()
    confirm_cleanup = False
    if existing_vols:
        confirm_cleanup = prompt_to_remove_volumes(existing_vols)

    # Остановим окружение, чтобы освободить тома перед их удалением
    compose_down()
    if confirm_cleanup:
        remove_docker_volumes(existing_vols)

    compose_up()

    # Ожидание готовности контейнеров
    wait_for_health(PERCONA_CONTAINER_NAME, MYSQL_READY_TIMEOUT_S)
    wait_for_health(POSTGRES_CONTAINER_NAME, POSTGRES_READY_TIMEOUT_S)

    # Дополнительные проверки подключения к базам данных
    wait_for_mysql_connection()
    wait_for_postgres_connection()

    # Подготовка MySQL и импорт дампа
    mysql_create_db()
    mysql_import_dump()

    # migration-assist (подготовка MySQL и PostgreSQL)
    run_migration_assist_mysql()
    try:
        run_migration_assist_postgres()
    except Exception as exc:
        print(
            "migration-assist postgres завершился с ошибкой; переключаюсь на локальные миграции из каталога 'postgres/'..."
        )
        print(f"Причина: {exc}")
        apply_postgres_migrations_from_dir(WORK_DIR / "postgres")

    # Генерация и правка pgloader.load
    run_migration_assist_pgloader()
    rewrite_pgloader_load(PGLOADER_LOAD_FILE)

    # Запуск pgloader (перенос данных)
    run_pgloader()

    # Пост-настройка и финальный дамп
    postgres_set_search_path()
    dump_path = postgres_dump_final()

    print("\n=== Готово ===")
    print(f"Лог pgloader: {PGLOADER_LOG_FILE}")
    print(f"Финальный дамп PostgreSQL: {dump_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Операция прервана пользователем", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


