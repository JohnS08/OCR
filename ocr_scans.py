#!/usr/bin/env python3
"""OCR сканированных PDF с сохранением исходной графики страниц.

Программа не перерисовывает исходные страницы: она добавляет к ним невидимый
текстовый слой. Поэтому изображения, формулы, таблицы и геометрия документа
визуально остаются такими же, как в исходном скане.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pymupdf as fitz
import pytesseract
from PIL import Image
from pytesseract import Output


DEFAULT_SOURCE = Path(r"C:\СКАН")
DEFAULT_OUTPUT_NAME = "РЕЗУЛЬТАТ"
DEFAULT_DPI = 300
DEFAULT_CONFIDENCE = 75.0
DEFAULT_PSM = 3
FONT_NAME = "OCRTextLayer"

# Символы, характерные для математических выражений. Дефис отдельно не
# является достаточным признаком формулы: он часто встречается в обычном тексте.
MATH_SYMBOLS = set("=+*/×÷≈≠≤≥±∑∫√∞∂∆^_{}[]{}|\\~")
STRONG_MATH_SYMBOLS = set("=+*/×÷≈≠≤≥±∑∫√∞∂∆^_{}|\\")
GREEK_OR_MATH = re.compile(r"[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]")
WORD_CHARS = re.compile(r"[A-Za-zА-Яа-яЁё]")
VARIABLE_TOKEN = re.compile(r"^[A-Za-zА-Яа-яЁё](?:\d+)?$")


@dataclass(frozen=True)
class OcrWord:
    """Одно слово из TSV-вывода Tesseract."""

    index: int
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block: int
    paragraph: int
    line: int

    @property
    def line_key(self) -> tuple[int, int, int]:
        return self.block, self.paragraph, self.line


@dataclass
class PageResult:
    accepted_words: int = 0
    recovered_table_words: int = 0
    uncertain_words: int = 0
    formula_lines: int = 0


class UserFacingError(RuntimeError):
    """Ошибка конфигурации, которую следует показать пользователю без трассировки."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Добавляет невидимый русско-английский OCR-слой к PDF, не меняя "
            "визуальное содержимое сканов."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Корневая папка с исходными PDF. Вложенные папки обрабатываются.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Папка результатов; по умолчанию <source>\\РЕЗУЛЬТАТ.",
    )
    parser.add_argument(
        "--languages",
        default="rus+eng",
        help="Языки Tesseract через знак +.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Разрешение рендеринга для OCR.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Минимальная уверенность Tesseract от 0 до 100.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=DEFAULT_PSM,
        help="Режим сегментации страницы Tesseract.",
    )
    parser.add_argument(
        "--tesseract",
        type=Path,
        default=None,
        help="Полный путь к tesseract.exe, если он отсутствует в PATH.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Путь к TrueType-шрифту с русскими и английскими символами.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Максимальное число секунд OCR на одну страницу.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать PDF, которые уже есть в папке результатов.",
    )
    parser.add_argument(
        "--reset-log",
        action="store_true",
        help="Очистить ocr.log перед текущим запуском.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Проверить Tesseract, языки и шрифт, затем завершить работу.",
    )
    return parser.parse_args()


def configure_logging(log_path: Path, reset_log: bool) -> logging.Logger:
    """Настраивает журнал обработки в UTF-8."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scan_ocr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    mode = "w" if reset_log else "a"
    handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def find_font(requested_font: Path | None) -> Path:
    """Находит TTF-шрифт, который способен отобразить кириллицу в текстовом слое."""
    candidates: list[Path] = []
    if requested_font is not None:
        candidates.append(requested_font)

    windows_font_dir = Path(r"C:\Windows\Fonts")
    candidates.extend(
        [
            windows_font_dir / "arial.ttf",
            windows_font_dir / "calibri.ttf",
            windows_font_dir / "times.ttf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise UserFacingError(
        "Не найден TrueType-шрифт с кириллицей. Укажите его параметром --font, "
        "например --font C:\\Windows\\Fonts\\arial.ttf."
    )


def configure_tesseract(executable: Path | None) -> None:
    """Задаёт путь к движку, если он передан явно."""
    if executable is not None:
        if not executable.is_file():
            raise UserFacingError(f"Не найден файл Tesseract: {executable}")
        pytesseract.pytesseract.tesseract_cmd = str(executable)


def validate_environment(languages: str, font_path: Path) -> str:
    """Проверяет доступность движка и нужных языковых данных."""
    try:
        version = str(pytesseract.get_tesseract_version()).splitlines()[0]
        installed_languages = set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise UserFacingError(
            "Tesseract не найден. Установите 64-битный Tesseract OCR и добавьте "
            "его папку в PATH, либо передайте путь через --tesseract."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise UserFacingError(f"Не удалось запустить Tesseract: {exc}") from exc

    wanted = {item.strip() for item in languages.split("+") if item.strip()}
    missing = sorted(wanted - installed_languages)
    if missing:
        raise UserFacingError(
            "В Tesseract отсутствуют языковые данные: "
            + ", ".join(missing)
            + ". Установите соответствующие .traineddata-файлы в tessdata."
        )

    if not font_path.is_file():
        raise UserFacingError(f"Не найден шрифт: {font_path}")
    return version


def iter_input_pdfs(source_dir: Path, output_dir: Path) -> Iterable[Path]:
    """Рекурсивно отдаёт PDF, исключая папку с уже готовыми результатами."""
    output_resolved = output_dir.resolve()
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        try:
            path.resolve().relative_to(output_resolved)
        except ValueError:
            yield path


def destination_for(input_pdf: Path, source_dir: Path, output_dir: Path) -> Path:
    """Сохраняет структуру вложенных папок без изменения имени PDF."""
    try:
        relative = input_pdf.relative_to(source_dir)
    except ValueError as exc:
        raise UserFacingError(f"Файл вне исходной папки: {input_pdf}") from exc
    return output_dir / relative


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    """Рендерит страницу в RGB без альфа-канала для передачи Tesseract."""
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    try:
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        pixmap = None


def remove_table_lines(image: Image.Image) -> Image.Image:
    """Создаёт временную копию страницы без длинных линий сетки.

    Эта копия используется только для второго OCR-прохода. Исходный PDF и его
    отображение не изменяются. Морфология OpenCV помогает Tesseract прочитать
    текст внутри ячеек, который первый проход нередко пропускает из-за рамок.
    """
    array = np.array(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(40, image.width // 50), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(40, image.height // 50))
    )
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    cleaned = cv2.inpaint(array, line_mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def recognize_words(image: Image.Image, languages: str, psm: int, timeout: int) -> list[OcrWord]:
    """Возвращает слова Tesseract с координатами и степенью уверенности."""
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        image,
        lang=languages,
        config=config,
        output_type=Output.DICT,
        timeout=timeout,
    )

    words: list[OcrWord] = []
    for index, raw_text in enumerate(data["text"]):
        text = " ".join(str(raw_text).split())
        if not text:
            continue
        width = int(data["width"][index])
        height = int(data["height"][index])
        if width <= 0 or height <= 0:
            continue
        words.append(
            OcrWord(
                index=index,
                text=text,
                confidence=safe_float(data["conf"][index]),
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=width,
                height=height,
                block=int(data["block_num"][index]),
                paragraph=int(data["par_num"][index]),
                line=int(data["line_num"][index]),
            )
        )
    return words


def words_overlap(first: OcrWord, second: OcrWord) -> bool:
    """Определяет, описывают ли два OCR-слова один и тот же участок страницы."""
    first_x1 = first.left + first.width
    first_y1 = first.top + first.height
    second_x1 = second.left + second.width
    second_y1 = second.top + second.height
    intersection_width = max(0, min(first_x1, second_x1) - max(first.left, second.left))
    intersection_height = max(0, min(first_y1, second_y1) - max(first.top, second.top))
    intersection = intersection_width * intersection_height
    if not intersection:
        return False
    union = first.width * first.height + second.width * second.height - intersection
    return union > 0 and intersection / union >= 0.45


def merge_ocr_passes(primary: list[OcrWord], table_pass: list[OcrWord]) -> tuple[list[OcrWord], int]:
    """Добавляет только новые слова из прохода без линий таблиц.

    Основной проход имеет приоритет. Благодаря этому не возникают дубли текста
    и второй, более агрессивный проход не заменяет результаты на рисунках.
    """
    merged = list(primary)
    recovered = 0
    for candidate in table_pass:
        if any(words_overlap(candidate, existing) for existing in merged):
            continue
        merged.append(candidate)
        recovered += 1
    return merged, recovered


def line_text(words: list[OcrWord]) -> str:
    return " ".join(word.text for word in sorted(words, key=lambda item: item.left))


def is_formula_line(words: list[OcrWord]) -> bool:
    """Консервативно определяет строку, похожую на математическую формулу.

    Алгоритм намеренно предпочитает оставить спорную строку графикой вместо
    добавления неверного текста. Обычные дефисы и номера страниц сами по себе
    формулой не считаются.
    """
    text = line_text(words)
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False

    strong_symbols = sum(char in STRONG_MATH_SYMBOLS for char in compact)
    all_symbols = sum(char in MATH_SYMBOLS for char in compact)
    greek_symbols = len(GREEK_OR_MATH.findall(compact))
    plain_tokens = [re.sub(r"[^A-Za-zА-Яа-яЁё0-9]", "", word.text) for word in words]
    variable_tokens = sum(bool(VARIABLE_TOKEN.fullmatch(token)) for token in plain_tokens if token)
    short_line = len(words) <= 6

    if strong_symbols >= 2:
        return True
    if "=" in compact and (variable_tokens >= 2 or all_symbols >= 2):
        return True
    if greek_symbols >= 1 and (strong_symbols >= 1 or variable_tokens >= 2):
        return True
    if short_line and strong_symbols >= 1 and variable_tokens >= 2:
        return True
    if short_line and all_symbols >= 3:
        return True
    return False


def bounding_box(words: list[OcrWord]) -> tuple[int, int, int, int]:
    x0 = min(word.left for word in words)
    y0 = min(word.top for word in words)
    x1 = max(word.left + word.width for word in words)
    y1 = max(word.top + word.height for word in words)
    return x0, y0, x1, y1


def describe_box(box: tuple[int, int, int, int]) -> str:
    return f"x={box[0]}, y={box[1]}, w={box[2] - box[0]}, h={box[3] - box[1]}"


def preview(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def split_words_by_status(
    words: list[OcrWord],
    confidence_threshold: float,
    source_name: str,
    page_number: int,
    logger: logging.Logger,
) -> tuple[list[OcrWord], PageResult]:
    """Выбирает слова, которые безопасно добавить в текстовый слой."""
    grouped: dict[tuple[int, int, int], list[OcrWord]] = defaultdict(list)
    for word in words:
        grouped[word.line_key].append(word)

    accepted: list[OcrWord] = []
    result = PageResult()
    for line_words in grouped.values():
        ordered_words = sorted(line_words, key=lambda item: item.left)
        text = line_text(ordered_words)
        if is_formula_line(ordered_words):
            result.formula_lines += 1
            logger.info(
                "ФОРМУЛА | файл=%s | стр=%d | %s | текст=%r",
                source_name,
                page_number,
                describe_box(bounding_box(ordered_words)),
                preview(text),
            )
            continue

        for word in ordered_words:
            if word.confidence < confidence_threshold:
                result.uncertain_words += 1
                logger.info(
                    "НЕУВЕРЕННО | файл=%s | стр=%d | %s | confidence=%.1f | текст=%r",
                    source_name,
                    page_number,
                    describe_box((word.left, word.top, word.left + word.width, word.top + word.height)),
                    word.confidence,
                    preview(word.text, 60),
                )
                continue
            accepted.append(word)
            result.accepted_words += 1

    return accepted, result


def add_invisible_text(
    page: fitz.Page,
    words: list[OcrWord],
    image_size: tuple[int, int],
    font_path: Path,
) -> int:
    """Добавляет слова в режиме PDF Tr=3: текст есть для поиска, но не виден."""
    if not words:
        return 0

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise UserFacingError("Получено изображение страницы с недопустимым размером.")

    page.insert_font(fontname=FONT_NAME, fontfile=str(font_path))
    scale_x = page.rect.width / image_width
    scale_y = page.rect.height / image_height
    inserted = 0

    for word in words:
        x0 = page.rect.x0 + word.left * scale_x
        y0 = page.rect.y0 + word.top * scale_y
        word_height = word.height * scale_y
        font_size = max(3.0, min(96.0, word_height * 0.95))
        baseline = min(page.rect.y1 - 0.1, y0 + font_size * 0.82)
        try:
            page.insert_text(
                (x0, baseline),
                word.text,
                fontsize=font_size,
                fontname=FONT_NAME,
                render_mode=3,
                overlay=True,
            )
            inserted += 1
        except (RuntimeError, ValueError) as exc:
            # Не прерываем документ из-за одного проблемного глифа: исходный
            # фрагмент остаётся графикой, а факт записывается в общий журнал.
            logging.getLogger("scan_ocr").warning(
                "Не удалось добавить слово в текстовый слой: %r (%s)", word.text, exc
            )
    return inserted


def process_pdf(
    input_pdf: Path,
    output_pdf: Path,
    args: argparse.Namespace,
    font_path: Path,
    logger: logging.Logger,
) -> tuple[int, PageResult]:
    """Обрабатывает один документ и атомарно записывает готовый PDF."""
    document = fitz.open(input_pdf)
    total = PageResult()
    temp_path = output_pdf.with_name(output_pdf.stem + ".partial.pdf")
    try:
        if document.needs_pass:
            raise UserFacingError("PDF защищён паролем и не может быть обработан.")

        for page_index, page in enumerate(document):
            image = render_page(page, args.dpi)
            table_image: Image.Image | None = None
            try:
                primary_words = recognize_words(image, args.languages, args.psm, args.timeout)
                table_image = remove_table_lines(image)
                # PSM 6 рассматривает очищенную область как единый текстовый блок;
                # это делает текст внутри ячеек доступным даже при плотной сетке.
                table_words = recognize_words(table_image, args.languages, 6, args.timeout)
                words, recovered = merge_ocr_passes(primary_words, table_words)
                if recovered:
                    logger.info(
                        "ТАБЛИЦА | файл=%s | стр=%d | добавлено_слов_после_удаления_линий=%d",
                        input_pdf.name,
                        page_index + 1,
                        recovered,
                    )
                accepted, page_result = split_words_by_status(
                    words,
                    args.confidence,
                    input_pdf.name,
                    page_index + 1,
                    logger,
                )
                inserted = add_invisible_text(page, accepted, image.size, font_path)
                total.accepted_words += inserted
                total.recovered_table_words += recovered
                total.uncertain_words += page_result.uncertain_words
                total.formula_lines += page_result.formula_lines
            finally:
                if table_image is not None:
                    table_image.close()
                image.close()

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        if temp_path.exists():
            temp_path.unlink()
        # garbage=0 и отсутствие оптимизации намеренно минимизируют изменения
        # в исходных потоках PDF; меняется лишь добавленный текстовый слой.
        document.save(temp_path, garbage=0, deflate=False)
        if output_pdf.exists():
            output_pdf.unlink()
        temp_path.replace(output_pdf)
        return len(document), total
    finally:
        document.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if not 72 <= args.dpi <= 600:
        raise UserFacingError("Параметр --dpi должен быть в диапазоне от 72 до 600.")
    if not 0 <= args.confidence <= 100:
        raise UserFacingError("Параметр --confidence должен быть в диапазоне от 0 до 100.")
    if args.timeout < 10:
        raise UserFacingError("Параметр --timeout должен быть не меньше 10 секунд.")

    source_dir = args.source.expanduser()
    output_dir = args.output.expanduser() if args.output else source_dir / DEFAULT_OUTPUT_NAME
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    if not source_dir.is_dir():
        raise UserFacingError(f"Исходная папка не найдена: {source_dir}")
    if source_dir == output_dir:
        raise UserFacingError("Папка результатов должна отличаться от исходной папки.")

    configure_tesseract(args.tesseract)
    font_path = find_font(args.font)
    version = validate_environment(args.languages, font_path)

    log_path = output_dir / "ocr.log"
    logger = configure_logging(log_path, args.reset_log)
    logger.info(
        "СТАРТ | source=%s | output=%s | languages=%s | dpi=%d | confidence=%.1f | tesseract=%s",
        source_dir,
        output_dir,
        args.languages,
        args.dpi,
        args.confidence,
        version,
    )

    if args.check:
        print("Проверка пройдена.")
        print(f"Tesseract: {version}")
        print(f"Шрифт: {font_path}")
        print(f"Языки: {args.languages}")
        print(f"Журнал: {log_path}")
        logger.info("ПРОВЕРКА | успешно завершена")
        return 0

    pdfs = sorted(iter_input_pdfs(source_dir, output_dir), key=lambda path: str(path).lower())
    if not pdfs:
        print(f"В папке {source_dir} не найдено PDF-файлов.")
        logger.warning("ЗАВЕРШЕНО | исходные PDF не найдены")
        return 0

    print(f"Найдено PDF: {len(pdfs)}. Результаты: {output_dir}")
    succeeded = skipped = failed = 0
    all_pages = all_text = all_table_recovered = all_uncertain = all_formulas = 0

    for number, input_pdf in enumerate(pdfs, start=1):
        output_pdf = destination_for(input_pdf, source_dir, output_dir)
        if output_pdf.exists() and not args.overwrite:
            skipped += 1
            print(f"[{number}/{len(pdfs)}] Пропуск: {input_pdf.name} (результат уже существует)")
            logger.info("ПРОПУСК | файл=%s | причина=результат уже существует", input_pdf)
            continue

        print(f"[{number}/{len(pdfs)}] OCR: {input_pdf.name}")
        started = time.perf_counter()
        try:
            pages, result = process_pdf(input_pdf, output_pdf, args, font_path, logger)
            elapsed = time.perf_counter() - started
            succeeded += 1
            all_pages += pages
            all_text += result.accepted_words
            all_table_recovered += result.recovered_table_words
            all_uncertain += result.uncertain_words
            all_formulas += result.formula_lines
            logger.info(
                "ГОТОВО | файл=%s | страниц=%d | слов_слой=%d | таблица_добавлено=%d | неуверенных=%d | формул=%d | секунд=%.1f",
                input_pdf,
                pages,
                result.accepted_words,
                result.recovered_table_words,
                result.uncertain_words,
                result.formula_lines,
                elapsed,
            )
            print(
                f"    готово: {pages} стр., текстовых слов {result.accepted_words}, "
                f"добавлено из таблиц {result.recovered_table_words}, "
                f"неуверенных {result.uncertain_words}, формул {result.formula_lines}"
            )
        except Exception as exc:  # Обрабатываем следующий PDF даже при ошибке одного файла.
            failed += 1
            logger.exception("ОШИБКА | файл=%s | %s", input_pdf, exc)
            print(f"    ошибка: {exc}", file=sys.stderr)

    logger.info(
        "ИТОГ | успешно=%d | пропущено=%d | ошибок=%d | страниц=%d | слов_слой=%d | таблица_добавлено=%d | неуверенных=%d | формул=%d",
        succeeded,
        skipped,
        failed,
        all_pages,
        all_text,
        all_table_recovered,
        all_uncertain,
        all_formulas,
    )
    print(
        "Завершено. "
        f"Успешно: {succeeded}; пропущено: {skipped}; ошибок: {failed}; "
        f"страниц: {all_pages}. Журнал: {log_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Операция прервана пользователем.", file=sys.stderr)
        raise SystemExit(130)
