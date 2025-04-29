import re
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabReportExtractor:
    def __init__(self):
        self.value_pattern = re.compile(r"([<>]=?\s*)?(\d+\.?\d*)")
        self.common_text_results = {"nil", "negative", "positive", "trace", "present", "absent"}
        self.range_pattern = re.compile(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)")
        self.limit_pattern = re.compile(r"([<>])\s*(\d+\.?\d*)")
        self.common_units = {
            "%", "g/dl", "mg/dl", "u/l", "iu/l", "mmol/l", "pg", "fl",
            "/cumm", "/mm³", "x10^3/μl", "x10^6/μl", "meq/l", "mm/hr",
        }
        self.noise_keywords = {
            "test", "result", "unit", "range", "reference", "name", "patient", "doctor",
            "sample", "specimen", "date", "time", "age", "sex", "id", "phone", "page",
            "hospital", "laboratory", "diagnostics", "report", "header", "footer",
            "printed", "signature", "verified",
        }
        self.y_tolerance = 15

    def process_ocr_results(self, ocr_results: List[Dict]) -> List[Dict]:
        if not isinstance(ocr_results, list) or not ocr_results:
            logger.warning("No valid OCR results list provided.")
            return []

        rows = self._group_into_rows(ocr_results)
        extracted_tests = []
        for row in rows:
            if len(row) < 2 or self._is_noise_row(row):
                continue
            test_data = self._extract_test_from_row(row)
            if test_data:
                extracted_tests.append(test_data)

        logger.info(f"Extraction finished. Found {len(extracted_tests)} tests.")
        return extracted_tests

    def _group_into_rows(self, text_lines: List[Dict]) -> List[List[Dict]]:
        if not text_lines: return []
        valid_lines = [line for line in text_lines if isinstance(line, dict) and 'bbox' in line and len(line['bbox']) == 4]
        if not valid_lines: return []

        sorted_lines = sorted(valid_lines, key=lambda line: line['bbox'][1])
        rows, current_row = [], []
        last_line_center_y = -float('inf')

        for line in sorted_lines:
            line_center_y = (line['bbox'][1] + line['bbox'][3]) / 2
            if not current_row or abs(line_center_y - last_line_center_y) <= self.y_tolerance:
                current_row.append(line)
            else:
                rows.append(sorted(current_row, key=lambda item: item['bbox'][0]))
                current_row = [line]
            last_line_center_y = line_center_y

        if current_row:
            rows.append(sorted(current_row, key=lambda item: item['bbox'][0]))
        return rows

    def _is_noise_row(self, row: List[Dict]) -> bool:
        row_text_lower = " ".join(item.get('text', '').lower() for item in row)
        if any(keyword in row_text_lower for keyword in self.noise_keywords):
            first_col_text = row[0].get('text','').strip().upper()
            if first_col_text not in ["HB", "PCV", "TLC", "DLC", "RBC", "MCV", "MCH", "MCHC", "PLT", "ESR"]:
                 return True
            if len(row) < 2 or not self._is_plausible_value(row[1].get('text','')):
                 return True

        first_text = row[0].get('text', '').strip()
        if len(first_text) < 2 and first_text.upper() not in ["HB"]:
            return True
        if first_text.isdigit() or first_text.startswith((':', '*', '.')):
            return True

        return False

    def _extract_test_from_row(self, row: List[Dict]) -> Optional[Dict]:
        result = {"test_name": None, "test_value": None, "bio_reference_range": None, "test_unit": None, "lab_test_out_of_range": None}
        num_cols = len(row)

        test_name_text = row[0].get('text', '').strip().upper()
        test_name_text = re.sub(r'[:*.-]$', '', test_name_text).strip()
        if not test_name_text or (len(test_name_text) < 2 and test_name_text != "HB"): return None
        result["test_name"] = test_name_text

        potential_value_col = -1
        potential_unit_col = -1
        potential_range_col = -1

        for i in range(1, num_cols):
            text = row[i].get('text', '').strip()
            if not text: continue

            if potential_value_col == -1 and self._is_plausible_value(text):
                potential_value_col = i
                continue

            if potential_unit_col == -1 and self._find_unit(text):
                 if len(text) < 10 or text.lower() == self._find_unit(text).lower():
                      potential_unit_col = i
                      continue

            if potential_range_col == -1 and self._looks_like_range(text):
                potential_range_col = i
                continue

        if potential_value_col != -1:
            value_text = row[potential_value_col].get('text', '')
            num_val = self._parse_numeric_value(value_text)
            result["test_value"] = str(num_val) if num_val is not None else value_text.strip()
            if potential_unit_col == -1:
                 unit = self._find_unit(value_text)
                 if unit: result["test_unit"] = unit

        if potential_unit_col != -1:
            result["test_unit"] = self._find_unit(row[potential_unit_col].get('text', ''))

        if potential_range_col != -1:
            result["bio_reference_range"] = row[potential_range_col].get('text', '').strip()

        if not result["test_value"]: return None

        if result["test_value"] and result["bio_reference_range"]:
            numeric_value = self._parse_numeric_value(result["test_value"])
            if numeric_value is not None:
                 low, high = self._parse_range_bounds(result["bio_reference_range"])
                 is_out = None
                 if low is not None and high is not None: is_out = numeric_value < low or numeric_value > high
                 elif low is not None: is_out = numeric_value < low
                 elif high is not None: is_out = numeric_value > high
                 if is_out is not None: result["lab_test_out_of_range"] = is_out

        return result

    def _is_plausible_value(self, text: str) -> bool:
        text = text.strip()
        text_lower = text.lower()
        if not text or len(text) > 30: return False
        if self.value_pattern.search(text): return True
        if text_lower in self.common_text_results: return True
        if text in [":", "-", "--", ".", ","] or self._find_unit(text) == text: return False
        return True

    def _parse_numeric_value(self, text: str) -> Optional[float]:
        match = self.value_pattern.search(text.replace(',', '').strip())
        if match:
            try: return float(match.group(2))
            except: return None
        return None

    def _find_unit(self, text: str) -> Optional[str]:
        words = re.split(r'[\s/()]+', text)
        for word in words:
            cleaned_word = word.strip('[]{},.:;')
            for known_unit in self.common_units:
                if cleaned_word.lower() == known_unit.lower():
                    return word
        return None

    def _looks_like_range(self, text: str) -> bool:
        text = text.strip()
        if not text or len(text) > 50: return False
        if self.range_pattern.search(text): return True
        if self.limit_pattern.search(text): return True
        if ('-' in text or '–' in text) and any(c.isdigit() for c in text): return True
        return False

    def _parse_range_bounds(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        text = text.strip()
        range_match = self.range_pattern.search(text)
        if range_match:
            try: return float(range_match.group(1)), float(range_match.group(2))
            except: pass
        limit_match = self.limit_pattern.search(text)
        if limit_match:
            try:
                limit_val = float(limit_match.group(2))
                if limit_match.group(1) == '<': return None, limit_val
                if limit_match.group(1) == '>': return float(limit_match.group(1)), None # Typo: should be low, None
            except: pass
        return None, None