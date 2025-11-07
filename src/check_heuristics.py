from collections import defaultdict
from pathlib import Path
from typing import Any
from configuration import ROOT_PATH
from unidecode import unidecode
import string

from ollama_entity_extraction.data_model.ConsoleTextColor import ConsoleTextColor


def get_names() -> (list[str], list[str]):
    raw_names: list[str] = (
        Path(ROOT_PATH, f"cejil_labeled_data/ner_text_files/person_names_llama3.1.txt").read_text().split("\n")
    )
    name_labels: list[str] = Path(ROOT_PATH, f"cejil_labeled_data/labels/name_labels.txt").read_text().split("\n")
    return raw_names, name_labels


def clean_accents(name_instances: list[str], show_normalizations: bool = False) -> list[list[str]]:
    grouped_names: list[list[str]] = []

    for name in name_instances:
        if name == unidecode(name):
            continue

        found_group = False
        for group in grouped_names:
            if any(unidecode(name) == unidecode(existing_name) for existing_name in group):
                if name not in group:
                    group.append(name)
                found_group = True
                break
        if not found_group:
            grouped_names.append([name])

    if show_normalizations:
        for group in grouped_names:
            if len(group) > 1:
                normalized = unidecode(group[0])
                print(f"{normalized}: {group}")

    return grouped_names


def sort_names_by_words(name_instances: list[str], show_normalizations: bool = True) -> defaultdict[Any, set]:
    normalized_names = defaultdict(set)
    for line in name_instances:
        name_groups = [name.strip() for name in line.split(",")]
        for name in name_groups:
            normalized_name = unidecode(name)
            words = normalized_name.split()
            normalized_name = " ".join(sorted(words))
            normalized_merged_name = " ".join(sorted(name.split()))
            normalized_names[normalized_name].add(normalized_merged_name)
    if show_normalizations:
        for normalized_name, merged_names in normalized_names.items():
            if len(merged_names) == 1:
                continue
            print(f"{ConsoleTextColor.BLUE(normalized_name)}: {ConsoleTextColor.YELLOW(str(merged_names))}")
    return normalized_names


def use_part_of_the_name(
    name_instances: list[str], word_count: int = 2, show_normalizations: bool = True
) -> defaultdict[Any, set]:
    normalized_names = defaultdict(set)
    for line in name_instances:
        name_groups = [name.strip() for name in line.split(",")]
        for name in name_groups:
            normalized_name = unidecode(name)
            words = normalized_name.split()
            normalized_name = " ".join(sorted(words))
            normalized_merged_name = " ".join(sorted(name.split()))
            normalized_names[normalized_name].add(normalized_merged_name)

    normalized_names_sorted = sorted(list(normalized_names.keys()), key=lambda x: -len(x.split()))

    for i, longer_name in enumerate(normalized_names_sorted):
        longer_words = set(longer_name.split())
        if len(longer_words) < word_count:
            continue
        for shorter_name in normalized_names_sorted[i + 1 :]:
            shorter_words = set(shorter_name.split())
            if len(shorter_words) < 2:
                break
            if shorter_words.issubset(longer_words):
                normalized_names[longer_name].update(normalized_names[shorter_name])
                normalized_names[shorter_name].clear()

    normalized_names = defaultdict(set, {k: v for k, v in normalized_names.items() if v})

    if show_normalizations:
        for normalized_name, merged_names in normalized_names.items():
            if len(merged_names) == 1:
                continue
            print(f"{ConsoleTextColor.BLUE(normalized_name)}: {ConsoleTextColor.YELLOW(str(merged_names))}")
    return normalized_names


def check_mistakes(normalized_names: defaultdict[Any, set], name_labels: list[str]):

    name_label_groups = []

    for name_group_string in name_labels:
        name_group = [n.strip() for n in name_group_string.split(",")]
        name_label_groups.append(name_group)

    found_name_group_indexes = []
    correct_merges = 0
    mistakes = 0

    for normalized_name, merged_names in normalized_names.items():
        if len(merged_names) < 2:
            continue
        name_found = False
        for name_group_index, name_label_group in enumerate(name_label_groups):
            # if name_group_index in found_name_group_indexes:
            #     continue

            # if normalized_name in name_label_group:
            if normalized_name in merged_names:
                found_name_group_indexes.append(name_group_index)
                name_found = True
                correct_merges += 1
                break

            # if name_found:
            #     break

        if not name_found:
            print(f"{normalized_name}: {merged_names}")
            mistakes += 1

    print(ConsoleTextColor.GREEN(f"There are {correct_merges} correct merges."))
    print(ConsoleTextColor.PINK(f"There are {mistakes} mistakes."))


def check_mistakes_short_names(normalized_names: defaultdict[Any, set], name_labels: list[str]):
    name_label_groups = []
    for name_group_string in name_labels:
        name_group = [" ".join(sorted(unidecode(n.strip()).split())) for n in name_group_string.split(",")]
        name_label_groups.append(name_group)

    corrects = 0
    mistakes = 0

    for normalized_name, merged_names in normalized_names.items():
        if len(merged_names) <= 1:
            continue

        found_in_groups = []
        for merged_name in merged_names:
            for i, name_label_group in enumerate(name_label_groups):
                if merged_name in name_label_group:
                    found_in_groups.append(i)
                    break

        if len(set(found_in_groups)) == 1:
            corrects += 1
        else:
            mistakes += 1
            print(f"Mistake - Normalized: {normalized_name}")
            print(f"Merged names: {merged_names}")
            print(f"Found in groups: {[name_label_groups[i] for i in found_in_groups]}\n")

    total = corrects + mistakes
    print(f"\nSummary:")
    print(f"Corrects: {corrects} ({(corrects / total) * 100:.2f}%)")
    print(f"Mistakes: {mistakes} ({(mistakes / total) * 100:.2f}%)")
    print(f"Total checked: {total}")

    return corrects, mistakes


def is_abbreviated_name(name: str) -> bool:
    return any(len(word.replace(".", "")) == 1 for word in name.split())


def get_name_words_set(raw_name: str) -> set[str]:
    return set([word for word in raw_name.split() if len(word.replace(".", "")) > 1])


def get_name_abbreviations(raw_name: str) -> set[str]:
    return set([word for word in raw_name.split() if len(word.replace(".", "")) == 1])


def is_word_starts_with(word: str, starts_with: str) -> bool:
    return word.startswith(starts_with)


def fix_abbreviations(raw_names: list[str]):
    abbreviated_groups = []
    sorted_names = sorted([name for name in raw_names], key=lambda x: len(x))
    # sorted_names = [unidecode(name) for name in sorted_names]

    for i, raw_name in enumerate(sorted_names):
        if not is_abbreviated_name(raw_name):
            continue

        raw_name_words = get_name_words_set(raw_name)

        if not raw_name_words:
            continue

        raw_name_abbreviations = get_name_abbreviations(raw_name)

        abbreviated_groups.append([raw_name])
        for candidate_name in sorted_names[i + 1 :]:
            candidate_name_words = get_name_words_set(candidate_name)
            candidate_names_extra_words = candidate_name_words.difference(raw_name_words)

            if not all(
                [
                    is_word_starts_with(word, starts_with.replace(".", ""))
                    for word, starts_with in zip(candidate_names_extra_words, raw_name_abbreviations)
                ]
            ):
                continue
            if raw_name_words.intersection(candidate_name_words) == raw_name_words:
                abbreviated_groups[-1].append(candidate_name)

    for group in abbreviated_groups:
        print(f"{group}")
        print("***" * 30)

    return raw_names


def clear_punctuations(raw_names: list[str]):
    cleaned_groups = []
    for i, raw_name in enumerate(raw_names):
        clean_raw_name = "".join([letter for letter in raw_name if letter not in string.punctuation])
        if raw_name == clean_raw_name:
            continue
        cleaned_groups.append([raw_name])
        clean_raw_name_words = clean_raw_name.split()
        if not clean_raw_name_words:
            continue
        for candidate_name in raw_names[i + 1 :]:
            clean_candidate_name = "".join([letter for letter in candidate_name if letter not in string.punctuation])
            clean_candidate_name_words = clean_candidate_name.split()
            if clean_raw_name_words == clean_candidate_name_words:
                cleaned_groups[-1].append(candidate_name)

    for group in cleaned_groups:
        print(f"{group}")
        print("***" * 30)

    return cleaned_groups


def apply_heuristic():
    _, name_labels = get_names()
    raw_names = []

    for line in name_labels:
        name_groups = [name.strip() for name in line.split(",")]
        for name in name_groups:
            raw_names.append(name)

    # clean_names = clean_accents(raw_names, True)
    # clean_names = sort_names_by_words(name_labels, True)
    # clean_names = use_part_of_the_name(name_labels, word_count=3, show_normalizations=True)
    # check_mistakes(clean_names, name_labels)
    # check_mistakes_short_names(clean_names, name_labels)

    # normalized_names = defaultdict(set[str])
    # normalized_names["Ali A. Altıparmak"] = {"Al Ali A.", "A. Al Altıparmak"}
    # fixed_names = fix_abbreviations(normalized_names)
    # print(fixed_names)

    # fix_abbreviations(raw_names)
    raw_names = clean_accents(raw_names, True)
    # clear_punctuations(raw_names)
    # fix_abbreviations(["A. Ali", "Ali Ali", "Gabo G"])


if __name__ == "__main__":
    apply_heuristic()
