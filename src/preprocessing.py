import re
import os

import pandas as pd

PAUSES = ["(.)", "(..)", "(...)", "&"]


def read_chat_file_pauses_to_dict(directory_path: str, label: int, filter_by_participant: str = None) -> list[dict]:
    """
    Read all .chat files in a certain directory and return the counts.
    """
    file_paths = _list_file_paths(directory_path)

    result = []
    for file_path in file_paths:
        file_strs = _read_file(file_path)
        filtered_strs = _filter_participant_speach_strs(file_strs=file_strs, participant=filter_by_participant)
        pauses_count = _count_pauses(filtered_strs)

        pauses_count["file_path"] = os.path.basename(file_path)
        pauses_count["label"] = label
        result.append(pauses_count)

    return result


def _list_file_paths(directory_path: str) -> list[str]:
    """
    List all the files in a certain directory.
    """
    file_root = os.path.join(os.getcwd(), directory_path)
    return [os.path.join(file_root, file) for file in os.listdir(directory_path)]


def _read_file(file_path: str) -> list[str]:
    """
    Read a chat file to a list of strings.
    """
    with open(file_path, encoding="utf8") as f:
        return f.readlines()


def _filter_speach_strs(file_strs: list[str]) -> list[str]:
    """
    Return only the lines that contain speech, e.g. start with *.
    """
    return [line for line in file_strs if line.startswith("*")]


def _get_participants(speach_strs: list[str]) -> set[str]:
    """
    Get all unique particpants.
    """
    return set([re.search("(?<=\*)(.*?)(?=\:)", line).group() for line in speach_strs])


def _count_pauses(speach_strs: list[str]) -> dict[str, int]:
    """
    Count the occurences of the pauses.
    """
    result = dict.fromkeys(PAUSES, 0)
    for line in speach_strs:
        for pause in PAUSES:
            result[pause] += line.count(pause)
    return result


def _filter_participant_speach_strs(participant: str, file_strs: list[str]) -> list[str]:
    """
    Filter the lines for a single participant, e.g. PAR or INV.
    """
    speach_strs = _filter_speach_strs(file_strs)

    if participant:
        if participant not in _get_participants(speach_strs):
            raise Exception(f"Participant '{participant}' is not in .chat file")

    return [line for line in speach_strs if line.startswith(f"*{participant}")]


def chat_count_dict_to_df(count_dict: list[dict]) -> pd.DataFrame:
    """
    Convert dict of counts to pd.DataFrame.
    """
    return pd.DataFrame.from_dict(count_dict)


if __name__ == "__main__":
    # Dementia
    directory_path = "./chat_files/dementia/cookie"
    chat_file_pauses_dict = read_chat_file_pauses_to_dict(directory_path, filter_by_participant="PAR", label=1)

    df_dementia = chat_count_dict_to_df(chat_file_pauses_dict)

    # Control
    directory_path = "./chat_files/control/cookie"
    chat_file_pauses_dict = read_chat_file_pauses_to_dict(directory_path, filter_by_participant="PAR", label=0)

    df_control = chat_count_dict_to_df(chat_file_pauses_dict)

    # Concat dataframes
    df = pd.concat([df_dementia, df_control]).sort_values("file_path").reset_index(drop=True)
