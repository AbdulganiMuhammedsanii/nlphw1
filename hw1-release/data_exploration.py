# Name(s): Edwin Dake, Abdulgani Muhammedsani
# Netid(s): ed433, amm546

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import json
import zipfile
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

## ================ Helper functions for loading data ==========================


def unzip_file(zip_filepath, dest_path):
    """
    Returns boolean indication of whether the file was successfully unzipped.

    Input:
      zip_filepath: String, path to the zip file to be unzipped
      dest_path: String, path to the directory to unzip the file to
    Output:
      result: Boolean, True if file was successfully unzipped, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        return True
    except Exception as e:
        return False


def unzip_data(zipTarget, destPath):
    """
    Unzips a directory, and places the contents in the original zipped
    folder into a folder at destPath. Overwrites contents of destPath if it
    already exists.

    Input:
            None
    Output:
            None

    E.g. if zipTarget = "../dataset/student_dataset.zip" and destPath = "data"
          then the contents of the zip file will be unzipped into a directory
          called "data" in the cwd.
    """
    # First, remove the destPath directory if it exists
    if os.path.exists(destPath):
        shutil.rmtree(destPath)

    unzip_file(zipTarget, destPath)

    # Get the name of the subdirectory
    sub_dir_name = os.path.splitext(os.path.basename(zipTarget))[0]
    sub_dir_path = os.path.join(destPath, sub_dir_name)

    # Move all files from the subdirectory to the parent directory
    for filename in os.listdir(sub_dir_path):
        shutil.move(os.path.join(sub_dir_path, filename), destPath)

    # Remove the subdirectory
    os.rmdir(sub_dir_path)


def read_json(filepath):
    """
    Reads a JSON file and returns the contents of the file as a dictionary.

    Input:
      filepath: String, path to the JSON file to be read
    Output:
      result: Dict, representing the contents of the JSON file
    """
    with open(filepath, "r") as f:
        return json.load(f)


def load_dataset(data_zip_path, dest_path):
    """
    Returns the training, validation, and test data as dictionaries.

    Input:
      data_zip_path: String, representing the path to the zip file containing the
      data
      dest_path: String, representing the path to the directory to unzip the data
      to
    Output:
      training_data: Dict, representing the training data
      validation_data: Dict, representing the validation data
      test_data: Dict, representing the test data
    """
    unzip_data(data_zip_path, dest_path)
    training_data = read_json(os.path.join(dest_path, "train.json"))
    validation_data = read_json(os.path.join(dest_path, "val.json"))
    test_data = read_json(os.path.join(dest_path, "test.json"))
    return training_data, validation_data, test_data


## =============================================================================

################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your
# implementation for any function with changed specs will most likely fail!
################################################################################

## ================== Functions for students to implement ======================


def stringify_labeled_doc(text, ner):
    """
    Returns a string representation of a tagged sentence from the dataset.

    Input:
      text: List[String], A document represented as a list of tokens, where each
      token is a string
      ner: List[String], A list of NER tags, where each tag corresponds to the
      token at the same index in `text`
    Output:
      result: String, representing the example in a readable format. Named entites
      are combined with their corresponding tokens, and surrounded by square
      brackets. Sequential named entity tags that are part of the same named
      entity should be combined into a single named entity. The format for named
      entities should be [TAG token1 token2 ... tokenN] where TAG is the tag for
      the named entity, and token1 ... tokenN are the tokens that make up the
      named entity. Note that tokens which are part of the same named entity
      should be separated by a single space. BIO prefix are stripped from the
      tags. O tags are ignored.

      E.g.
      ["Gavin", "Fogel", "is", "cool", "."]
      ["B-PER", "I-PER", "O", "O", "O"]

      returns "[PER Gavin Fogel] is cool."
    """
    # TODO: YOUR CODE HERE
    result = []
    current_entity = []
    current_tag = None

    for token, tag in zip(text, ner):
        if tag == "O":  # Non-entity token
            if current_entity:
                # Close the previous entity
                result.append(f"[{current_tag} {' '.join(current_entity)}]")
                current_entity = []  # Reset the entity tracker
                current_tag = None  # Reset entity type

            result.append(token)  # Add the regular token

        else:  # Handling named entities
            entity_prefix, entity_type = tag.split("-") if "-" in tag else ("", tag)

            if entity_prefix == "B":  # New entity starts
                if current_entity:
                    # Close the previous entity before starting a new one
                    result.append(f"[{current_tag} {' '.join(current_entity)}]")
                    current_entity = []

                # Start a new entity
                current_entity.append(token)
                current_tag = entity_type  # Update entity type

            elif entity_prefix == "I" and entity_type == current_tag:
                # Continue the existing entity
                current_entity.append(token)

            else:
                # Edge case: "I-" without "B-" should be treated as a mistake; start a new entity
                if current_entity:
                    result.append(f"[{current_tag} {' '.join(current_entity)}]")

                # Start a new entity
                current_entity = [token]
                current_tag = entity_type

    # If there is any leftover entity at the end, append it
    if current_entity:
        result.append(f"[{current_tag} {' '.join(current_entity)}]")

    # Join the result into a single string and return it
    return " ".join(result)


text = ["Gavin", "Fogel", "is", "cool", "."]
ner = ["B-PER", "I-PER", "O", "O", "O"]

# Example usage:
text2 = [
    "ZIFA",
    "said",
    "Renate",
    "Goetschl",
    "of",
    "Austria",
    "won",
    "the",
    "women's",
    "World",
    "Cup",
    "downhill",
    "race",
    "in",
    "Germany",
]
ner2 = [
    "B-ORG",
    "O",
    "B-PER",
    "I-PER",
    "O",
    "B-LOC",
    "O",
    "O",
    "O",
    "B-MISC",
    "I-MISC",
    "O",
    "O",
    "O",
    "B-LOC",
]

# print(stringify_labeled_doc(text, ner))
# print(stringify_labeled_doc(text2, ner2))


def validate_ner_sequence(ner):
    """
    Returns True if the named entity list is valid, False otherwise.

    Input:
      ner: List[String], representing a list of tags
    Output:
      result: Boolean, True if the named entity list is valid sequence, False otherwise
    """
    # possibly b followed by i but tags dont match return false
    # if i exists and b is not prior also false

    valid_prefix = {"B", "I"}
    for i, tag in enumerate(ner):
        if tag == "O":
            continue  # 'O' tokens are allowed
        if "-" not in tag:
            return False

        prefix, tag_type = tag.split("-", 1)

        if prefix == "I":
            if i == 0:
                return False
            prev_tag = ner[i - 1]
            if prev_tag == "O":
                return False

            # make sure previous matches
            if "-" not in tag:
                return False
            prev_prefix, prev_entity_type = prev_tag.split("-", 1)
            if prev_entity_type != tag_type:
                return False

            if prev_prefix not in valid_prefix:
                return False
    return True


ner3 = ["B-PER", "I-PER", "O", "O", "O"]

# Example usage:

ner4 = [
    "B-ORG",
    "O",
    "B-PER",
    "I-PER",
    "O",
    "B-LOC",
    "O",
    "O",
    "O",
    "B-MISC",
    "I-MISC",
    "O",
    "O",
    "I-LOC",
]

ner5 = ["B-PER", "I-LOC", "O", "O", "O"]

# print(True, validate_ner_sequence(ner3))
# print(False, validate_ner_sequence(ner4))
# print(False, validate_ner_sequence(ner5))
