"""

All the dataformats to perform type checking.

Note: Data annotations are supported in Python 3.8.  This file would not be utilized when running on earlier versions of Python

"""


# Format for dataset statistics
class DataBinEntry(TypedDict):
 binPos: int
 binVal: int

class DataBin(TypedDict):
 minBinPos: int
 maxBinPos: int
 numBins: int
 bins: DataBinEntry


# Format for Kaggle dataset
class Kaggle_longAnswerCandidate(TypedDict):
    start_token: int
    top_level: bool
    end_token: int

class Kaggle_longAnswer(TypedDict):
    start_token: int
    candidate_index: int
    end_token: int

class Kaggle_shortAnswer(TypedDict):
    start_token: int
    end_token: int

class Kaggle_annotation(TypedDict):
    yes_no_answer: str
    long_answer: Kaggle_longAnswer
    short_answers: Kaggle_shortAnswer
    annotation_id: int

class Kaggle_dataset(TypedDict):
    document_text: str
    long_answer_candidates: List[Kaggle_longAnswerCandidate]
    question_text: str
    annotations: List[Kaggle_annotation]
    document_url: str
    example_id: int

# Structure of the Kaggle data
# Note: TypedDict is supported in python3.8
DatasetKaggle = TypedDict('DatasetKaggle', {'document_text': str, 'long_answer_candidates': List[TypedDict('long_answer_candidate', {'start_token': int, 'top_level': bool, 'end_token': int})], 'question_text': str, 'annotations': List[TypedDict('annotation', {'yes_no_answer': str, 'long_answer': TypedDict('long_answer', {'start_token': int, 'candidate_index': int, 'end_token': int}), 'short_answers': List[TypedDict('short_answer', {'start_token': int, 'end_token': int})], 'annotation_id': int})], 'document_url': str, 'example_id': int})


# Format for SQuAD v2.0 dataset
class SQuADv2_answers(TypedDict):
    test: str
    answer_start: int


class SQuADv2_qa(TypedDict):
    question: str
    id: str
    answers: list[SQuADv2_answers]
    is_impossible: bool


class SQuADv2_paragraph(TypedDict):
    qas: list
    context: str


class SQuADv2_data(TypedDict):
    title: str
    paragraphs: list[SQuADv2_paragraph]


class SQuADv2(TypedDict):
    version: str
    data: list[SQuADv2_data]