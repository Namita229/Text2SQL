import nltk
import sqlparse
import pickle
import json  # Added for Spider dataset loading
from collections import Counter
#nltk.download('punkt')

# Constants (same as your version)
NLNUM = 3
TRAINNUM = 10000
MAXTOKENLEN = 75
MAXLEN = 512
GENLEN = 256

def canonicalize_sql(sql_query):
    """
    Canonicalize the SQL code by replacing table names, column names, and literals with placeholders.
    """
    tokens = sqlparse.parse(sql_query)[0].tokens
    token_list = []

    column_map = {}
    table_map = {}
    val_map = {}

    column_counter = 0
    table_counter = 0
    val_counter = 0

    for token in tokens:
        # Identify table names
        if token.ttype is None and token.value.upper() not in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'ORDER BY']:
            if token.value not in table_map:
                table_map[token.value] = f'table_{table_counter}'
                table_counter += 1
            token_list.append(table_map[token.value])
        # Identify column names
        elif token.ttype is None and token.value not in table_map:
            if token.value not in column_map:
                column_map[token.value] = f'column_{column_counter}'
                column_counter += 1
            token_list.append(column_map[token.value])
        # Identify literals (numbers or strings)
        elif token.ttype == sqlparse.tokens.Literal.String.Single or token.ttype == sqlparse.tokens.Literal.Number.Integer:
            if token.value not in val_map:
                val_map[token.value] = f'val_{val_counter}'
                val_counter += 1
            token_list.append(val_map[token.value])
        else:
            token_list.append(token.value.upper())

    canonicalized_sql = ' '.join(token_list)
    return canonicalized_sql, {'columns': column_map, 'tables': table_map, 'values': val_map}

def tokenize_sql(sql_query):
    """
    Tokenize SQL query into its component tokens (keywords, identifiers, literals).
    """
    return sqlparse.format(sql_query, reindent=False, keyword_case='upper').split()

def preprocess_sql_dataset(input_path, output_path, vocab_src_path, vocab_code_path, src_freq=2, code_freq=2):
    """
    Preprocess the SQL dataset from the Spider dataset.
    Canonicalize and tokenize the SQL code and its natural language description.
    """
    dataset = []
    src_vocab_counter = Counter()
    code_vocab_counter = Counter()

    # Load Spider dataset (JSON format)
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Process each example in the dataset
    for example in data:
        # Extract natural language description (intent)
        nl_list = [example['question']]
        code_list = [example['query']]  # Direct SQL query from Spider dataset

        # Canonicalize SQL code
        canonicalized_code, maps = canonicalize_sql(example['query'])  # Changed to handle 'query'

        # Tokenize NL intent
        tokenized_nl = nltk.word_tokenize(example['question'].lower())

        # Tokenize SQL code
        tokenized_code = tokenize_sql(canonicalized_code)

        # Update vocab counters
        src_vocab_counter.update(tokenized_nl)
        code_vocab_counter.update(tokenized_code)

        # Append the processed entry to the dataset
        dataset.append({
            'intent': tokenized_nl,
            'code': tokenized_code,
            'maps': maps
        })

    # Save processed dataset
    with open(output_path, 'wb') as file:
        pickle.dump(dataset, file)

    # Create and save vocabularies
    src_vocab = {token for token, count in src_vocab_counter.items() if count >= src_freq}
    code_vocab = {token for token, count in code_vocab_counter.items() if count >= code_freq}

    with open(vocab_src_path, 'wb') as file:
        pickle.dump(src_vocab, file)

    with open(vocab_code_path, 'wb') as file:
        pickle.dump(code_vocab, file)

def preprocess_dataset(data_dir, split, vocab_dir):
    """
    Wrapper function for preprocessing train, dev, and test splits of SQL dataset.
    """
    input_file = "C:/Users/vaish/OneDrive/Desktop/ML-SQL/dataset/spider_data/train_spider.json"
    output_file = f"{data_dir}/{split}_processed.bin"
    vocab_src_file = f"{vocab_dir}/vocab_src.bin"
    vocab_code_file = f"{vocab_dir}/vocab_code.bin"

    preprocess_sql_dataset(input_file, output_file, vocab_src_file, vocab_code_file)

# Example usage
data_directory = "C:/Users/vaish/OneDrive/Desktop/ML-SQL/dataset/spider_data"  # Directory where you downloaded the Spider dataset
vocab_directory = "C:/Users/vaish/OneDrive/Desktop/ML-SQL/vocab"  # Directory where vocab will be saved

preprocess_dataset(data_directory, "train_spider", vocab_directory)
preprocess_dataset(data_directory, "dev", vocab_directory)
preprocess_dataset(data_directory, "test", vocab_directory)
