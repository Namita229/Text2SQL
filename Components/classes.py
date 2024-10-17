import numpy
import torch

# Define the Example class
class Example(object):
    def __init__(self, src_sent, tgt_grammar, tgt_code, tgt_dfa=None, idx=0, meta=None):
        """
        src_sent: Natural language query (list of tokens)
        tgt_grammar: Target grammar rules (e.g., for SQL, like SELECT_STATEMENT, etc.)
        tgt_code: The SQL code to be generated
        tgt_dfa: DFA (optional, if needed)
        idx: Example index
        meta: Additional metadata
        """
        self.src_sent = src_sent  # Natural language input query
        self.tgt_code = tgt_code  # Target SQL code sequence
        self.tgt_dfa = tgt_dfa    # DFA (if applicable for SQL constraints, optional)
        self.tgt_grammar = tgt_grammar  # Grammar actions for SQL (e.g., SELECT, WHERE, etc.)
        self.idx = idx
        self.meta = meta

# Define the GrammarExample class
class GrammarExample(object):
    def __init__(self, example, vocab, grammar, copy=True):
        """
        example: Instance of the Example class
        vocab: Vocabulary object with source and code tokens
        grammar: Grammar object with nonterminal2id mappings for SQL
        copy: Boolean indicating whether to allow copying from source sentence
        """
        self.example = example
        self.grammar = grammar
        self.action_num = len(self.example.tgt_grammar)  # Number of grammar actions
        self.code_num = len(self.example.tgt_code)  # Length of the target SQL code sequence
        self.vocab = vocab
        self.src_sent = self.example.src_sent  # Source sentence (natural language SQL query)
        
        # Convert each token in the source sentence into vocabulary indices
        self.src_sent_idx = [self.vocab.source.word2id[token] 
                if (token in self.vocab.source.word2id) else self.vocab.source.unk_id 
                for token in self.src_sent]
        self.copy = copy
        self.init_index_tensors()

    def init_index_tensors(self):
        """Initialize the tensors for grammar rule application, primitive token generation, and copying."""
        self.app_rule_idx_row = []
        self.app_rule_mask_row = []
        self.primitive_row = []
        self.primitive_gen_mask_row = []
        self.primitive_copy_mask_row = []
        self.primitive_copy_idx_mask = [[True for _ in self.src_sent] for _ in range(self.action_num)]

        e = self.example

        for t in range(self.action_num):
            app_rule_idx = token_idx = 0
            app_rule_mask = gen_token_mask = copy_mask = True

            # Get the grammar rule index
            app_rule_idx = self.grammar.nonterminal2id[e.tgt_grammar[t]]
            app_rule_mask = False

            # Get the SQL token (column, table name, etc.) to generate
            token = e.tgt_code[t][0]
            token_idx = self.vocab.code[e.tgt_code[t]]
            token_can_copy = False

            # If copying is allowed and the token exists in the input, handle copying
            if self.copy and token in self.src_sent:
                token_pos_list = [idx for idx, _token in enumerate(self.src_sent) if _token == token]
                for pos in token_pos_list:
                    self.primitive_copy_idx_mask[t][pos] = False
                copy_mask = False
                token_can_copy = True
            
            # If the token cannot be copied or is unknown, we generate it
            if token_can_copy is False or token_idx != self.vocab.code.unk_id:
                gen_token_mask = False

            self.app_rule_idx_row.append(app_rule_idx)
            self.app_rule_mask_row.append(app_rule_mask)
            self.primitive_row.append(token_idx)
            self.primitive_gen_mask_row.append(gen_token_mask)
            self.primitive_copy_mask_row.append(copy_mask)

# Define the Batch class
class Batch(object):
    def __init__(self, examples, vocab, grammar, copy=True):
        """
        examples: A list of GrammarExample instances
        vocab: The vocabulary for source sentences and code (SQL tokens)
        grammar: The SQL grammar (mapping nonterminal rules)
        copy: Boolean indicating whether to allow copying from source sentence
        """
        # Create a GrammarExample for each Example
        self.examples = [GrammarExample(example, vocab, grammar) for example in examples]
        
        # Maximum number of grammar actions and SQL code tokens across all examples
        self.max_action_num = max(e.action_num for e in self.examples)
        self.max_code_num = max(e.code_num for e in self.examples)
        self.src_sents = [e.src_sent for e in self.examples]  # Source sentences
        self.src_sents_len = [len(e.src_sent) for e in self.examples]  # Sentence lengths
        self.max_src_sents_len = max(self.src_sents_len)  # Max source sentence length
        self.copy = copy
        self.init_index_tensors()

    def __len__(self):
        return len(self.examples)

    def list_to_longtensor(self, data):
        """Convert a list of sequences into padded LongTensor."""
        tensor_list = [torch.LongTensor(seq) for seq in data]
        return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)

    def list_to_floattensor(self, data):
        """Convert a list of sequences into padded FloatTensor."""
        tensor_list = [torch.FloatTensor(seq) for seq in data]
        return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0.)

    def list_to_booltensor(self, data):
        """Convert a list of sequences into padded BoolTensor."""
        tensor_list = [torch.BoolTensor(seq) for seq in data]
        return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=True)

    def init_index_tensors(self):
        """Initialize the index tensors for source sentence tokens, grammar rules, and SQL tokens."""
        self.src_sents_idx_matrix = numpy.zeros((len(self.examples), self.max_src_sents_len), dtype='int')
        self.src_sents_mask = numpy.ones((len(self.examples), self.max_src_sents_len), dtype='bool')

        self.apply_rule_idx_matrix = []
        self.apply_rule_mask = []
        self.primitive_idx_matrix = []
        self.primitive_gen_mask = []
        self.primitive_copy_mask = []
        self.primitive_copy_idx_mask = numpy.ones((len(self.examples),self.max_action_num, self.max_src_sents_len), dtype='bool')

        for e_id, e in enumerate(self.examples):
            self.src_sents_idx_matrix[e_id,:len(e.src_sent_idx)] = e.src_sent_idx
            self.src_sents_mask[e_id,:len(e.src_sent_idx)] = False

            primitive_copy_idx_mask = numpy.array(e.primitive_copy_idx_mask)
            self.primitive_copy_idx_mask[e_id,:primitive_copy_idx_mask.shape[0],:primitive_copy_idx_mask.shape[1]] = primitive_copy_idx_mask

            self.apply_rule_idx_matrix.append(e.app_rule_idx_row)
            self.apply_rule_mask.append(e.app_rule_mask_row)
            self.primitive_idx_matrix.append(e.primitive_row)
            self.primitive_gen_mask.append(e.primitive_gen_mask_row)
            self.primitive_copy_mask.append(e.primitive_copy_mask_row)

        # Convert everything to PyTorch tensors
        self.src_sents_idx_matrix = torch.LongTensor(self.src_sents_idx_matrix)
        self.src_sents_mask = torch.BoolTensor(self.src_sents_mask)
        self.apply_rule_idx_matrix = self.list_to_longtensor(self.apply_rule_idx_matrix)
        self.apply_rule_mask = self.list_to_booltensor(self.apply_rule_mask)
        self.primitive_idx_matrix = self.list_to_longtensor(self.primitive_idx_matrix)
        self.primitive_gen_mask = self.list_to_booltensor(self.primitive_gen_mask)
        self.primitive_copy_mask = self.list_to_booltensor(self.primitive_copy_mask)
        self.primitive_copy_idx_mask = torch.from_numpy(self.primitive_copy_idx_mask)

    def pin_memory(self):
        """Prepare data for faster GPU access by pinning memory."""
        self.src_sents_idx_matrix = self.src_sents_idx_matrix.pin_memory()
        self.src_sents_mask = self.src_sents_mask.pin_memory()
        self.apply_rule_idx_matrix = self.apply_rule_idx_matrix.pin_memory()
        self.apply_rule_mask = self.apply_rule_mask.pin_memory()
        self.primitive_idx_matrix = self.primitive_idx_matrix.pin_memory()
        self.primitive_gen_mask = self.primitive_gen_mask.pin_memory()
        self.primitive_copy_mask = self.primitive_copy_mask.pin_memory()
        self.primitive_copy_idx_mask = self.primitive_copy_idx_mask.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        """Move tensors to the specified device (CPU/GPU)."""
        self.src_sents_idx_matrix = self.src_sents_idx_matrix.to(device)
        self.src_sents_mask = self.src_sents_mask.to(device)
        self.apply_rule_idx_matrix = self.apply_rule_idx_matrix.to(device)
        self.apply_rule_mask = self.apply_rule_mask.to(device)
        self.primitive_idx_matrix = self.primitive_idx_matrix.to(device)
        self.primitive_gen_mask = self.primitive_gen_mask.to(device)
        self.primitive_copy_mask = self.primitive_copy_mask.to(device)
        self.primitive_copy_idx_mask = self.primitive_copy_idx_mask.to(device)

# Example usage
example = Example(
    src_sent=["show", "all", "employees", "in", "department", "sales"],
    tgt_grammar=["SELECT_STATEMENT", "TABLE_REFERENCE", "WHERE_CLAUSE"],
    tgt_code=["SELECT *", "FROM employees", "WHERE department = 'sales'"]
)

# Suppose vocab and grammar objects exist, this would generate the batch
# vocab = ...
# grammar = ...
# batch = Batch([example], vocab, grammar)
