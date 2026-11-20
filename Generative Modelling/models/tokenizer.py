import re
import json
import os
from typing import List, Dict, Union

# Comprehensive vocabulary covering Human Faces (CelebA), Canine (Dogs), and General Scenes
CORE_VOCAB = [
    "<pad>", "<start>", "<end>", "<unk>",
    # Visual Domain: Human Faces, Portraits & Attributes
    "a", "an", "the", "photo", "picture", "image", "of", "portrait", "headshot", "close-up", "face", "faces",
    "man", "woman", "men", "women", "person", "people", "girl", "boy", "female", "male", "human",
    "young", "youthful", "mature", "senior", "old", "adult", "teen", "teenager",
    "attractive", "handsome", "beautiful", "pretty", "gorgeous", "cute", "charming", "elegant",
    # Expressions & Emotions
    "smiling", "smile", "smiles", "happy", "cheerful", "laughing", "grin", "grinning",
    "neutral", "serious", "calm", "pensive", "confident", "friendly", "expression",
    # Hair Colors, Styles & Textures
    "hair", "hairstyle", "blonde", "blond", "black", "brown", "brunette", "gray", "grey", "white", "red",
    "dark", "light", "golden", "silver", "auburn", "chestnut",
    "wavy", "straight", "curly", "kinky", "short", "long", "medium", "length",
    "bangs", "fringe", "ponytail", "bun", "braids", "bald", "balding", "receding", "hairline",
    # Facial Features & Facial Hair
    "beard", "bearded", "mustache", "moustache", "goatee", "sideburns", "stubble", "shadow", "shaved", "clean",
    "eyes", "blue", "green", "brown", "hazel", "amber", "dark-eyed", "eyebrows", "arched", "bushy", "thin",
    "lips", "mouth", "nose", "cheeks", "cheekbones", "jaw", "jawline", "chin", "ears", "forehead",
    "skin", "pale", "fair", "tan", "tanned", "olive", "smooth", "clear", "rosy",
    # Accessories & Cosmetics
    "wearing", "with", "eyeglasses", "glasses", "spectacles", "sunglasses", "shades", "frames",
    "hat", "cap", "beanie", "fedora", "headband", "earrings", "necklace", "jewelry",
    "tie", "necktie", "bow-tie", "collar", "shirt", "suit", "jacket", "dress",
    "makeup", "lipstick", "red", "pink", "glossy", "eyeliner", "mascara", "blush",
    # Actions, Poses & Composition
    "looking", "look", "looks", "staring", "facing", "posing", "turned", "profile",
    "front", "frontal", "view", "viewer", "camera", "direct", "gaze",
    "indoor", "indoors", "outdoor", "outdoors", "studio", "background", "backdrop",
    "lighting", "soft", "cinematic", "dramatic", "natural", "bokeh", "blurred",
    # Connectors & General Adjectives
    "in", "on", "at", "by", "near", "against", "and", "or", "having", "holding",
    "detailed", "sharp", "clean", "high", "quality", "hd", "4k", "photorealistic",
    # Animals & props fallback
    "dog", "dogs", "puppy", "canine", "cat", "pet", "animal", "golden", "retriever", "husky"
]

class SimpleTokenizer:
    """
    Deterministic, offline tokenizer for text-conditioned generation.
    Word-level parsing with subword/character fallback, ensuring 100% offline reliability.
    """
    def __init__(self, vocab_file: str = None, max_vocab_size: int = 4096):
        self.max_vocab_size = max_vocab_size
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"
        
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self._init_default_vocab()

    def _init_default_vocab(self):
        special = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        vocab = list(special)
        
        for w in CORE_VOCAB:
            w_clean = w.lower().strip()
            if w_clean and w_clean not in vocab:
                vocab.append(w_clean)
                
        # Add single ASCII characters for byte/char fallback
        for c in "abcdefghijklmnopqrstuvwxyz0123456789.,!?-_\'\"":
            if c not in vocab:
                vocab.append(c)
                
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.pad_token]

    @property
    def start_id(self) -> int:
        return self.word2idx[self.start_token]

    @property
    def end_id(self) -> int:
        return self.word2idx[self.end_token]

    @property
    def unk_id(self) -> int:
        return self.word2idx[self.unk_token]

    def add_words(self, words: List[str]):
        """Expands vocabulary with new domain-specific words."""
        for word in words:
            word = word.lower().strip()
            if word and word not in self.word2idx:
                if len(self.word2idx) < self.max_vocab_size:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def tokenize_text(self, text: str) -> List[str]:
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def encode(self, text: str, max_length: int = 64) -> List[int]:
        raw_tokens = self.tokenize_text(text)
        token_ids = [self.start_id]
        
        for token in raw_tokens:
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            else:
                for char in token:
                    token_ids.append(self.word2idx.get(char, self.unk_id))
            if len(token_ids) >= max_length - 1:
                break
                
        token_ids.append(self.end_id)
        
        if len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_id] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
            token_ids[-1] = self.end_id
            
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for tid in token_ids:
            if tid in (self.pad_id, self.start_id, self.end_id):
                continue
            words.append(self.idx2word.get(tid, self.unk_token))
        return " ".join(words)

    def save_vocab(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, indent=2)

    def load_vocab(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}
