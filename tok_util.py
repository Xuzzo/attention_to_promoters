import sys
import os

sys.path.insert(0,'/Users/mmfp/opt/anaconda3/lib/python3.7/site-packages/')
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages')

from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx


class GenomicTokenizer(BaseTokenizer):
    def __init__(self, lang='en', ngram=5, stride=2):
        self.lang = lang
        self.ngram = ngram
        self.stride = stride
        
    def tokenizer(self, t):
        t = t.upper()
        if self.ngram == 1:
            toks = list(t)
        else:
            toks = [t[i:i+self.ngram] for i in range(0, len(t), self.stride) if len(t[i:i+self.ngram]) == self.ngram]
        if len(toks[-1]) < self.ngram:
            toks = toks[:-1]
        
        return toks


class GenomicVocab(Vocab):
    def __init__(self, itos):
        self.itos = itos
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
        
    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c >= min_freq]
        itos.insert(0, 'pad')
        return cls(itos)


def _genomic_join_texts(texts:Collection[str], mark_fields:bool=False):
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    text_col = f'{BOS} {FLD} {1} ' + df[0].astype(str) if mark_fields else  '' + df[0].astype(str)
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)   
    return text_col.values


def _get_genomic_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                   min_freq:int=2, mark_fields:bool=False):
    return [GenomicTokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, mark_fields=mark_fields),
            GenomicNumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]


class GenomicTextClasDataBunch(TextClasDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
                min_freq:int=2, mark_fields:bool=False, pad_idx=0, pad_first=True, bs=64, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames. `kwargs` are passed to the dataloader creation."
        processor = _get_genomic_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        d1 = src.databunch(**kwargs)
        
        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=False)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=bs, sampler=sampler, **kwargs))
            
        return cls(*dataloaders, path=path, collate_fn=collate_fn)


class GenomicTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000, mark_fields:bool=False):
        self.tokenizer,self.chunksize,self.mark_fields = ifnone(tokenizer, Tokenizer()),chunksize,mark_fields

    def process_one(self, item):  
        return self.tokenizer._process_all_1(_genomic_join_texts([item], self.mark_fields))[0]
    
    def process(self, ds):
        ds.items = _genomic_join_texts(ds.items, self.mark_fields)
        tokens = []
        for i in range(0,len(ds),self.chunksize):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

 
class GenomicNumericalizeProcessor(PreProcessor):
    "`PreProcessor` that numericalizes the tokens in `ds`."
    def __init__(self, ds:ItemList=None, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=3):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def process_one(self,item): return np.array(self.vocab.numericalize(item), dtype=np.int64)
    def process(self, ds):
        if self.vocab is None: self.vocab = GenomicVocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        super().process(ds)


class GenomicVocab(Vocab):
    def __init__(self, itos):
        self.itos = itos
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
        
    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c >= min_freq]
        itos.insert(0, 'pad')
        return cls(itos)


def _genomic_join_texts(texts:Collection[str], mark_fields:bool=False):
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    text_col = f'{BOS} {FLD} {1} ' + df[0].astype(str) if mark_fields else  '' + df[0].astype(str)
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)   
    return text_col.values


def _get_genomic_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                   min_freq:int=2, mark_fields:bool=False):
    return [GenomicTokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, mark_fields=mark_fields),
            GenomicNumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]


class GenomicTextClasDataBunch(TextClasDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
                min_freq:int=2, mark_fields:bool=False, pad_idx=0, pad_first=True, bs=64, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames. `kwargs` are passed to the dataloader creation."
        processor = _get_genomic_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        d1 = src.databunch(**kwargs)
        
        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=False)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=bs, sampler=sampler, **kwargs))
            
        return cls(*dataloaders, path=path, collate_fn=collate_fn)