"""Coarse domain heuristics for dataset metadata.

The mappings prefer exact dataset class and dataset-name matches, then fall
back to keyword rules when no explicit hint is available.
"""

# Direct mapping from dataset class name to a coarse domain string.
CLASS_TO_DOMAIN = {
    # Molecular / materials science
    "MoleculeNet": "molecules",
    "QM7b": "molecules",
    "QM9": "molecules",
    "ZINC": "molecules",
    # Citation / academic graphs
    "Planetoid": "citation",
    "CitationFull": "citation",
    "CoraFull": "citation",
    "WikiCS": "citation",
    "WebKB": "citation",
    "WikipediaNetwork": "citation",
    "Coauthor": "citation",
    "LINKXDataset": "citation",
    # Social / communication / e-commerce
    "Actor": "social",
    "Amazon": "social",
    "Reddit": "social",
    "Reddit2": "social",
    "Flickr": "social",
    "FacebookPagePage": "social",
    "GitHub": "social",
    "EmailEUCore": "social",
    "LastFMAsia": "social",
    "GemsecDeezer": "social",
    "Twitch": "social",
    # Finance / transactions
    "EllipticBitcoinDataset": "finance",
    # Transport / infrastructure
    "Airports": "transport",
    # Benchmarks / synthetic generators
    "LRGBDataset": "benchmark",
    "GNNBenchmarkDataset": "benchmark",
    "AttributedGraphDataset": "benchmark",
    "HeterophilousGraphDataset": "benchmark",
    # Collections / wrappers
    "SNAPDataset": "collection",
    "SuiteSparseMatrixCollection": "collection",
    "TUDataset": "collection",
}

# Mapping based on dataset.name (lowercased) when available.
NAME_TO_DOMAIN = {
    # Citation-style datasets
    "cora": "citation",
    "citeseer": "citation",
    "pubmed": "citation",
    "dblp": "citation",
    "corafull": "citation",
    "cora_ml": "citation",
    "computers": "social",
    "photo": "social",
    "wikipedia": "citation",
    "wikics": "citation",
    "cornell": "citation",
    "texas": "citation",
    "wisconsin": "citation",
    "chameleon": "citation",
    "squirrel": "citation",
    "crocodile": "citation",
    # LINKXDataset (facebook) variants
    "amherst41": "social",
    "cornell5": "social",
    "johnshopkins55": "social",
    "penn94": "social",
    "reed98": "social",
    "genius": "social",
    # Heterophilous benchmark variants
    "amazon-ratings": "benchmark",
    "roman-empire": "benchmark",
    "tolokers": "benchmark",
    "minesweeper": "benchmark",
    "questions": "benchmark",
    # Airports
    "airports": "transport",
    # Finance
    "elliptic_bitcoin": "finance",
    # Social specifics
    "facebook_page-page": "social",
    "email_eu_core": "social",
    "lastfm_asia": "social",
    "deezer_europe": "social",
    "gemsec_deezer": "social",
    "coco-sp": "benchmark",
    "pascalvoc-sp": "benchmark",
    "twitch-de": "social",
    "twitch-en": "social",
    "twitch-es": "social",
    "twitch-fr": "social",
    "twitch-ru": "social",
    "twitch-tw": "social",
    "twitch-uk": "social",
    "twitch-us": "social",
    # Molecular datasets (mostly MoleculeNet subsets)
    "bbbp": "molecules",
    "tox21": "molecules",
    "esol": "molecules",
    "freesolv": "molecules",
    "sider": "molecules",
    "clintox": "molecules",
    "hiv": "molecules",
    "muv": "molecules",
    "toxcast": "molecules",
    "bace": "molecules",
    "pcba": "molecules",
    "qm8": "molecules",
    "qm9": "molecules",
    "zinc": "molecules",
    "pepfunc": "molecules",
    "pepstruct": "molecules",
    "pcqm-contact": "molecules",
    "peptides-func": "molecules",
    "peptides-struct": "molecules",
    # Social / community
    "reddit": "social",
    "reddit2": "social",
    "lastfm": "social",
    "deezer": "social",
    "deezer_europe": "social",
    "twitch": "social",
    "gemsec_deezer": "social",
    "facebook": "social",
    "github": "social",
    "email": "social",
    "polblogs": "social",
    "yelp": "social",
    # OGB common datasets
    "ogbn-arxiv": "citation",
    "ogbn-products": "social",
    "ogbn-proteins": "biology",
    "ogbg-molhiv": "molecules",
    "ogbg-molpcba": "molecules",
}

# Keyword fallbacks checked after exact matches (lowercased compare).
KEYWORD_DOMAINS = [
    ("molecules", ("qm", "zinc", "mol", "pcqm", "tox", "drug", "chem", "bio", "pept", "pcba", "hiv")),
    ("citation", ("cora", "citeseer", "pubmed", "dblp", "citation", "wikics", "arxiv")),
    ("social", ("reddit", "amazon", "social", "twitch", "facebook", "deezer", "lastfm", "github", "yelp", "blog")),
    ("knowledge_graph", ("fb15", "wordnet", "wikidata", "nell", "dbp", "icews", "gdel", "kg")),
    ("vision3d", ("shrec", "modelnet", "shapenet", "faust", "tosca", "voc", "pascal", "mnist", "s3dis")),
]
