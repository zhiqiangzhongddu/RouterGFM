from yacs.config import CfgNode as CN

from ._dataset import _default_dataset_cfg


# Paper Sec. 4.1: nine target applications across four task families.
ROUTERGFM_TARGET_DATASETS = [
    "photo:node",
    "ogbn-arxiv:node",
    "airports:node",
    "chameleon:node",
    "dblp:edge",
    "cornell:edge",
    "qm7b:graph",
    "toxcast:graph",
    "mnist:graph",
]

# Paper App. B.1: the twelve source corpora of the 588-expert pool. They also
# serve as additional historical applications (same-source experts excluded),
# which supplies same-family evidence for LP, multi-label, and regression.
ROUTERGFM_SOURCE_DATASETS = [
    "cora", "pubmed", "computers", "flickr", "actor", "squirrel", "email",
    "bbbp", "tox21", "qm9", "proteins", "cifar10",
]

ROUTERGFM_HISTORY_EXTRA = [
    "cora:node", "pubmed:node", "computers:node", "flickr:node",
    "actor:node", "squirrel:node", "email:node",
    "cora:edge", "actor:edge", "squirrel:edge", "email:edge",
    "bbbp:graph", "tox21:graph", "qm9:graph", "proteins:graph", "cifar10:graph",
]


def set_routergfm_cfg(cfg: CN) -> None:
    """Attach RouterGFM defaults (paper Sec. 3, App. B) to ``cfg.moe.routergfm``.

    Stages (``task``):
      history   fit frozen-expert task heads on historical supports and record
                per-instance routing losses on diagnostic sets (Alg. 1, l.1-4)
      router    build the context graph + local archive and train the router
                with application-masked episodes (Alg. 1, l.5-12)
      deploy    select a team for a target application, fit its heads, and mix
                predictions with context-dependent weights (Alg. 1, l.13-21)
      benchmark multi-seed deploy (+ matched-pool baselines) -> results TSV
      analysis  paper diagnostics (insertion, calibration, team size, archive
                reliability, specialization, shift)
    """
    rg = cfg.moe.routergfm = CN()
    # history | descriptors | router | deploy | benchmark | analysis | selection_baseline | matched_baseline
    rg.task = "benchmark"
    rg.output_root = "outputs/routergfm"  # all RouterGFM artifacts live below this root
    rg.device_batch_size = 256  # batch size for frozen-encoder embedding passes

    # ------------------------------------------------------------------ #
    # Expert pool (App. B.1): 7 architectures x 7 objectives x 12 sources
    # ------------------------------------------------------------------ #
    rg.experts = CN()
    rg.experts.checkpoint_root = "outputs/pretrained_models"
    rg.experts.architectures = ["gcn", "gat", "gin", "h2gcn", "fagcn", "nodeformer", "transformer"]
    rg.experts.objectives = [
        "attr_masking", "context_pred", "dgi", "edge_pred", "graphcl", "infograph", "supervised",
    ]
    rg.experts.sources = list(ROUTERGFM_SOURCE_DATASETS)
    rg.experts.strict = True  # fail when an (arch, objective, source) checkpoint is missing
    rg.experts.exclude_same_source = True  # empty diagonal: never evaluate an expert on its own source
    rg.experts.shard_index = 0  # history generation: process experts[i::num_shards]
    rg.experts.num_shards = 1

    # ------------------------------------------------------------------ #
    # Applications: (dataset, task level, support budget, split seed)
    # ------------------------------------------------------------------ #
    rg.apps = CN()
    rg.apps.targets = list(ROUTERGFM_TARGET_DATASETS)  # "dataset:task_level"
    rg.apps.history_extra = list(ROUTERGFM_HISTORY_EXTRA)  # extra historical-only applications
    rg.apps.budgets = [5, 100]  # support examples per class (count for regression / multi-label)
    rg.apps.lp_split = (0.1, 0.05, 0.1)  # App. B.3 routed LP convention (positive-edge ratios)
    rg.apps.seeds = [42, 0, 100, 123, 2024]
    rg.apps.history_seeds = [42, 0, 100, 123, 2024]  # seeds recorded for historical applications
    rg.apps.max_diagnostic = 2000  # cap on |D_a| recorded per historical application (fixed subsample)
    rg.apps.data = _default_dataset_cfg()  # dataset construction; must match the experts' pretraining
    rg.apps.data.induced = True

    # ------------------------------------------------------------------ #
    # Task heads fitted on frozen encoders (Sec. 3.1, App. B.1)
    # ------------------------------------------------------------------ #
    rg.heads = CN()
    rg.heads.type = "linear"  # linear | mlp
    rg.heads.hidden_dim = 128
    rg.heads.epochs = 300
    rg.heads.lr = 0.01
    rg.heads.weight_decay = 5e-4
    rg.heads.standardize_inputs = True  # z-score embeddings with support statistics before the head
    rg.heads.oof_folds = 5  # out-of-fold support predictions for support-fitted integration rules

    # ------------------------------------------------------------------ #
    # Routing losses (App. B.2)
    # ------------------------------------------------------------------ #
    rg.loss = CN()
    rg.loss.regression = "abs"  # abs | sq : error in support median/MAD-normalized units
    rg.loss.scale_floor = 1e-6  # MAD floor for regression normalization

    # ------------------------------------------------------------------ #
    # Label-free context descriptors z_a(x) (Sec. 3.1, 3.4)
    # ------------------------------------------------------------------ #
    rg.descriptors = CN()
    rg.descriptors.num_spectral = 6  # leading Laplacian eigenvalues / feature singular values kept
    rg.descriptors.clip = 5.0  # clip standardized descriptors to [-clip, clip]

    # ------------------------------------------------------------------ #
    # Local archive M (Eq. 5)
    # ------------------------------------------------------------------ #
    rg.archive = CN()
    rg.archive.num_cells = 16  # B_b: k-means cells per historical application
    rg.archive.min_cell_size = 5  # reduce B_b so that |D_b| / B_b >= min_cell_size
    rg.archive.kmeans_iters = 50
    rg.archive.num_families = 4  # coarse context families (for the missing-family diagnostic)
    # none | half_cells | missing_family | reversed | shuffled (App. D.2 / D.5 controls)
    rg.archive.perturbation = "none"
    rg.archive.perturbation_seed = 0

    # ------------------------------------------------------------------ #
    # Heterogeneous context graph H (Sec. 3.2) and node features (Sec. 3.3)
    # ------------------------------------------------------------------ #
    rg.graph = CN()
    rg.graph.text_backend = "bert"  # bert | hash (hash: deterministic offline embedding for tests)
    rg.graph.text_model = "bert-base-uncased"
    rg.graph.text_max_length = 256
    rg.graph.text_cache_dir = "outputs/routergfm/text_cache"
    rg.graph.local_files_only = False
    rg.graph.hash_dim = 64  # embedding width of the hash backend
    rg.graph.description_dir = ""  # optional JSON descriptions: <dir>/{dataset,architecture,objective}/<name>.json
    rg.graph.use_text = True
    rg.graph.use_numeric = True

    # ------------------------------------------------------------------ #
    # Router: encoder, scorer, retrieval (Eq. 3-8), training (Eq. 9-11)
    # ------------------------------------------------------------------ #
    rg.router = CN()
    rg.router.hidden_dim = 128
    rg.router.num_layers = 2
    rg.router.dropout = 0.1
    rg.router.key_dim = 32  # d_k of k_phi
    rg.router.key_hidden_dim = 128
    rg.router.topk = 5  # K
    rg.router.retrieval_j = 32  # J nearest archive records
    rg.router.per_app_cap = 8  # max retrieved records from one source application
    rg.router.bandwidth = 0.3  # h in Eq. 6 for L_loc during training (and at deployment unless select_bandwidth)
    rg.router.bandwidth_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    rg.router.select_bandwidth = True  # select h jointly with rho / tau on validation applications from bandwidth_grid
    rg.router.rho = -1.0  # Eq. 7 in [0,1]; < 0 selects rho on validation applications from rho_grid
    rg.router.rho_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    rg.router.rho_train = 1.0  # rho used inside L_loc during training
    rg.router.tau = -1.0  # Eq. 8 temperature; < 0 selects tau on validation applications from tau_grid
    rg.router.tau_grid = [0.01, 0.02, 0.05, 0.1, 0.2]
    rg.router.huber_delta = 0.1
    rg.router.lambda_rank = 0.1  # lambda_r (ListMLE)
    rg.router.lambda_local = 1.0  # lambda_l (local squared loss)
    rg.router.lr = 1e-3
    rg.router.weight_decay = 1e-5
    rg.router.epochs = 200
    rg.router.episodes_per_step = 8  # pseudo-target applications per update
    rg.router.local_pairs_per_episode = 2048  # sampled (x, e) pairs for L_loc per episode
    rg.router.patience = 30
    rg.router.val_datasets = []  # base datasets held out for router selection; [] -> automatic
    rg.router.num_val_datasets = 2  # automatic choice: same-family datasets first
    rg.router.seed = 42
    rg.router.per_seed = False  # train one router per target seed instead of per (dataset, budget)
    rg.router.skip_if_exists = True

    # ------------------------------------------------------------------ #
    # Prediction integration (Sec. 3.4; fixed-team rules of App. C/D.2)
    # ------------------------------------------------------------------ #
    rg.integration = CN()
    # Rules evaluated in one deploy pass on the same team and fitted heads.
    rg.integration.rules = [
        "routergfm", "global", "uniform", "simplex_stacking", "local_mlp", "no_centering", "shuffled",
    ]
    rg.integration.stacking_epochs = 300
    rg.integration.stacking_lr = 0.05
    rg.integration.local_mlp_hidden = 64
    rg.integration.local_mlp_epochs = 300
    rg.integration.local_mlp_lr = 0.01
    rg.integration.local_mlp_weight_decay = 1e-3
    rg.integration.eval_cells = 16  # context partition of the target queries for worst-cell / coverage

    # ------------------------------------------------------------------ #
    # Deployment / benchmark
    # ------------------------------------------------------------------ #
    rg.deploy = CN()
    rg.deploy.target = "photo:node"  # "dataset:task_level"
    rg.deploy.budget = 5
    rg.deploy.seed = 42
    rg.benchmark = CN()
    rg.benchmark.num_runs = 5
    rg.benchmark.methods = ["routergfm"]  # routergfm plus any matched-pool baseline keys
    rg.benchmark.run_tasks_tsv = False
    rg.benchmark.tasks_tsv = "slurm/moe.routergfm.tsv"

    # ------------------------------------------------------------------ #
    # Matched-pool baselines (App. C): selection, frozen-expert mixtures
    # ------------------------------------------------------------------ #
    # Per-method subtrees are attached by src/config/_moe_routergfm_<method>.py.
    rg.baselines = CN()
    rg.baselines.method = ""  # selection: metadata_mlp | nearest_application | metagl | metagl_metadata | logme | model_spider
    #                          # matched:   metagl_u | sagmm_pe | meta_des | kdem | ppem
    # "dataset:task_level" evaluated by a baseline run; Table 9 selection runs use the five node / single-label
    # graph classification targets (photo, ogbn-arxiv, airports, chameleon, mnist)
    rg.baselines.datasets = list(ROUTERGFM_TARGET_DATASETS)
    rg.baselines.budgets = [5, 100]
    rg.baselines.num_runs = 5  # seeds taken from apps.seeds
    rg.baselines.topk = 5  # team / shortlist size K
    rg.baselines.candidate_pool = 16  # M candidate experts for SAGMM-PE / META-DES (<= 0: all of E_a)
    # historical_mean (mean normalised rank over same-family historical apps; never-observed experts last,
    # ties in catalog order) | eligible (all of E_a) | random; see src/moe/routergfm/baselines/candidates.py
    rg.baselines.candidate_rule = "historical_mean"
    rg.baselines.run_tasks_tsv = False
    # header: method dataset task_level budget (selection baselines: slurm/moe.routergfm_selection.tsv)
    rg.baselines.tasks_tsv = "slurm/moe.routergfm_baselines.tsv"
    rg.baselines.output_dir = "outputs/routergfm/baselines"
    rg.baselines.skip_if_exists = True

    # ------------------------------------------------------------------ #
    # Analyses (App. D)
    # ------------------------------------------------------------------ #
    rg.analysis = CN()
    # insertion | calibration | team_size | archive_reliability | specialization | shift
    rg.analysis.kind = "team_size"
    rg.analysis.team_sizes = [1, 2, 3, 5, 8, 16]
    rg.analysis.calibration_apps = [0, 1, 2, 4, 8]
    rg.analysis.holdout_architecture = "transformer"
    rg.analysis.holdout_fraction = 0.2  # new-expert (seen factors) holdout share
    rg.analysis.shift_root = "data/splits_shift"
