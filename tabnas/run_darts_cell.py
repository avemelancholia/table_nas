import logging
import numpy as np
import os
import pickle
import copy
import random
import torch
from fvcore.common.config import CfgNode
from pathlib import Path

from data import TabNasDataset, TabNasTorchDataset
from naslib import utils
from naslib.defaults.trainer import Trainer
from naslib.utils import set_seed, setup_logger, get_config_from_args


from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core import primitives as core_ops
from naslib.search_spaces.nasbenchasr.conversions import (
    flatten,
    copy_structure,
    make_compact_mutable,
    make_compact_immutable,
)
from naslib.search_spaces.nasbenchasr.encodings import encode_asr
from naslib.utils.encodings import EncodingType
from naslib.utils.log import log_every_n_seconds, log_first_n
from naslib.search_spaces.core.primitives import AbstractPrimitive, Identity

from naslib.optimizers import (
    DARTSOptimizer,
    GSparseOptimizer,
    OneShotNASOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
)
from operations import (
    ResBlockLinearOP,
    LinearOP,
    LinearBottleneckOP,
    HeadOP,
    TransformerOP,
)
from log_utils import parse_log


class TabNasTrainer(Trainer):
    def _store_accuracies(self, logits, target, split):
        """Update the accuracy counters"""
        logits = logits.clone().detach().cpu()
        target = target.clone().detach().cpu()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
        n = logits.size(0)

        if split == "train":
            self.train_top1.update(prec1.data.item(), n)
            self.train_top5.update(prec5.data.item(), n)
        elif split == "val":
            self.val_top1.update(prec1.data.item(), n)
            self.val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError(
                "Unknown split: {}. Expected either 'train' or 'val'")

    def evaluate(
        self,
        retrain: bool = True,
        search_model: str = "",
        resume_from: str = "",
        best_arch: Graph = None,
        dataset_api: object = None,
        metric: Metric = None,
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool)      : Reset the weights from the architecure search
            search_model (str)  : Path to checkpoint file that was created during search. If not provided,
                                  then try to load 'model_final.pth' from search
            resume_from (str)   : Resume retraining from the given checkpoint file.
            best_arch           : Parsed model you want to directly evaluate and ignore the final model
                                  from the optimizer.
            dataset_api         : Dataset API to use for querying model performance.
            metric              : Metric to query the benchmark for.
        """
        logger.info("Start evaluation")
        if not best_arch:
            if not search_model:
                search_model = os.path.join(
                    self.config.save, "search", "model_final.pth"
                )
            # required to load the architecture
            self._setup_checkpointers(search_model)

            best_arch = self.optimizer.get_final_architecture()
        # logger.info(f"Final architecture hash: {best_arch.get_hash()}")

        if True:
            best_arch.to(self.device)
            if retrain:
                logger.info("Starting retraining from scratch")
                best_arch.reset_weights(inplace=True)

                (
                    self.train_queue,
                    self.valid_queue,
                    self.test_queue,
                ) = self.build_eval_dataloaders(self.config)

                optim = self.build_eval_optimizer(
                    best_arch.parameters(), self.config)
                scheduler = self.build_eval_scheduler(optim, self.config)

                start_epoch = self._setup_checkpointers(
                    resume_from,
                    search=False,
                    period=self.config.evaluation.checkpoint_freq,
                    model=best_arch,  # checkpointables start here
                    optim=optim,
                    scheduler=scheduler,
                )

                grad_clip = self.config.evaluation.grad_clip
                loss = torch.nn.CrossEntropyLoss()

                self.train_top1.reset()
                self.train_top5.reset()
                self.val_top1.reset()
                self.val_top5.reset()

                # Enable drop path
                best_arch.update_edges(
                    update_func=lambda edge: edge.data.set(
                        "op", DropPathWrapper(edge.data.op)
                    ),
                    scope=best_arch.OPTIMIZER_SCOPE,
                    private_edge_data=True,
                )

                # train from scratch
                epochs = self.config.evaluation.epochs
                for e in range(start_epoch, epochs):
                    best_arch.train()

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(
                                torch.cuda.memory_summary()),
                            n=20,
                        )

                    # update drop path probability
                    drop_path_prob = self.config.evaluation.drop_path_prob * e / epochs
                    best_arch.update_edges(
                        update_func=lambda edge: edge.data.set(
                            "drop_path_prob", drop_path_prob
                        ),
                        scope=best_arch.OPTIMIZER_SCOPE,
                        private_edge_data=True,
                    )

                    # Train queue
                    for i, (input_train, target_train) in enumerate(self.train_queue):
                        input_train = input_train.to(self.device)
                        target_train = target_train.to(
                            self.device, non_blocking=True)

                        optim.zero_grad()
                        logits_train = best_arch(input_train)
                        train_loss = loss(logits_train, target_train)
                        if hasattr(
                            best_arch, "auxilary_logits"
                        ):  # darts specific stuff
                            log_first_n(
                                logging.INFO, "Auxiliary is used", n=10)
                            auxiliary_loss = loss(
                                best_arch.auxilary_logits(), target_train
                            )
                            train_loss += (
                                self.config.evaluation.auxiliary_weight * auxiliary_loss
                            )
                        train_loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                best_arch.parameters(), grad_clip
                            )
                        optim.step()

                        self._store_accuracies(
                            logits_train, target_train, "train")
                        log_every_n_seconds(
                            logging.INFO,
                            "Epoch {}-{}, Train loss: {:.5}, learning rate: {}".format(
                                e, i, train_loss, scheduler.get_last_lr()
                            ),
                            n=5,
                        )

                    # Validation queue
                    if self.valid_queue:
                        best_arch.eval()
                        for i, (input_valid, target_valid) in enumerate(
                            self.valid_queue
                        ):
                            input_valid = input_valid.to(self.device).float()
                            target_valid = target_valid.to(self.device).float()

                            # just log the validation accuracy
                            with torch.no_grad():
                                logits_valid = best_arch(input_valid)
                                self._store_accuracies(
                                    logits_valid, target_valid, "val"
                                )

                    scheduler.step()
                    self.periodic_checkpointer.step(e)
                    self._log_and_reset_accuracies(e)

            # Disable drop path
            best_arch.update_edges(
                update_func=lambda edge: edge.data.set(
                    "op", edge.data.op.get_embedded_ops()
                ),
                scope=best_arch.OPTIMIZER_SCOPE,
                private_edge_data=True,
            )

            # measure final test accuracy
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            best_arch.eval()

            for i, data_test in enumerate(self.test_queue):
                input_test, target_test = data_test
                input_test = input_test.to(self.device)
                target_test = target_test.to(self.device, non_blocking=True)

                n = input_test.size(0)

                with torch.no_grad():
                    logits = best_arch(input_test)

                    prec1, prec5 = utils.accuracy(
                        logits, target_test, topk=(1, 1))
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

                log_every_n_seconds(
                    logging.INFO,
                    "Inference batch {} of {}.".format(
                        i, len(self.test_queue)),
                    n=5,
                )

            logger.info(
                "Evaluation finished. Test accuracies: top-1 = {:.5}, top-5 = {:.5}".format(
                    top1.avg, top5.avg
                )
            )

            return top1.avg

    @staticmethod
    def build_search_dataloaders(config):
        return train_queue, valid_queue, None

    @staticmethod
    def build_eval_dataloaders(config):
        return train_queue, valid_queue, test_queue


class DropPathWrapper(AbstractPrimitive):
    """
    A wrapper for the drop path training regularization.
    """

    def __init__(self, op, device="cuda:0"):
        super().__init__(locals())
        self.op = op
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")

    def forward(self, x, edge_data):
        x = self.op(x, edge_data)
        if (
            edge_data.drop_path_prob > 0.0
            and not isinstance(self.op, Identity)
            and self.training
        ):
            keep_prob = 1.0 - edge_data.drop_path_prob
            mask = torch.FloatTensor(x.size(0), 1).bernoulli_(keep_prob)
            mask = mask.to(self.device)
            x.div_(keep_prob)
            x.mul_(mask)
        return x

    def get_embedded_ops(self):
        return self.op


class TabNasSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nas-bench-asr.
    Note: currently we do not support building a naslib object for
    nas-bench-asr architectures.
    """

    QUERYABLE = True
    OPTIMIZER_SCOPE = [
        "cells_stage_1",
        "cells_stage_2",
        "cells_stage_3",
        "cells_stage_4",
    ]

    def __init__(self):
        super().__init__()
        self.load_labeled = False
        self.max_epoch = 100
        self.max_nodes = 3
        self.accs = None
        self.compact = None

        self.n_blocks = NUM_BLOCKS
        self.n_cells_per_block = [1] * NUM_BLOCKS
        self.features = NUM_FEATURES
        self.filters = [NUM_FILTERS] * NUM_BLOCKS

        self.num_classes = NUM_CLASSES
        self.dropout_rate = 0.0
        self.use_norm = True

        self._create_macro_graph()

    def _create_macro_graph(self):
        cell = self._create_cell()

        # Macrograph defintion
        n_nodes = self.n_blocks + 2
        self.add_nodes_from(range(1, n_nodes + 1))

        for node in range(1, n_nodes):
            self.add_edge(node, node + 1)

        # Create the cell blocks and add them as subgraphs of nodes 2 ... 5
        for idx, node in enumerate(range(2, 2 + self.n_blocks)):
            scope = f"cells_stage_{idx + 1}"
            cells_block = self._create_cells_block(
                cell, n=self.n_cells_per_block[idx], scope=scope
            )
            self.nodes[node]["subgraph"] = cells_block.set_input([node - 1])

            # Assign the list of operations to the cell edges
            cells_block.update_edges(
                update_func=lambda edge: _set_cell_edge_ops(
                    edge, filters=self.filters[idx], use_norm=self.use_norm
                ),
                scope=scope,
                private_edge_data=True,
            )

        start_node = 1
        for idx, node in enumerate(range(start_node, start_node + self.n_blocks)):
            if node == start_node:
                op = LinearOP(self.features, self.filters[0])
            else:
                op = core_ops.Identity()

            self.edges[node, node + 1].set("op", op)
        # Assign the LSTM + Linear layer to the last edge in the macro graph
        self.edges[self.n_blocks + 1, self.n_blocks + 2].set(
            "op", HeadOP(self.filters[0], self.num_classes)
        )

    def _create_cells_block(self, cell, n, scope):
        block = Graph()
        block.name = f"{n}_cells_block"

        block.add_nodes_from(range(1, n + 2))

        for node in range(2, n + 2):
            block.add_node(
                node, subgraph=cell.copy().set_scope(
                    scope).set_input([node - 1])
            )

        for node in range(1, n + 2):
            block.add_edge(node, node + 1)

        return block

    def _create_cell(self):
        normal_cell = Graph()
        normal_cell.name = (
            "cell"  # Use the same name for all cells with shared attributes
        )

        # Input nodes
        normal_cell.add_node(1)
        normal_cell.add_node(2)

        # Intermediate nodes
        normal_cell.add_node(3)
        normal_cell.add_node(4)
        normal_cell.add_node(5)
        normal_cell.add_node(6)

        # Output node
        normal_cell.add_node(7)

        # Edges
        for i in range(2, 7):
            normal_cell.add_edge(1, i)  # input 1
            # normal_cell.add_edge(2, i)
            normal_cell.add_edge(i, 7)

        normal_cell.add_edges_from([(3, 4), (3, 5), (3, 6)])
        normal_cell.add_edges_from([(4, 5), (4, 6)])
        normal_cell.add_edges_from([(5, 6)])

        # Edges connecting to the output are always the identity

        return normal_cell

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from nas-bench-asr
        """
        metric_to_asr = {
            Metric.VAL_ACCURACY: "val_per",
            Metric.TEST_ACCURACY: "test_per",
            Metric.PARAMETERS: "params",
            Metric.FLOPS: "flops",
        }

        assert self.compact is not None
        assert metric in [
            Metric.TRAIN_ACCURACY,
            Metric.TRAIN_LOSS,
            Metric.VAL_ACCURACY,
            Metric.TEST_ACCURACY,
            Metric.PARAMETERS,
            Metric.FLOPS,
            Metric.TRAIN_TIME,
            Metric.RAW,
        ]
        query_results = dataset_api["asr_data"].full_info(self.compact)

        if metric != Metric.VAL_ACCURACY:
            if metric == Metric.TEST_ACCURACY:
                return query_results[metric_to_asr[metric]]
            elif (metric == Metric.PARAMETERS) or (metric == Metric.FLOPS):
                return query_results["info"][metric_to_asr[metric]]
            elif metric in [
                Metric.TRAIN_ACCURACY,
                Metric.TRAIN_LOSS,
                Metric.TRAIN_TIME,
                Metric.RAW,
            ]:
                return -1
        else:
            if full_lc and epoch == -1:
                return [loss for loss in query_results[metric_to_asr[metric]]]
            elif full_lc and epoch != -1:
                return [loss for loss in query_results[metric_to_asr[metric]][:epoch]]
            else:
                # return the value of the metric only at the specified epoch
                return float(query_results[metric_to_asr[metric]][epoch])

    def get_compact(self):
        assert self.compact is not None
        return self.compact

    def get_hash(self):
        return self.get_compact()

    def set_compact(self, compact):
        self.compact = make_compact_immutable(compact)

    def sample_random_architecture(self, dataset_api):
        search_space = [
            [len(OP_NAMES)] + [2] * (idx + 1) for idx in range(self.max_nodes)
        ]
        flat = flatten(search_space)
        m = [random.randrange(opts) for opts in flat]
        m = copy_structure(m, search_space)

        compact = m
        self.set_compact(compact)
        return compact

    def mutate(self, parent, mutation_rate=1, dataset_api=None):
        """
        This will mutate the cell in one of two ways:
        change an edge; change an op.
        Todo: mutate by adding/removing nodes.
        Todo: mutate the list of hidden nodes.
        Todo: edges between initial hidden nodes are not mutated.
        """
        parent_compact = parent.get_compact()
        parent_compact = make_compact_mutable(parent_compact)
        compact = copy.deepcopy(parent_compact)

        for _ in range(int(mutation_rate)):
            mutation_type = np.random.choice([2])

            if mutation_type == 1:
                # change an edge
                # first pick up a node
                node_id = np.random.choice(3)
                node = compact[node_id]
                # pick up an edge id
                edge_id = np.random.choice(len(node[1:])) + 1
                # edge ops are in [identity, zero] ([0, 1])
                new_edge_op = int(not compact[node_id][edge_id])
                # apply the mutation
                compact[node_id][edge_id] = new_edge_op

            elif mutation_type == 2:
                # change an op
                node_id = np.random.choice(3)
                node = compact[node_id]
                op_id = node[0]
                list_of_ops_ids = list(range(len(OP_NAMES)))
                list_of_ops_ids.remove(op_id)
                new_op_id = random.choice(list_of_ops_ids)
                compact[node_id][0] = new_op_id

        self.set_compact(compact)

    def get_nbhd(self, dataset_api=None):
        """
        Return all neighbors of the architecture
        """
        compact = self.get_compact()
        # edges, ops, hiddens = compact
        nbhd = []
        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return "asr"

    def get_max_epochs(self):
        return 39

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        return encode_asr(self, encoding_type=encoding_type)


def _set_cell_edge_ops(edge, filters, use_norm):
    if use_norm and edge.head == 7:
        edge.data.set("op", core_ops.Identity())
        edge.data.finalize()
    elif edge.head % 2 == 0:  # Edge from intermediate node
        edge.data.set(
            "op",
            [
                LinearOP(filters, filters, dropout_rate=DROPOUT_RATE),
                core_ops.Zero(stride=1),
                ResBlockLinearOP(filters, dropout_rate=DROPOUT_RATE),
                LinearBottleneckOP(filters, filters * 2,
                                   dropout_rate=DROPOUT_RATE),
            ],
        )
    elif edge.tail % 2 == 0:  # Edge to intermediate node. Should always be Identity.
        edge.data.finalize()
    else:
        edge.data.set("op", [core_ops.Zero(stride=1), core_ops.Identity()])


if __name__ == "__main__":
    datasets = Path("/home/data")
    OP_NAMES = ["linear", "zero", "resblock", "linear_bottleneck"]

    with open("/home/table_nas/darts_cell.yaml") as f:
        config = CfgNode.load_cfg(f)

    torch.manual_seed(config.seed)

    DROPOUT_RATE = config.dropout_rate
    NUM_FILTERS = config.num_filters
    NUM_BLOCKS = config.num_blocks
    metrics = {}
    opts = (("darts", DARTSOptimizer),)
    for name_opt, opt_class in opts:
        for dataset_name in ["covtype", "higgs-small", "otto", "adult", "churn"]:
            ds_train = TabNasTorchDataset(datasets, dataset_name, "train")
            ds_test = TabNasTorchDataset(datasets, dataset_name, "test")
            NUM_FEATURES = ds_train.num_features
            NUM_CLASSES = ds_train.num_classes
            print(dataset_name, NUM_CLASSES, NUM_FEATURES)

            dataset = TabNasDataset(config, ds_train, ds_test)

            train_queue, valid_queue, test_queue, train_transform, valid_transform = (
                dataset.get_loaders()
            )

            search_space = TabNasSearchSpace()
            log_path = Path(config.save) / \
                f"dartscell_{name_opt}_{dataset_name}.log"

            if log_path.exists():
                os.remove(log_path)

            logger = setup_logger(str(log_path))
            print(log_path)
            logger.setLevel(logging.INFO)

            optimizer = opt_class(**config.search)
            optimizer.adapt_search_space(search_space, config.dataset)

            trainer = TabNasTrainer(optimizer, config)
            trainer.search()
            trainer.evaluate()

            results = parse_log(
                config.save + f"dartscell_{name_opt}_{dataset_name}.log"
            )

            metrics[f"{name_opt}_{dataset_name}"] = results

            print(results)

            handlers = logger.handlers[:]
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()

            with open("/home/experiments/metrics_dartscell.pickle", "wb") as f:
                pickle.dump(metrics, f)
