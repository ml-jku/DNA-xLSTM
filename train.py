"""
Main training entry point for pre-training and downstream fine-tuning.
"""

import os
import json
import random
import time
from functools import wraps
from typing import Callable, List, Sequence

import fsspec
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

import src
import src.models.nn.utils as U
import src.utils as utils
from src.dataloaders import SequenceDataset
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks

# Enable TensorFloat32 for speed optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Register OmegaConf resolvers
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver("min", lambda x, y: min([x, y]))

log = src.utils.train.get_logger(__name__)

class DummyExperiment:
    """Dummy experiment to handle logging when not in rank zero."""

    def nop(self, *args, **kwargs):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Decorator to return the real experiment on rank 0 and DummyExperiment otherwise."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that retries on failure and handles rank zero logging."""
        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                rank_zero_warn(
                    "A wandb run is already in progress; new instances will reuse this run. "
                    "Call `wandb.finish()` before instantiating `WandbLogger` if this is not desired."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                self._experiment = wandb._attach(attach_id)
            else:
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        log.error("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        log.warning(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # Define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric(
                        "*", step_metric="trainer/global_step", step_sync=True
                    )

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor to reduce memory and increase speed
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(config, logger=False)

        # Dataset initialization
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check configuration
        self._check_config()

        # Flags
        self._has_setup = False
        self._state = None
        self.val_loader_names = None
        self.test_loader_names = None

        # Initialize components
        self.encoder = None
        self.decoder = None
        self.model = None
        self.task = None
        self.loss = None
        self.loss_val = None
        self.metrics = None
        self.setup(in_init=True)

    def setup(self, stage=None, in_init=False):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        if not in_init:
            current_device = int(
                os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0))
            )
            torch.cuda.set_device(f"cuda:{current_device}")

        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Combine encoders and decoders from model and task configurations
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        config_path = self.hparams.model.pop("config_path", None)
        if config_path is not None:
            with open(config_path) as f:
                model_config_from_file = json.load(f)
            self.hparams.model.update(model_config_from_file)
            # Check if dropout_layer_norm is compiled
            try:
                from flash_attn.ops.layer_norm import dropout_add_layer_norm
            except ImportError:
                if self.hparams.model.get("fused_dropout_add_ln", None) is not None:
                    self.hparams.model.update({"fused_dropout_add_ln": False})

        # Handle special cases for certain models
        model_name = self.hparams.model.get("_name_", "")
        if "caduceus" in model_name or "xlstm" in model_name:
            OmegaConf.update(
                self.hparams.model.config,
                "complement_map",
                self.dataset.tokenizer.complement_map,
                force_add=True,
            )

        # Instantiate model configuration
        if (
            config_target := self.hparams.model.get("config", None)
        ) and config_target.get("_target_", None):
            model_hparams = OmegaConf.to_container(self.hparams.model, resolve=True)
            model_hparams["config"] = hydra.utils.instantiate(model_hparams["config"])
            self.model = utils.instantiate(registry.model, model_hparams)
        else:
            self.model = utils.instantiate(registry.model, self.hparams.model)

        # Post-initialization hook
        if (hook_name := self.hparams.train.post_init_hook.get("_name_")) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs["_name_"]
            for module in self.modules():
                if hasattr(module, hook_name):
                    getattr(module, hook_name)(**kwargs)

        if self.hparams.train.get("compile_model", False):
            self.model = torch.compile(self.model, dynamic=False)

        # Instantiate task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Combine encoders and decoders
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = getattr(self.task, "loss_val", self.task.loss)
        self.metrics = self.task.metrics

    def load_state_dict(self, state_dict, strict=False):
        if self.hparams.train.pretrained_model_state_hook.get("_name_") is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            state_dict = model_state_hook(self.model, state_dict)

        log.info("Custom load_state_dict function is running.")
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        state_mode = self.hparams.train.state.mode
        assert state_mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        n_context = self.hparams.train.state.n_context
        n_context_eval = self.hparams.train.state.n_context_eval
        assert n_context is None or (isinstance(n_context, int) and n_context >= 0)
        assert n_context_eval is None or (
            isinstance(n_context_eval, int) and n_context_eval >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, (tuple, list)):
            return type(state)(self._detach_state(s) for s in state)
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, training=True):
        """Handle state context logic."""
        key = "n_context" if training else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        if n_context == 0 and self.hparams.train.state.mode not in ["tbptt"]:
            self._initialize_state()
            return

        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]
        elif self.hparams.train.state.mode == "tbptt":
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def forward(self, batch):
        return self.task.forward(
            batch, self.encoder, self.model, self.decoder, self._state
        )

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t)
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):
        self._process_state(batch, batch_idx, training=(prefix == "train"))
        x, y, w = self.forward(batch)
        
        # Compute loss
        loss_fn = self.loss if prefix == "train" else self.loss_val
        loss = loss_fn(x, y, **w)

        # handle rare exception where all entries in batch are pad tokens
        pad_token_idx = 4
        valid_mask = y.squeeze(-1) != pad_token_idx
        loss = loss[valid_mask]  # loss is unreduced cross entropy loss

        if loss.numel() == 0:
            # No valid loss entries, set loss to zero, metrics will not be logged for this step
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            metrics = {}
        else:
            # Metrics
            loss = loss.mean()  # Compute mean loss over valid entries
            metrics = self.metrics(x, y, **w)
            metrics["loss"] = loss
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        log_on_step = (
            "eval" in self.hparams
            and self.hparams.eval.get("log_on_step", False)
            and prefix == "train"
        )
        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def forward(self, batch):
        return self.task.forward(
            batch, self.encoder, self.model, self.decoder, self._state
        )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log loss and epoch
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": float(self.current_epoch)}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ema = (
            self.val_loader_names[dataloader_idx].endswith("/ema")
            and self.optimizers().optimizer.stepped
        )
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if "optimizer_param_grouping" in self.hparams.train:
            add_optimizer_hooks(
                self.model, **self.hparams.train.optimizer_param_grouping
            )

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(
            registry.optimizer, self.hparams.optimizer, params
        )
        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(set(frozenset(hp.items()) for hp in hps))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        # Layer Decay
        if self.hparams.train.layer_decay.get("_name_") is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay["_name_"],
                partial=True,
            )

            # Group parameters by layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                layer_id = get_num_layer(name)
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        "params": [],
                        "lr": None,
                        "weight_decay": self.hparams.optimizer.weight_decay,
                    }
                layer_wise_groups[layer_id]["params"].append(p)
                num_max_layers = max(num_max_layers, layer_id)

            # Update learning rates for each layer
            for layer_id, group in layer_wise_groups.items():
                group["lr"] = self.hparams.optimizer.lr * (
                    self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id)
                )

            # Reset optimizer parameter groups
            optimizer.param_groups = []
            for group in layer_wise_groups.values():
                optimizer.add_param_group(group)

        # Log optimizer info for debugging
        keys = set(k for hp in hps for k in hp.keys())
        utils.train.log_optimizer(log, optimizer, keys)

        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer

        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        log.info("Creating train loader")
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders."""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        """Return all validation and test loaders."""
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for EMA
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders += val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders += test_loaders

        # Optionally remove loaders
        eval_loader_names = []
        eval_loaders = []
        if not self.hparams.train.get("remove_val_loader_in_eval", False):
            eval_loader_names += val_loader_names
            eval_loaders += val_loaders
        if not self.hparams.train.get("remove_test_loader_in_eval", False):
            eval_loader_names += test_loader_names
            eval_loaders += test_loaders
        return eval_loader_names, eval_loaders

    def val_dataloader(self):
        self.val_loader_names, val_loaders = self._eval_dataloaders()
        log.info("Creating validation loaders")
        log.info(self.val_loader_names)
        return val_loaders

    def test_dataloader(self):
        self.test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in self.test_loader_names]
        log.info("Creating test loaders")
        log.info(self.test_loader_names)
        return test_loaders


def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Progressive Resizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        log.info(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            log.info(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Configure DDP automatically
    n_devices = config.trainer.get("devices", 1)
    if isinstance(n_devices, Sequence):
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get("strategy", None) is None:
        config.trainer.strategy = dict(
            _target_="pytorch_lightning.strategies.DDPStrategy",
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    # Instantiate trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )
    return trainer


def fsspec_exists(filename):
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Load pretrained model if specified
    if config.train.get("pretrained_model_path", None) is not None:
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )

    # Initial validation
    if config.train.validate_at_start:
        log.info("Running validation before training")
        trainer.validate(model)

    log.info(f"{config.train.ckpt=} {fsspec_exists(config.train.ckpt)=}")
    if config.train.ckpt is not None and fsspec_exists(config.train.ckpt):
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)

    if config.train.test:
        if config.train.get("cross_validation", False):  # First, load the best validation model
            best_val_ckpt = os.path.join(
                model.hparams.callbacks.model_checkpoint.dirpath,
                f"{model.hparams.callbacks.model_checkpoint.filename}.ckpt",
            )
            # Update config so we do not load just the backbone
            config.train.pretrained_model_state_hook.update({"_name_": None})
            # Remove validation loader
            config.train.update({"remove_val_loader_in_eval": True})
            config.train.update({"remove_test_loader_in_eval": False})
            ckpt = torch.load(best_val_ckpt)
            log.info(f"Loaded best validation checkpoint from epoch {ckpt['epoch']}")
            log.info("Testing approaching ...")
            trainer.validate(model, ckpt_path=best_val_ckpt)
        else:
            trainer.validate(model)
            log.info("Testing approaching ...")


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    # Process config
    config = utils.train.process_config(config)

    # Pretty print config
    utils.train.print_config(config, resolve=True)

    train(config)


if __name__ == "__main__":
    main()
